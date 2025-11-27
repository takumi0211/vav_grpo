"""TD3 critic-based reward utilities for GRPO training.

This module replaces the former ``critic_value`` package with a compact,
lazy-loading helper that can score HVAC actions on the fly. It keeps all
heavy imports (simulator_humid, torch load) inside the class so simply
importing the module will not pull large dependencies until the reward model
is actually used.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import pickle
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ----------------------------- small helpers ----------------------------- #


def _default_device(name: str | None = None) -> torch.device:
    if name:
        dev = torch.device(name)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _PathCompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: N802
        # Checkpoint may have been saved on Windows
        if module.startswith("pathlib") and "WindowsPath" in name:
            return pathlib.PosixPath
        # When original module path is missing, map to our local stubs
        if module == "simulator_humid.agents.rl.training_td3":
            mod = sys.modules.get(module)
            if mod is None:
                mod = types.ModuleType(module)
                sys.modules[module] = mod
            # Populate expected symbols
            for cls in [TrainingConfig, ObservationNormalizer, TwinQNetwork, ActionScaler]:
                setattr(mod, cls.__name__, cls)
            mod.build_action_scaler = build_action_scaler
            if hasattr(mod, name):
                return getattr(mod, name)
            # If still missing, fall back to globals (defensive)
            if name in globals():
                return globals()[name]
            raise ModuleNotFoundError(f"Missing class {name} in shim module {module}")
        if module.startswith("simulator_humid"):
            dummy = type(name, (), {})
            return dummy
        if module == "__main__" and name == "TrainingConfig":
            return TrainingConfig
        return super().find_class(module, name)


def parse_json_payload(text: str) -> dict[str, Any]:
    """Parse JSON from text, falling back to the first {...} block."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            return json.loads(snippet)
        raise


def action_dict_to_vector(
    action: dict[str, Any], *, zone_count: int, low_bounds: np.ndarray, high_bounds: np.ndarray
) -> np.ndarray:
    """Convert JSON action payloads into clipped actuator vectors."""

    zone_values = action.get("zone_dampers")
    if not isinstance(zone_values, list) or len(zone_values) != zone_count:
        raise ValueError("zone_dampers must be a list matching the number of zones.")
    try:
        zone_array = np.array([float(v) for v in zone_values], dtype=np.float32)
        oa = float(action.get("oa_damper"))
        coil = float(action.get("coil_valve"))
        fan = float(action.get("fan_speed"))
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid actuator value: {exc}") from exc
    vector = np.concatenate([zone_array, np.array([oa, coil, fan], dtype=np.float32)], axis=0)
    return np.clip(vector, low_bounds, high_bounds)


def parse_state_vector(value: Any) -> np.ndarray | None:
    """Parse state vectors stored as JSON text, Python lists, or ndarrays."""

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):  # pragma: no cover - defensive
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32)
        return arr
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return np.asarray(parsed, dtype=np.float32)
    return None


def softmax_from_scores(values: Sequence[float], temperature: float) -> np.ndarray:
    """Stable softmax that ignores NaNs and extreme values."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr)
    if not mask.any():
        return np.full_like(arr, np.nan, dtype=np.float64)

    finite_vals = arr[mask]
    mean = float(np.mean(finite_vals))
    std = float(np.std(finite_vals))
    span = float(np.max(finite_vals) - np.min(finite_vals))
    scale = std if std > 1e-6 else max(span, 1.0)
    normed = (arr - mean) / scale
    normed = np.clip(normed, -30.0, 30.0)
    normed[~mask] = -30.0

    temp = float(max(temperature, 1e-6))
    normed /= temp
    normed -= np.max(normed)
    exp = np.exp(normed)
    total = float(np.sum(exp))
    if not np.isfinite(total) or total <= 0.0:
        probs = np.full_like(arr, np.nan, dtype=np.float64)
    else:
        probs = exp / total
    probs[~mask] = np.nan
    return probs


def derive_dims(config) -> Tuple[int, int]:
    """Return (obs_dim, action_dim) for the zone configuration."""

    zone_count = len(config.zones)
    action_dim = zone_count + 3
    obs_dim = zone_count * 4 + 4 + action_dim
    return obs_dim, action_dim


def build_action_bounds(config) -> Tuple[np.ndarray, np.ndarray]:
    """Construct low/high actuator bounds using the RL training config."""

    zone_count = len(config.zones)
    low = np.concatenate(
        [
            np.full(zone_count, config.damper_min, dtype=np.float32),
            np.array([config.oa_min, config.coil_min, config.fan_min], dtype=np.float32),
        ]
    )
    high = np.concatenate(
        [
            np.full(zone_count, config.damper_max, dtype=np.float32),
            np.array([config.oa_max, config.coil_max, config.fan_max], dtype=np.float32),
        ]
    )
    return low, high


# ----------------------- minimal TD3 components ------------------------ #


class TrainingConfig:
    """Lightweight stand-in for the original TrainingConfig.

    Only the attributes referenced during reward computation are kept.
    """

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class ObservationNormalizer(nn.Module):
    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.normalize_tensor(x)

    def normalize(self, arr: np.ndarray, update: bool = False) -> np.ndarray:
        if update:
            return arr  # training-time updates not needed here
        denom = np.sqrt(self.var.cpu().numpy() + self.eps)
        out = (arr - self.mean.cpu().numpy()) / denom
        return np.clip(out, -self.clip, self.clip)

    def normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        denom = torch.sqrt(self.var + self.eps)
        out = (x - self.mean) / denom
        return torch.clamp(out, -self.clip, self.clip)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        mean = torch.as_tensor(state_dict.get("mean", 0.0), dtype=torch.float32)
        var = torch.as_tensor(state_dict.get("var", 1.0), dtype=torch.float32)
        if mean.shape != self.mean.shape:
            mean = mean.view_as(self.mean)
        if var.shape != self.var.shape:
            var = var.view_as(self.var)
        self.mean.copy_(mean)
        self.var.copy_(var)
        self.clip = float(state_dict.get("clip", self.clip))
        self.eps = float(state_dict.get("eps", self.eps))
        # ignore "count" or other keys
        return None


class ActionScaler(nn.Module):
    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        super().__init__()
        low_t = torch.as_tensor(low, dtype=torch.float32)
        high_t = torch.as_tensor(high, dtype=torch.float32)
        self.register_buffer("low", low_t)
        self.register_buffer("high", high_t)

    def unscale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map physical action values to tanh space [-1, 1]."""

        # Clamp to bounds for safety
        act = torch.clamp(action, self.low, self.high)
        span = self.high - self.low
        scaled = (act - self.low) / torch.clamp(span, min=1e-6)
        return scaled * 2.0 - 1.0


class TwinQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        input_dim = obs_dim + action_dim
        # If checkpoint hidden sizes are provided, use them; otherwise match observed shapes [256,256,128]
        h_list = list(hidden_sizes) if hidden_sizes else [256, 256, 128]

        def _branch():
            layers: List[nn.Module] = []
            dims = [input_dim] + h_list
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))  # indices 0,3,6 ...
                layers.append(nn.LayerNorm(dims[i + 1]))        # indices 1,4,7 ...
                layers.append(nn.ReLU())                        # non-parametric (indices 2,5,8 ...)
            net = nn.Sequential(*layers)
            out = nn.Linear(dims[-1], 1)
            branch = nn.Module()
            branch.net = net  # type: ignore[attr-defined]
            branch.output = out  # type: ignore[attr-defined]
            return branch

        self.q1 = _branch()
        self.q2 = _branch()

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1.output(self.q1.net(x)), self.q2.output(self.q2.net(x))


def build_action_scaler(config: TrainingConfig) -> ActionScaler:
    low, high = build_action_bounds(config)
    return ActionScaler(low, high)


@dataclass
class ParsedSample:
    obs_vec: np.ndarray
    action_vec: np.ndarray
    action_payload: dict[str, Any]
    q_min: float | None = None
    q_mean: float | None = None
    error: str | None = None


class TD3RewardModel:
    """Lightweight wrapper around the TD3 critic checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_path: str | os.PathLike = "data/td3_policy_final.pt",
        device: str | None = None,
        assume_state_normalized: bool = False,
    ) -> None:
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.device = _default_device(device)
        self.assume_state_normalized = assume_state_normalized

        self.config = None
        self.normalizer = None
        self.critic = None
        self.scaler = None
        self.obs_dim = 0
        self.action_dim = 0
        self.low_bounds: np.ndarray | None = None
        self.high_bounds: np.ndarray | None = None
        self.zone_count = 0

        self._load_checkpoint()

    # ------------------------------ internals ----------------------------- #

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"TD3 checkpoint not found: {self.checkpoint_path}")

        pickle_module = types.ModuleType("pickle")
        pickle_module.Unpickler = _PathCompatibleUnpickler

        # Ensure the expected module path exists for unpickling
        if "simulator_humid.agents.rl.training_td3" not in sys.modules:
            shim = types.ModuleType("simulator_humid.agents.rl.training_td3")
            shim.TrainingConfig = TrainingConfig
            shim.ObservationNormalizer = ObservationNormalizer
            shim.TwinQNetwork = TwinQNetwork
            shim.build_action_scaler = build_action_scaler
            sys.modules["simulator_humid.agents.rl.training_td3"] = shim

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
            pickle_module=pickle_module,
        )

        if "config" not in checkpoint or "obs_normalizer" not in checkpoint:
            raise ValueError("Checkpoint is missing required keys ('config', 'obs_normalizer').")

        self.config: TrainingConfig = checkpoint["config"]
        self.obs_dim, self.action_dim = derive_dims(self.config)
        self.zone_count = len(self.config.zones)

        obs_state = checkpoint["obs_normalizer"]
        normalizer = ObservationNormalizer(
            self.obs_dim,
            clip=self.config.obs_norm_clip,
            eps=self.config.obs_norm_eps,
        )
        normalizer.load_state_dict(obs_state)
        self.normalizer = normalizer.to(self.device)

        chs = getattr(self.config, "critic_hidden_sizes", None)
        h_default = (256, 256)
        if chs is None or chs == []:
            chs = getattr(self.config, "hidden_sizes", None)
        if chs is None or chs == []:
            critic_hidden = h_default
        else:
            critic_hidden = tuple(chs)
        critic = TwinQNetwork(self.obs_dim, self.action_dim, critic_hidden).to(self.device)
        if "critic_state_dict" not in checkpoint:
            raise ValueError("Checkpoint missing 'critic_state_dict'.")
        critic.load_state_dict(checkpoint["critic_state_dict"])
        critic.eval()
        self.critic = critic

        scaler = build_action_scaler(self.config).to(self.device)
        self.scaler = scaler

        self.low_bounds, self.high_bounds = build_action_bounds(self.config)

    # --------------------------- parsing helpers -------------------------- #

    def _resolve_obs(self, state_json: Any, state_raw_json: Any) -> np.ndarray:
        vec_norm = parse_state_vector(state_json)
        vec_raw = parse_state_vector(state_raw_json)

        if vec_norm is not None:
            obs_vec = vec_norm
            if not self.assume_state_normalized and self.normalizer is not None:
                obs_vec = self.normalizer.normalize(obs_vec.copy(), update=False)
        elif vec_raw is not None and self.normalizer is not None:
            obs_vec = self.normalizer.normalize(vec_raw.astype(np.float32, copy=False), update=False)
        else:
            raise ValueError("Missing state_json/state_raw_json for reward computation.")

        if obs_vec.shape[0] != self.obs_dim:
            raise ValueError(f"Observation length {obs_vec.shape[0]} != expected {self.obs_dim}.")
        return obs_vec.astype(np.float32, copy=False)

    def parse_action_from_text(self, text: str) -> Tuple[np.ndarray, dict[str, Any]]:
        payload = parse_json_payload(text)
        action_obj = payload.get("action") if isinstance(payload, dict) else None
        if not isinstance(action_obj, dict):
            raise ValueError("Response is missing an 'action' object.")
        if self.low_bounds is None or self.high_bounds is None:
            raise RuntimeError("Action bounds are not initialized.")
        vector = action_dict_to_vector(
            action_obj,
            zone_count=self.zone_count,
            low_bounds=self.low_bounds,
            high_bounds=self.high_bounds,
        )
        return vector, action_obj

    # ----------------------------- scoring API ---------------------------- #

    def score_batch(self, samples: Iterable[ParsedSample]) -> List[ParsedSample]:
        bucket: List[ParsedSample] = []
        obs_list: List[torch.Tensor] = []
        action_list: List[torch.Tensor] = []

        for sample in samples:
            bucket.append(sample)
            obs_list.append(torch.from_numpy(sample.obs_vec))
            action_list.append(torch.from_numpy(sample.action_vec))

        if not bucket:
            return []

        obs_tensor = torch.stack(obs_list).to(self.device)
        if not self.assume_state_normalized and self.normalizer is not None:
            obs_tensor = self.normalizer.normalize_tensor(obs_tensor)

        action_tensor = torch.stack(action_list).to(self.device)

        with torch.no_grad():
            action_tanh = self.scaler.unscale_action(action_tensor)  # type: ignore[arg-type]
            q1, q2 = self.critic(obs_tensor, action_tanh)  # type: ignore[misc]

        q1_np = q1.squeeze(1).cpu().numpy()
        q2_np = q2.squeeze(1).cpu().numpy()
        q_mean = 0.5 * (q1_np + q2_np)

        for idx, sample in enumerate(bucket):
            # より安全側を使う: twin-Qの最小値
            sample.q_min = float(q1_np[idx])  # use q1 for rewards
            sample.q_mean = float(q_mean[idx])
        return bucket


def td3_model_from_env() -> TD3RewardModel:
    path = os.getenv("TD3_CHECKPOINT_PATH", "data/td3_policy_final.pt")
    device = os.getenv("TD3_CRITIC_DEVICE")
    # デフォルトで正規化を適用（state_jsonが既にスケール済みと思ってもズレが出るため）
    assume_norm = os.getenv("TD3_ASSUME_NORMALIZED", "0") != "0"
    return TD3RewardModel(checkpoint_path=path, device=device, assume_state_normalized=assume_norm)
