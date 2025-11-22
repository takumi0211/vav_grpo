#!/usr/bin/env python3
"""Shared helpers for TD3 critic value inspection utilities.

This module centralises checkpoint loading and column name construction so
`generate_dataset.py` and `evaluate_critic.py` stay in sync.
"""

from __future__ import annotations

import os
import pathlib
import pickle
import re
import sys
import types
from dataclasses import replace
from typing import Iterable, Sequence, Tuple

import torch

if __package__ in (None, ""):
    _project_root = pathlib.Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from simulator_humid.agents.rl.training_td3 import (  # type: ignore  # noqa: E402
    ObservationNormalizer,
    TrainingConfig,
)
from simulator_humid.utils.paths import RL_OUTPUT_DIR  # type: ignore  # noqa: E402

# Windows で保存された checkpoint を macOS/Linux で読み込む際の互換性対策
if os.name != "nt":
    pathlib.WindowsPath = pathlib.PosixPath


class _PathCompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: N802
        if module.startswith("pathlib") and "WindowsPath" in name:
            return pathlib.PosixPath
        if module == "__main__" and name == "TrainingConfig":
            return TrainingConfig
        return super().find_class(module, name)


def default_device(name: str | None = None) -> torch.device:
    """Resolve a torch.device, preferring the requested name when valid."""

    if name:
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_td3_checkpoint(
    checkpoint_path: pathlib.Path | None,
    *,
    device: torch.device,
) -> tuple[TrainingConfig, ObservationNormalizer, dict]:
    """Load TD3 checkpoint, returning config, normalizer, and raw dict."""

    path = checkpoint_path or RL_OUTPUT_DIR / "td3_policy_final.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    pickle_module = types.ModuleType("pickle")
    pickle_module.Unpickler = _PathCompatibleUnpickler

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
        pickle_module=pickle_module,
    )

    if "config" not in checkpoint or "obs_normalizer" not in checkpoint:
        raise ValueError("Checkpoint is missing required keys ('config', 'obs_normalizer').")

    raw_config: TrainingConfig = checkpoint["config"]
    config = replace(raw_config)

    obs_state = checkpoint["obs_normalizer"]
    obs_dim = int(len(obs_state["mean"]))
    normalizer = ObservationNormalizer(obs_dim, clip=config.obs_norm_clip, eps=config.obs_norm_eps)
    normalizer.load_state_dict(obs_state)

    return config, normalizer, checkpoint


def derive_dims(config: TrainingConfig) -> tuple[int, int]:
    """Return (obs_dim, action_dim) for the current zone configuration."""

    zone_count = len(config.zones)
    action_dim = zone_count + 3
    obs_dim = zone_count * 4 + 4 + action_dim
    return obs_dim, action_dim


def _safe_zone_name(name: str, fallback_idx: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return base or f"zone{fallback_idx}"


def critic_input_columns(config: TrainingConfig) -> tuple[list[str], list[str]]:
    """Construct ordered column names for obs and action tensors.

    Obs layout mirrors `PolicyController.build_observation`:
    temp/CO2 errors & deltas per zone, four global features, then previous
    (scaled) actions. Action layout expects current tanh actions.
    """

    zone_keys = [_safe_zone_name(z.name, idx + 1) for idx, z in enumerate(config.zones)]

    obs_cols: list[str] = []
    for key in zone_keys:
        obs_cols.extend(
            [
                f"{key}_temp_error",
                f"{key}_temp_delta",
                f"{key}_co2_error",
                f"{key}_co2_delta",
            ]
        )

    obs_cols.extend(["outdoor_temp", "outdoor_temp_slope", "sin_time", "cos_time"])
    obs_cols.extend([f"prev_{key}_damper" for key in zone_keys])
    obs_cols.extend(["prev_oa_damper", "prev_coil_valve", "prev_fan_speed"])

    action_cols = [f"action_{key}_tanh" for key in zone_keys]
    action_cols.extend(["action_oa_tanh", "action_coil_tanh", "action_fan_tanh"])

    return obs_cols, action_cols


def ensure_columns(df_columns: Iterable[str], required: Sequence[str]) -> list[str]:
    """Return missing columns to help with user-friendly validation."""

    df_set = set(df_columns)
    return [col for col in required if col not in df_set]
