"""LLM sampling + critic evaluation helpers shared by CLI scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass
class LLMDecisionResult:
    """Container for one LLM sample and metadata."""

    action_vector: np.ndarray
    thought_process: str
    action_payload: dict[str, Any]
    raw_response_text: str
    model_thinking: str = ""
    error: str | None = None


def pick_row(df: pd.DataFrame, sample_id: int | None, row_index: int | None) -> pd.Series:
    """Return a row by sample_id when present, otherwise by positional index."""

    if sample_id is not None and "sample_id" in df.columns:
        matches = df[df["sample_id"] == sample_id]
        if matches.empty:
            raise ValueError(f"sample_id {sample_id} not found in dataset")
        return matches.iloc[0]
    idx = row_index if row_index is not None else 0
    if idx < 0 or idx >= len(df):
        raise ValueError(f"Row index {idx} outside dataset bounds 0..{len(df)-1}")
    return df.iloc[idx]


def _parse_vector_column(value: Any) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return np.asarray(json.loads(text), dtype=np.float32)


def resolve_observation_vector(
    row: pd.Series,
    *,
    normalizer,
    obs_dim: int,
) -> np.ndarray:
    """Return a normalized observation vector for the selected dataset row."""

    raw_vec = _parse_vector_column(row.get("state_raw_json"))
    stored_vec = _parse_vector_column(row.get("state_json"))

    if raw_vec is not None:
        if stored_vec is None:
            raise ValueError("Row has state_raw_json but no normalized state_json.")
        obs_vec_norm = stored_vec
    else:
        if stored_vec is None:
            raise ValueError("Row is missing state_json values.")
        obs_vec_norm = normalizer.normalize(stored_vec.copy(), update=False)

    if obs_vec_norm.shape[0] != obs_dim:
        raise ValueError(f"Observation length {obs_vec_norm.shape[0]} != expected {obs_dim}.")
    return obs_vec_norm


def build_action_bounds(config) -> tuple[np.ndarray, np.ndarray]:
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


def evaluate_candidate(
    *,
    critic,
    scaler,
    device: torch.device,
    obs_tensor_norm: torch.Tensor,
    action_vec_scaled: np.ndarray,
) -> dict[str, float]:
    """Return q1/q2/q_min/q_mean for one (state, action) pair."""

    scaled_tensor = torch.from_numpy(action_vec_scaled.astype(np.float32, copy=False)).to(device)
    tanh_tensor = scaler.unscale_action(scaled_tensor).unsqueeze(0)

    with torch.no_grad():
        q1, q2 = critic(obs_tensor_norm, tanh_tensor)
    q1_val = float(q1.squeeze(1).item())
    q2_val = float(q2.squeeze(1).item())
    q_min = float(np.minimum(q1_val, q2_val))
    q_mean = 0.5 * (q1_val + q2_val)
    return {"q1": q1_val, "q2": q2_val, "q_min": q_min, "q_mean": q_mean}


def softmax_from_scores(values: np.ndarray, temperature: float) -> np.ndarray:
    """Stable softmax on arbitrary q-values (handles inf/NaN gracefully)."""

    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return values
    mask = np.isfinite(values)
    if not mask.any():
        return np.full(n, 1.0 / n, dtype=np.float64)
    finite_vals = values[mask]
    mean = float(np.mean(finite_vals))
    std = float(np.std(finite_vals))
    span = float(np.max(finite_vals) - np.min(finite_vals))
    scale = std if std > 1e-6 else max(span, 1.0)
    normalized = (values - mean) / scale
    normalized = np.clip(normalized, -30.0, 30.0)
    normalized[~mask] = -30.0
    normalized -= np.max(normalized)
    temp = float(max(temperature, 1e-6))
    normalized /= temp
    exp = np.exp(normalized)
    total = float(np.sum(exp))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return exp / total


def parse_json_payload(text: str) -> dict[str, Any]:
    """Parse JSON from model text; fall back to the first {...} block."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            return json.loads(snippet)
        raise ValueError(f"Could not parse JSON from model output: {text}")


def action_dict_to_vector(
    action: dict[str, Any],
    *,
    zone_count: int,
    low_bounds: np.ndarray,
    high_bounds: np.ndarray,
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
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid actuator value: {exc}") from exc
    vector = np.concatenate([zone_array, np.array([oa, coil, fan], dtype=np.float32)], axis=0)
    return np.clip(vector, low_bounds, high_bounds)

