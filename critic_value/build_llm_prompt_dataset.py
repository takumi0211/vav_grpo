#!/usr/bin/env python3
"""Rebuild critic_value/data/dataset.csv directly from an Ollama llm_actions_log.

The log already contains the exact prompts shown to the model. This script:
1. Parses each prompt's observation JSON.
2. Reconstructs the raw critic observation vector in PolicyController order.
3. Normalizes it with the TD3 checkpoint's ObservationNormalizer.
4. Writes both the raw and normalized vectors (JSON encoded) alongside the prompt.

By default it keeps only rows from 2025-07-29, matching the run date in
outputs/ollama/llm_actions_log.csv.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    import pathlib
    import sys

    _project_root = pathlib.Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from critic_value.common import (  # noqa: E402
    critic_input_columns,
    derive_dims,
    load_td3_checkpoint,
)
from critic_value.prompt_utils import compose_prompt_from_agent, sanitize_observation  # noqa: E402

@dataclass
class ParsedObservation:
    timestamp_iso: str
    payload: dict[str, Any]


class PromptParseError(ValueError):
    pass


def _extract_observation_from_prompt(prompt: str) -> ParsedObservation:
    marker = "Observation for the next 5-minute HVAC control interval:"
    start = prompt.find(marker)
    if start == -1:
        raise PromptParseError("Observation marker not found in prompt")
    brace_start = prompt.find("{", start)
    if brace_start == -1:
        raise PromptParseError("Opening brace for observation JSON not found")
    depth = 0
    end = None
    for idx in range(brace_start, len(prompt)):
        ch = prompt[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        raise PromptParseError("Could not find matching closing brace for observation JSON")
    block = prompt[brace_start:end]
    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise PromptParseError(f"Failed to parse observation JSON: {exc}") from exc
    timestamp_iso = str(data.get("timestamp", "")).strip()
    return ParsedObservation(timestamp_iso=timestamp_iso, payload=data)


def _minute_of_day(ts: datetime) -> float:
    return float(ts.hour * 60 + ts.minute + ts.second / 60.0)


def _build_state_vector(obs: dict[str, Any], *, config) -> np.ndarray:
    zones_data = obs.get("zones") or []
    if not isinstance(zones_data, list) or not zones_data:
        raise ValueError("Observation missing 'zones' array")
    zone_lookup = {str(z.get("name", "")).strip().lower(): z for z in zones_data}

    ts_str = str(obs.get("timestamp", ""))
    if not ts_str:
        raise ValueError("Observation missing timestamp")
    timestamp = datetime.fromisoformat(ts_str)

    values: List[float] = []
    for zone_cfg in config.zones:
        key = str(zone_cfg.name).strip().lower()
        zone = zone_lookup.get(key)
        if zone is None:
            raise ValueError(f"Zone '{zone_cfg.name}' missing from observation")
        temp_error = float(zone.get("temp_error_c", 0.0))
        temp_delta = float(zone.get("temp_delta_c_per_min", 0.0))
        co2_ppm = float(zone.get("co2_ppm", config.co2_target_ppm))
        co2_error = co2_ppm - float(config.co2_target_ppm)
        co2_delta = float(zone.get("co2_delta_ppm_per_min", 0.0))
        values.extend([temp_error, temp_delta, co2_error, co2_delta])

    outdoor = obs.get("outdoor", {})
    values.append(float(outdoor.get("temperature_c", 0.0)))
    values.append(float(outdoor.get("temperature_trend_c_per_hour", 0.0)))

    minute = _minute_of_day(timestamp)
    angle = 2.0 * math.pi * (minute / 1440.0)
    values.append(math.sin(angle))
    values.append(math.cos(angle))

    prev = obs.get("previous_action", {})
    zone_dampers = prev.get("zone_dampers", [])
    if not isinstance(zone_dampers, list) or len(zone_dampers) != len(config.zones):
        raise ValueError("previous_action.zone_dampers does not match zone count")
    values.extend(float(v) for v in zone_dampers)
    values.append(float(prev.get("oa_damper", 0.0)))
    values.append(float(prev.get("coil_valve", 0.0)))
    values.append(float(prev.get("fan_speed", 0.0)))

    return np.asarray(values, dtype=np.float32)


def _normalize(obs: np.ndarray, normalizer) -> np.ndarray:
    return normalizer.normalize(obs, update=False)


def _ensure_checkpoint(path: Path) -> Path:
    if path.exists():
        return path
    fallback = Path("outputs/rl/td3_policy_final.pt")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"TD3 checkpoint not found: {path}")


def _load_logs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("No log files loaded")
    df = pd.concat(frames, ignore_index=True)
    if "timestamp_iso" not in df.columns:
        raise ValueError("Log CSV missing 'timestamp_iso' column")
    df["timestamp_iso"] = pd.to_datetime(df["timestamp_iso"], errors="coerce")
    df = df.dropna(subset=["timestamp_iso", "prompt"])
    df = df.sort_values("timestamp_iso").reset_index(drop=True)
    return df


def build_dataset(
    *,
    log_paths: List[Path],
    checkpoint_path: Path,
    date_filter: str | None,
    output_path: Path,
) -> pd.DataFrame:
    checkpoint_path = _ensure_checkpoint(checkpoint_path)
    device = torch.device("cpu")
    config, normalizer, _ = load_td3_checkpoint(checkpoint_path, device=device)
    obs_cols, _ = critic_input_columns(config)
    obs_dim, _ = derive_dims(config)
    if len(obs_cols) != obs_dim:
        raise RuntimeError("Mismatch between derived obs_dim and critic_input_columns")

    df = _load_logs(log_paths)
    if date_filter:
        mask = df["timestamp_iso"].dt.strftime("%Y-%m-%d") == date_filter
        df = df[mask]
    if df.empty:
        raise ValueError("No log rows remain after applying filters")

    rows: List[dict[str, Any]] = []
    for sample_id, row in enumerate(df.itertuples(index=False)):
        raw_prompt = str(getattr(row, "prompt", ""))
        try:
            parsed = _extract_observation_from_prompt(raw_prompt)
        except PromptParseError as exc:
            raise ValueError(f"Failed to parse prompt for timestamp {row.timestamp_iso}: {exc}") from exc
        clean_state = sanitize_observation(parsed.payload)
        prompt_text = compose_prompt_from_agent(clean_state)
        state_raw = _build_state_vector(parsed.payload, config=config)
        if state_raw.shape[0] != obs_dim:
            raise ValueError(
                f"State vector length {state_raw.shape[0]} does not match expected obs_dim {obs_dim}"
            )
        state_scaled = _normalize(state_raw.copy(), normalizer)
        rows.append(
            {
                "sample_id": sample_id,
                "timestamp": parsed.timestamp_iso or row.timestamp_iso.isoformat(),
                "state_json": json.dumps(state_scaled.tolist()),
                "state_raw_json": json.dumps(state_raw.tolist()),
                "prompt": prompt_text,
            }
        )

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset.csv from an Ollama llm_actions_log.csv")
    parser.add_argument(
        "--log",
        nargs="+",
        type=Path,
        default=[Path("outputs/ollama/llm_actions_log.csv")],
        help="Path(s) to llm_actions_log.csv files (default: outputs/ollama/llm_actions_log.csv)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("critic_value/td3_policy_final.pt"),
        help="TD3 checkpoint containing the observation normalizer",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2025-07-29",
        help="YYYY-MM-DD filter; keep only rows matching this date (default: 2025-07-29)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("critic_value/data/dataset.csv"),
        help="Destination CSV path (default: critic_value/data/dataset.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataset(
        log_paths=list(args.log),
        checkpoint_path=args.checkpoint,
        date_filter=args.date,
        output_path=args.output,
    )
    print(f"Wrote {len(df)} samples to {args.output}")


if __name__ == "__main__":
    main()
