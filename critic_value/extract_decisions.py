#!/usr/bin/env python3
"""Run TD3 policy and export decision-step observations/actions for GRPO.

This keeps changes local to critic_value (no run_td3 modification required).

Usage:
    python critic_value/extract_decisions.py \
        --episodes 1 \
        --checkpoint outputs/rl/td3_policy_final.pt \
        --weather weather_data/outdoor_temp_20250729.csv \
        --output critic_value/data/decisions_day1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

# make project importable when run as script
if __package__ in (None, ""):
    import sys

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from simulator_humid.agents.rl.run_td3 import (
    load_td3_policy,
    resolve_device,
)
from simulator_humid.agents.rl.training_td3 import (
    TrainingConfig,
    PolicyController,
    build_action_scaler,
    build_simulation_kwargs,
)
from simulator_humid.simulation import run_simulation
from critic_value.common import critic_input_columns  # type: ignore  # noqa: E402


def _date_from_weather(path: Path) -> str | None:
    """Extract YYYYMMDD from weather filename like outdoor_temp_20250726.csv."""
    stem = path.stem
    for token in stem.split("_"):
        if token.isdigit() and len(token) == 8:
            return token
    return None


def run_one_episode(checkpoint: Path, weather: Path, device: torch.device, episode_id: int) -> pd.DataFrame:
    import sys

    # Ensure TrainingConfig is discoverable for pickled checkpoints expecting __main__.TrainingConfig
    sys.modules.setdefault("__main__", sys.modules[__name__])
    setattr(sys.modules["__main__"], "TrainingConfig", TrainingConfig)

    actor, normalizer, config = load_td3_policy(checkpoint_path=checkpoint, device=device, weather_csv=weather)

    # Override start_time date to match the weather file (keeps HH:MM at 00:00)
    date_token = _date_from_weather(weather)
    if date_token:
        from datetime import datetime

        y, m, d = int(date_token[:4]), int(date_token[4:6]), int(date_token[6:8])
        config.start_time = datetime(y, m, d, 0, 0)

    scaler = build_action_scaler(config).to(device)
    controller = PolicyController(
        policy=actor,
        scaler=scaler,
        config=config,
        device=device,
        normalizer=normalizer,
        update_normalizer=False,
        exploration_noise=0.0,
    )
    sim_kwargs = build_simulation_kwargs(config)
    with torch.no_grad():
        _df = run_simulation(action_callback=controller, verbose_steps=False, **sim_kwargs)

    traj = controller.trajectory()
    obs = np.asarray(traj.get("obs"), dtype=np.float32)
    acts = np.asarray(traj.get("tanh_actions"), dtype=np.float32)
    timestamps: List = traj.get("timestamps", [])
    decision_mask = np.asarray(traj.get("decision_mask"), dtype=bool)

    if obs.shape[0] != decision_mask.shape[0] or acts.shape[0] != decision_mask.shape[0]:
        raise RuntimeError("trajectory shapes mismatch")

    # filter to decision steps only
    obs_dec = obs[decision_mask]
    acts_dec = acts[decision_mask]
    ts_dec = [timestamps[i] for i, flag in enumerate(decision_mask) if flag] if timestamps else []

    zone_count = len(config.zones)
    obs_cols, act_cols = critic_input_columns(config)

    # Reorder observation columns so that each zone's temp/CO2 features stay grouped.
    obs_data: dict[str, np.ndarray] = {}
    # Offsets for each block in the original observation vector.
    block = {
        "temp_error": 0,
        "temp_delta": zone_count,
        "co2_error": 2 * zone_count,
        "co2_delta": 3 * zone_count,
    }
    for idx in range(zone_count):
        obs_data[f"zone_{idx+1}_temp_error"] = obs_dec[:, block["temp_error"] + idx]
        obs_data[f"zone_{idx+1}_temp_delta"] = obs_dec[:, block["temp_delta"] + idx]
        obs_data[f"zone_{idx+1}_co2_error"] = obs_dec[:, block["co2_error"] + idx]
        obs_data[f"zone_{idx+1}_co2_delta"] = obs_dec[:, block["co2_delta"] + idx]

    global_offset = 4 * zone_count
    obs_data["outdoor_temp"] = obs_dec[:, global_offset]
    obs_data["outdoor_temp_slope"] = obs_dec[:, global_offset + 1]
    obs_data["sin_time"] = obs_dec[:, global_offset + 2]
    obs_data["cos_time"] = obs_dec[:, global_offset + 3]

    prev_offset = global_offset + 4
    for idx in range(zone_count):
        obs_data[f"prev_zone_{idx+1}_damper"] = obs_dec[:, prev_offset + idx]
    prev_offset += zone_count
    obs_data["prev_oa_damper"] = obs_dec[:, prev_offset]
    obs_data["prev_coil_valve"] = obs_dec[:, prev_offset + 1]
    obs_data["prev_fan_speed"] = obs_dec[:, prev_offset + 2]

    df_obs = pd.DataFrame(obs_data, columns=[c for c in obs_cols if c in obs_data])
    df_act = pd.DataFrame(acts_dec, columns=act_cols)
    df = pd.concat([df_obs, df_act], axis=1)
    df.insert(0, "step", np.arange(len(df), dtype=np.int32))
    df.insert(1, "decision_step", 1)
    if ts_dec:
        df.insert(2, "timestamp", pd.to_datetime(ts_dec))
    df.insert(0, "episode_id", episode_id)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract decision-step obs/actions by running TD3 policy")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/rl/td3_policy_final.pt"))
    parser.add_argument("--weather", type=Path, default=Path("weather_data/outdoor_temp_20250729.csv"))
    parser.add_argument("--output", type=Path, default=Path("critic_value/data/decisions.csv"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dfs: List[pd.DataFrame] = []
    for ep in range(args.episodes):
        dfs.append(run_one_episode(args.checkpoint, args.weather, device, ep))

    out = pd.concat(dfs, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows ({len(dfs)} episodes) to {args.output}")


if __name__ == "__main__":
    main()
