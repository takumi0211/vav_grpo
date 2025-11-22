#!/usr/bin/env python3
"""Create a small synthetic [state + action] dataset for the TD3 critic.

The script samples plausible observations based on the checkpoint's running
normalizer statistics, injects randomized previous actions, and writes
`data/dataset.csv` under `critic_value/` by default.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    _project_root = pathlib.Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from critic_value.common import (  # type: ignore  # noqa: E402
    critic_input_columns,
    derive_dims,
    load_td3_checkpoint,
)
from simulator_humid.agents.rl.training_td3 import build_action_scaler  # type: ignore  # noqa: E402
from simulator_humid.utils.paths import RL_OUTPUT_DIR  # type: ignore  # noqa: E402


def sample_row(
    *,
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    action_dim: int,
    scaler,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one observation/action pair consistent with training layout."""

    obs = rng.normal(loc=obs_mean, scale=np.sqrt(obs_var + 1e-6)).astype(np.float32)

    tanh_prev = np.clip(rng.normal(0.0, 0.35, size=action_dim), -0.95, 0.95).astype(np.float32)
    scaled_prev = scaler.scale_action(torch.from_numpy(tanh_prev)).cpu().numpy()
    obs[-action_dim:] = scaled_prev

    tanh_action = np.clip(rng.normal(0.0, 0.45, size=action_dim), -0.95, 0.95).astype(np.float32)
    return obs, tanh_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a toy dataset for the TD3 critic.")
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=RL_OUTPUT_DIR / "td3_policy_final.pt",
        help="Path to TD3 checkpoint (default: outputs/rl/td3_policy_final.pt)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "data" / "dataset.csv",
        help="Destination CSV file (default: critic_value/data/dataset.csv)",
    )
    parser.add_argument("--rows", type=int, default=12, help="Number of unique observations to generate (default: 12)")
    parser.add_argument(
        "--variants-per-state",
        type=int,
        default=1,
        help="Number of action variants to pair with each observation (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    device = torch.device("cpu")
    config, normalizer, checkpoint = load_td3_checkpoint(args.checkpoint, device=device)
    obs_dim, action_dim = derive_dims(config)
    obs_cols, action_cols = critic_input_columns(config)

    obs_state = normalizer.state_dict()
    obs_mean = np.asarray(obs_state["mean"], dtype=np.float32)
    obs_var = np.asarray(obs_state["var"], dtype=np.float32)

    if obs_mean.shape[0] != obs_dim:
        raise RuntimeError(f"obs_mean length {obs_mean.shape[0]} != expected {obs_dim}")

    scaler = build_action_scaler(config)
    rng = np.random.default_rng(args.seed)

    obs_rows = []
    action_rows = []
    sample_ids = []
    variants = []

    n_obs = max(1, args.rows)
    n_variants = max(1, args.variants_per_state)

    base_obs = []
    for _ in range(n_obs):
        obs_row, action_row = sample_row(
            obs_mean=obs_mean,
            obs_var=obs_var,
            action_dim=action_dim,
            scaler=scaler,
            rng=rng,
        )
        base_obs.append(obs_row)

    for sample_idx, obs_row in enumerate(base_obs):
        for variant_idx in range(n_variants):
            obs_rows.append(obs_row)
            _, action_row = sample_row(
                obs_mean=obs_mean,
                obs_var=obs_var,
                action_dim=action_dim,
                scaler=scaler,
                rng=rng,
            )
            action_rows.append(action_row)
            sample_ids.append(sample_idx)
            variants.append(variant_idx)

    obs_df = pd.DataFrame(obs_rows, columns=obs_cols)
    action_df = pd.DataFrame(action_rows, columns=action_cols)
    df = pd.concat([obs_df, action_df], axis=1)
    df.insert(0, "sample_id", np.asarray(sample_ids, dtype=np.int32))
    df.insert(1, "action_variant", np.asarray(variants, dtype=np.int32))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output} (obs={n_obs}, variants per obs={n_variants})")


if __name__ == "__main__":
    main()
