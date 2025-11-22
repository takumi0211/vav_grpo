#!/usr/bin/env python3
"""Rebuild a cleaned prompt/state dataset excluding unrealistic temperatures.

Rules:
- Only rows where every |zone_i_temp_error| <= 20 are kept (setpoint=26 => temps in [6, 46]).
- Optionally tighten via --temp-threshold if needed.
- Uses the same prompt/state construction as build_state_prompt_dataset.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

# ensure repo root on sys.path
if __package__ in (None, ""):
    import pathlib
    import sys

    _project_root = pathlib.Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from critic_value.dataset_utils import (  # noqa: E402
    drop_action_columns,
    filter_decision_steps,
    iter_episode_groups,
    sample_evenly,
)
from critic_value.prompt_utils import build_state_dict, compose_prompt_from_agent  # noqa: E402


def _filter_realistic(df: pd.DataFrame, temp_threshold: float) -> pd.DataFrame:
    zone_cols = [c for c in df.columns if c.startswith("zone_") and c.endswith("_temp_error")]
    mask = pd.Series(True, index=df.index)
    for col in zone_cols:
        mask &= df[col].abs() <= temp_threshold
    return df[mask].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a cleaned prompt/state dataset.")
    parser.add_argument("--sources", nargs="+", type=Path, default=[Path("critic_value/data/decisions_all.csv")])
    parser.add_argument("--per-day", type=int, default=25, help="states to sample per source")
    parser.add_argument("--temp-threshold", type=float, default=20.0, help="max |temp_error| to keep")
    parser.add_argument("--output", type=Path, default=Path("critic_value/data/dataset.csv"))
    args = parser.parse_args()

    rows: List[dict[str, Any]] = []
    sample_id = 0
    for src in args.sources:
        df = pd.read_csv(src)
        df = filter_decision_steps(df)
        df = drop_action_columns(df)
        df = _filter_realistic(df, args.temp_threshold)

        for group in iter_episode_groups(df):
            sampled = sample_evenly(group, args.per_day)
            for _, base in sampled.iterrows():
                row = base.copy()
                meta_cols = {"sample_id", "variant_id", "variant_name", "episode_id", "timestamp", "step", "decision_step"}
                obs_cols = [c for c in row.index if c not in meta_cols]
                state_list = [float(row[c]) for c in obs_cols]
                state_json = json.dumps(state_list)
                state_dict = build_state_dict(row)
                prompt = compose_prompt_from_agent(state_dict)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "episode_id": row.get("episode_id", np.nan),
                        "timestamp": row.get("timestamp", ""),
                        "state_json": state_json,
                        "prompt": prompt,
                    }
                )
                sample_id += 1

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote cleaned dataset with {len(out_df)} samples to {args.output}")


if __name__ == "__main__":
    main()
