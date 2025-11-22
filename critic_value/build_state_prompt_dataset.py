#!/usr/bin/env python3
"""Build a lightweight dataset of states + prompts for LLM GRPO.

Reads one or more decision-step CSVs (no action needed at training time),
samples a fixed number of states per day/episode, and emits a CSV containing:
  - sample_id
  - episode_id (if available)
  - timestamp (if available)
  - state_json : JSON-encoded observation vector (order matches critic input)
  - prompt     : templated instruction for the LLM to output actions

Default: 25 states per source * 4 sources = 100 rows.

Usage example:
  python critic_value/build_state_prompt_dataset.py \
      --sources critic_value/data/decisions_all.csv \
      --per-day 25 \
      --output critic_value/data/dataset.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

# ensure repository root is on sys.path when executed as a script
if __package__ in (None, ""):
    import pathlib
    import sys

    _project_root = pathlib.Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd

from critic_value.dataset_utils import (
    drop_action_columns,
    filter_decision_steps,
    iter_episode_groups,
    sample_evenly,
)
from critic_value.prompt_utils import build_state_dict, compose_prompt_from_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Create state+prompt dataset for LLM GRPO")
    parser.add_argument("--sources", nargs="+", type=Path, required=True, help="CSV files with decision_step rows")
    parser.add_argument("--per-day", type=int, default=25, help="states to sample per source")
    parser.add_argument("--output", type=Path, default=Path("critic_value/data/dataset.csv"))
    args = parser.parse_args()

    rows: List[pd.Series] = []
    for src in args.sources:
        df = pd.read_csv(src)
        df = filter_decision_steps(df)
        df = drop_action_columns(df)

        for group in iter_episode_groups(df):
            sampled = sample_evenly(group, args.per_day)
            for _, base in sampled.iterrows():
                row = base.copy()
                meta_cols = {"sample_id", "variant_id", "variant_name", "episode_id", "timestamp", "step", "decision_step"}
                obs_cols = [c for c in row.index if c not in meta_cols]
                row["state_json"] = json.dumps([float(row[c]) for c in obs_cols])
                row["prompt"] = compose_prompt_from_agent(build_state_dict(row))
                rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.insert(0, "sample_id", np.arange(len(out_df), dtype=np.int32))

    # keep only minimal columns
    keep = [c for c in ["sample_id", "episode_id", "timestamp", "state_json", "prompt"] if c in out_df.columns]
    out_df = out_df[keep]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
