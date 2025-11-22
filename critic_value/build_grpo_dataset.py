#!/usr/bin/env python3
"""Build a GRPO-ready dataset: sample decision steps and attach action variants.

Config (feel free to tweak below):
  - days / files: specify multiple decision-step CSVs (one per roll-out day)
  - per day: take N decision steps (default 25) spaced roughly evenly
  - per step: generate V action variants (default 4) + base action = V+1 rows
Output: critic_value/data/dataset.csv (features + action variants) and
        critic_value/data/value.csv (with q1/q2/q_min/q_mean via evaluate_critic.py)

Run:
  python critic_value/build_grpo_dataset.py \
    --sources critic_value/data/decisions_day1.csv critic_value/data/decisions_day2.csv \
    --per-day 25 --variants 4

Prereqs:
  - Each source CSV must be decision_step==1 only (as produced by run_td3.py
    after filtering) and contain action_*_tanh columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from critic_value.dataset_utils import (
    ensure_sample_ids,
    filter_decision_steps,
    iter_episode_groups,
    sample_evenly,
)


def _build_variants() -> List[Dict[str, float]]:
    return [
        {"name": "base", "delta": {}},
        {
            "name": "vent_up",
            "delta": {
                "action_oa_tanh": +0.30,
                "action_fan_tanh": +0.20,
                "action_zone_1_tanh": +0.10,
                "action_zone_2_tanh": +0.10,
                "action_zone_3_tanh": +0.10,
                "action_zone_4_tanh": +0.10,
            },
        },
        {
            "name": "energy_save",
            "delta": {
                "action_oa_tanh": -0.30,
                "action_coil_tanh": -0.30,
                "action_fan_tanh": -0.30,
                "action_zone_1_tanh": -0.10,
                "action_zone_2_tanh": -0.10,
                "action_zone_3_tanh": -0.10,
                "action_zone_4_tanh": -0.10,
            },
        },
        {
            "name": "cooling_boost",
            "delta": {
                "action_coil_tanh": +0.40,
                "action_fan_tanh": +0.10,
                "action_zone_1_tanh": +0.15,
                "action_zone_2_tanh": +0.15,
                "action_zone_3_tanh": +0.15,
                "action_zone_4_tanh": +0.15,
            },
        },
        {
            "name": "co2_relief",
            "delta": {
                "action_oa_tanh": +0.50,
                "action_fan_tanh": +0.05,
                "action_zone_1_tanh": +0.20,
                "action_zone_2_tanh": +0.20,
                "action_zone_3_tanh": +0.20,
                "action_zone_4_tanh": +0.20,
            },
        },
    ]
def main() -> None:
    parser = argparse.ArgumentParser(description="Build GRPO dataset by sampling steps and generating action variants")
    parser.add_argument("--sources", nargs="+", type=Path, required=True, help="decision-step CSVs (one per day)")
    parser.add_argument("--per-day", type=int, default=25, help="steps sampled per day")
    parser.add_argument("--variants", type=int, default=4, help="additional variants per step (base + variants)")
    parser.add_argument("--output", type=Path, default=Path("critic_value/data/dataset.csv"))
    args = parser.parse_args()

    rows: List[pd.Series] = []
    variant_templates = _build_variants()
    if args.variants < len(variant_templates) - 1:
        variant_templates = variant_templates[: args.variants + 1]

    for src in args.sources:
        df = pd.read_csv(src)
        df = filter_decision_steps(df)
        df = df.drop(columns=["sample_id", "variant_id", "variant_name"], errors="ignore")

        for group in iter_episode_groups(df):
            sampled = sample_evenly(group, args.per_day)
            for _, base in sampled.iterrows():
                for vid, variant in enumerate(variant_templates):
                    row = base.copy()
                    row["variant_id"] = vid
                    row["variant_name"] = variant["name"]
                    for col, delta in variant["delta"].items():
                        row[col] = float(np.clip(row[col] + delta, -0.999, 0.999))
                    rows.append(row)

    out_df = ensure_sample_ids(pd.DataFrame(rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
