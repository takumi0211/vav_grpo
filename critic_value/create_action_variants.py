#!/usr/bin/env python3
"""Generate a small dataset that fixes one observation and varies actions.

This is handy for checking how the TD3 critic評価値 (Q) changes when only the
action is perturbed at a specific step.

Usage:
    python critic_value/create_action_variants.py --step 1040 \
        --source critic_value/data/dataset.csv --output critic_value/data/dataset.csv

Default base step is 1040 (17:20), which exists in the decision-step dataset
produced by run_td3.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _build_variants() -> List[Dict[str, float]]:
    """Predefined action perturbations (delta from base action)."""

    return [
        {"name": "base", "delta": {}},
        {
            "name": "vent_plus",
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
    parser = argparse.ArgumentParser(description="Create action variants for a single observation.")
    parser.add_argument("--source", type=Path, default=Path("critic_value/data/dataset.csv"), help="Input dataset with decision_step=1 rows")
    parser.add_argument("--step", type=int, default=1040, help="Base step to duplicate")
    parser.add_argument("--output", type=Path, default=Path("critic_value/data/dataset.csv"), help="Output dataset path")
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    if "step" not in df.columns:
        raise SystemExit("source dataset must have a 'step' column")

    base_rows = df[df["step"] == args.step]
    if base_rows.empty:
        raise SystemExit(f"step {args.step} not found in {args.source}")
    base = base_rows.iloc[0].copy()

    variants = _build_variants()
    action_cols = [c for c in df.columns if c.startswith("action_")]

    out_rows: List[pd.Series] = []
    for vid, variant in enumerate(variants):
        row = base.copy()
        row["variant_id"] = vid
        row["variant_name"] = variant["name"]
        for col, delta in variant["delta"].items():
            row[col] = float(np.clip(row[col] + delta, -0.999, 0.999))
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    # ensure sample_id exists and is consecutive
    out_df.insert(0, "sample_id", np.arange(len(out_df), dtype=np.int32))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} variants based on step {args.step} to {args.output}")


if __name__ == "__main__":
    main()
