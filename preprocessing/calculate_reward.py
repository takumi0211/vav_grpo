import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


Q_COLS = [f"q_action_{i}" for i in range(4)]
REWARD_COLS = [f"reward_action_{i}" for i in range(4)]


def softmax(values: Sequence[float], tau: float) -> np.ndarray:
    """Boltzmann policy with temperature tau."""
    logits = np.asarray(values, dtype=np.float64) / tau
    logits -= np.max(logits)
    probs = np.exp(logits, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Softmax underflow/overflow encountered.")
    return probs / total


def compute_rewards(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    rewards = np.zeros((len(df), len(Q_COLS)), dtype=np.float64)
    q_values = df[Q_COLS].to_numpy(dtype=np.float64)
    for idx, row in enumerate(q_values):
        rewards[idx] = softmax(row, tau)
    for col_idx, col_name in enumerate(REWARD_COLS):
        df[col_name] = rewards[:, col_idx]
    return df


def process_path(path: Path, tau: float, overwrite: bool) -> Path:
    df = pd.read_csv(path)
    missing = [col for col in Q_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing Q columns: {missing}")
    df = compute_rewards(df, tau)
    if overwrite:
        out_path = path
    else:
        out_path = path.with_name(path.stem + "_with_reward" + path.suffix)
    df.to_csv(out_path, index=False)
    return out_path


def iter_targets(paths: Iterable[Path]) -> Iterable[Path]:
    for target in paths:
        if target.is_dir():
            yield from sorted(target.rglob("*_q_dataset_harmony.csv"))
        elif target.suffix.lower() == ".csv":
            yield target
        else:
            raise ValueError(f"Unsupported path: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate reward_action_* columns via Boltzmann policy."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("data")],
        help="CSV files or directories to process (default: data/).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
        help="Temperature parameter for softmax (default: 0.1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input files instead of creating *_with_reward.csv",
    )
    args = parser.parse_args()

    for csv_path in iter_targets(args.paths):
        out_path = process_path(csv_path, args.tau, args.overwrite)
        print(f"Wrote rewards to {out_path}")


if __name__ == "__main__":
    main()
