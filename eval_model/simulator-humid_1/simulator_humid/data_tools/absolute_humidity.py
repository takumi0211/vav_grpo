"""Add absolute humidity derived from relative humidity and dry-bulb temperature."""
from __future__ import annotations

if __package__ in (None, ""):
    import pathlib
    import sys

    _pkg_dir = pathlib.Path(__file__).resolve().parent
    while _pkg_dir.name != "simulator_humid" and _pkg_dir.parent != _pkg_dir:
        _pkg_dir = _pkg_dir.parent
    if _pkg_dir.name == "simulator_humid":
        _project_root = _pkg_dir.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        del _project_root
    del _pkg_dir

from pathlib import Path

import pandas as pd

from phyvac.phyvac import tdb_rh2h_x

from simulator_humid.utils.paths import WEATHER_DATA_DIR


# Absolute humidity (humidity ratio) returned by phyvac.tdb_rh2h_x is in kg water / kg dry air.
# Convert to g/kg to make the numbers easier to interpret while keeping the underlying physics.
KG_TO_G = 1000.0


def calculate_absolute_humidity_g_per_kg(temp_c: float, relative_humidity: float) -> float:
    """Return absolute humidity in g/kg dry air for given temperature and relative humidity."""
    _, humidity_ratio = tdb_rh2h_x(float(temp_c), float(relative_humidity))
    return humidity_ratio * KG_TO_G


def add_absolute_humidity_column(csv_path: Path) -> None:
    """Append an absolute humidity column to the provided CSV file in-place."""
    df = pd.read_csv(csv_path)
    required_cols = {"temp_c", "relative_humidity"}
    missing = required_cols.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    df["absolute_humidity_g_per_kg"] = [
        calculate_absolute_humidity_g_per_kg(t, rh) for t, rh in zip(df["temp_c"], df["relative_humidity"])
    ]
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    default_path = WEATHER_DATA_DIR / "outdoor_temp_20250729.csv"
    add_absolute_humidity_column(default_path)
