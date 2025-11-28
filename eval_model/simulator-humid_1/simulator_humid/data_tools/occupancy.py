"""Generate synthetic occupancy time-series scenarios for RL training.

The script produces configurable weekday/weekend occupancy profiles for each zone
used in the VAV simulation environment.  Each profile contains 24 hourly values
representing the expected number of occupants.  By default the generator writes
two artefacts: combined JSON/CSV summaries under ``--output-dir`` and per-zone
pattern libraries organised as ``people_data/zone*/zone*_pattern_XXX.csv``.

Example
-------
$ python generate_occupancy_scenarios.py --num-scenarios 20

This will create ``occupancy_scenarios.json`` and ``occupancy_scenarios.csv``
inside ``data/occupancy_scenarios`` and 20 pattern CSVs per zone under
``people_data``.
"""

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

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from simulator_humid.simulation import (
    DEFAULT_ZONE_OCCUPANCY_WEEKDAY,
    DEFAULT_ZONE_OCCUPANCY_WEEKEND,
)


@dataclass(frozen=True)
class ZoneProfileParams:
    """Parameter ranges for synthesising occupancy profiles per zone."""

    name: str
    weekday_peak_range: tuple[float, float]
    weekday_start_range: tuple[float, float]
    weekday_peak_hour_range: tuple[float, float]
    weekday_end_range: tuple[float, float]
    weekday_width_range: tuple[float, float]
    weekday_lunch_dip_fraction_range: tuple[float, float]
    weekday_lunch_center_range: tuple[float, float]
    weekday_lunch_width_range: tuple[float, float]
    weekday_noise_fraction_range: tuple[float, float]
    weekday_ramp_up_k_range: tuple[float, float]
    weekday_ramp_down_k_range: tuple[float, float]
    weekday_early_tail_fraction_range: tuple[float, float]
    weekday_early_tail_width_range: tuple[float, float]
    weekday_late_tail_fraction_range: tuple[float, float]
    weekday_late_tail_width_range: tuple[float, float]
    weekday_cutoff_threshold: float
    weekend_active_prob: float
    weekend_peak_range: tuple[float, float]
    weekend_start_range: tuple[float, float]
    weekend_peak_hour_range: tuple[float, float]
    weekend_end_range: tuple[float, float]
    weekend_width_range: tuple[float, float]
    weekend_noise_fraction_range: tuple[float, float]
    weekend_ramp_up_k_range: tuple[float, float]
    weekend_ramp_down_k_range: tuple[float, float]
    weekend_lunch_dip_fraction_range: tuple[float, float]
    weekend_lunch_center_range: tuple[float, float]
    weekend_lunch_width_range: tuple[float, float]
    weekend_cutoff_threshold: float
    weekend_early_tail_fraction_range: tuple[float, float]
    weekend_early_tail_width_range: tuple[float, float]
    weekend_late_tail_fraction_range: tuple[float, float]
    weekend_late_tail_width_range: tuple[float, float]


ZONE_PARAMS: Dict[str, ZoneProfileParams] = {
    "Zone 1": ZoneProfileParams(
        name="Zone 1",
        weekday_peak_range=(13.0, 22.0),
        weekday_start_range=(7.0, 9.3),
        weekday_peak_hour_range=(10.5, 13.8),
        weekday_end_range=(17.2, 20.8),
        weekday_width_range=(1.3, 2.5),
        weekday_lunch_dip_fraction_range=(0.08, 0.28),
        weekday_lunch_center_range=(12.0, 13.2),
        weekday_lunch_width_range=(0.45, 0.95),
        weekday_noise_fraction_range=(0.012, 0.045),
        weekday_ramp_up_k_range=(2.2, 3.6),
        weekday_ramp_down_k_range=(2.0, 3.3),
        weekday_early_tail_fraction_range=(0.0, 0.05),
        weekday_early_tail_width_range=(0.9, 1.4),
        weekday_late_tail_fraction_range=(0.02, 0.08),
        weekday_late_tail_width_range=(0.9, 1.4),
        weekday_cutoff_threshold=0.35,
        weekend_active_prob=0.35,
        weekend_peak_range=(3.0, 7.0),
        weekend_start_range=(9.5, 12.0),
        weekend_peak_hour_range=(11.5, 14.0),
        weekend_end_range=(14.0, 17.8),
        weekend_width_range=(0.9, 1.7),
        weekend_noise_fraction_range=(0.01, 0.04),
        weekend_ramp_up_k_range=(1.6, 3.0),
        weekend_ramp_down_k_range=(1.6, 2.8),
        weekend_lunch_dip_fraction_range=(0.0, 0.15),
        weekend_lunch_center_range=(12.0, 13.0),
        weekend_lunch_width_range=(0.4, 0.9),
        weekend_cutoff_threshold=0.2,
        weekend_early_tail_fraction_range=(0.0, 0.03),
        weekend_early_tail_width_range=(0.9, 1.4),
        weekend_late_tail_fraction_range=(0.0, 0.04),
        weekend_late_tail_width_range=(0.9, 1.4),
    ),
    "Zone 2": ZoneProfileParams(
        name="Zone 2",
        weekday_peak_range=(9.0, 15.0),
        weekday_start_range=(7.2, 9.0),
        weekday_peak_hour_range=(10.5, 13.5),
        weekday_end_range=(17.0, 20.2),
        weekday_width_range=(1.2, 2.3),
        weekday_lunch_dip_fraction_range=(0.1, 0.32),
        weekday_lunch_center_range=(12.2, 13.4),
        weekday_lunch_width_range=(0.5, 1.1),
        weekday_noise_fraction_range=(0.015, 0.05),
        weekday_ramp_up_k_range=(2.0, 3.2),
        weekday_ramp_down_k_range=(2.0, 3.2),
        weekday_early_tail_fraction_range=(0.0, 0.06),
        weekday_early_tail_width_range=(0.8, 1.5),
        weekday_late_tail_fraction_range=(0.01, 0.07),
        weekday_late_tail_width_range=(0.9, 1.5),
        weekday_cutoff_threshold=0.3,
        weekend_active_prob=0.28,
        weekend_peak_range=(2.0, 5.5),
        weekend_start_range=(10.0, 12.5),
        weekend_peak_hour_range=(11.5, 14.5),
        weekend_end_range=(14.0, 17.0),
        weekend_width_range=(0.9, 1.6),
        weekend_noise_fraction_range=(0.012, 0.045),
        weekend_ramp_up_k_range=(1.6, 2.6),
        weekend_ramp_down_k_range=(1.6, 2.7),
        weekend_lunch_dip_fraction_range=(0.0, 0.12),
        weekend_lunch_center_range=(12.0, 13.6),
        weekend_lunch_width_range=(0.4, 0.9),
        weekend_cutoff_threshold=0.18,
        weekend_early_tail_fraction_range=(0.0, 0.03),
        weekend_early_tail_width_range=(0.8, 1.3),
        weekend_late_tail_fraction_range=(0.0, 0.04),
        weekend_late_tail_width_range=(0.8, 1.3),
    ),
    "Zone 3": ZoneProfileParams(
        name="Zone 3",
        weekday_peak_range=(8.0, 13.5),
        weekday_start_range=(7.0, 8.8),
        weekday_peak_hour_range=(10.0, 13.0),
        weekday_end_range=(16.8, 19.8),
        weekday_width_range=(1.1, 2.1),
        weekday_lunch_dip_fraction_range=(0.08, 0.28),
        weekday_lunch_center_range=(12.0, 13.0),
        weekday_lunch_width_range=(0.45, 0.9),
        weekday_noise_fraction_range=(0.015, 0.045),
        weekday_ramp_up_k_range=(2.0, 3.0),
        weekday_ramp_down_k_range=(1.9, 3.0),
        weekday_early_tail_fraction_range=(0.0, 0.05),
        weekday_early_tail_width_range=(0.8, 1.4),
        weekday_late_tail_fraction_range=(0.01, 0.05),
        weekday_late_tail_width_range=(0.8, 1.4),
        weekday_cutoff_threshold=0.25,
        weekend_active_prob=0.24,
        weekend_peak_range=(1.5, 4.5),
        weekend_start_range=(10.0, 12.0),
        weekend_peak_hour_range=(11.0, 14.0),
        weekend_end_range=(14.0, 17.0),
        weekend_width_range=(0.8, 1.5),
        weekend_noise_fraction_range=(0.01, 0.04),
        weekend_ramp_up_k_range=(1.6, 2.5),
        weekend_ramp_down_k_range=(1.6, 2.5),
        weekend_lunch_dip_fraction_range=(0.0, 0.1),
        weekend_lunch_center_range=(12.0, 13.2),
        weekend_lunch_width_range=(0.4, 0.8),
        weekend_cutoff_threshold=0.15,
        weekend_early_tail_fraction_range=(0.0, 0.03),
        weekend_early_tail_width_range=(0.7, 1.2),
        weekend_late_tail_fraction_range=(0.0, 0.03),
        weekend_late_tail_width_range=(0.7, 1.2),
    ),
    "Zone 4": ZoneProfileParams(
        name="Zone 4",
        weekday_peak_range=(5.0, 9.0),
        weekday_start_range=(7.5, 9.5),
        weekday_peak_hour_range=(10.5, 13.2),
        weekday_end_range=(17.0, 19.5),
        weekday_width_range=(1.0, 1.8),
        weekday_lunch_dip_fraction_range=(0.05, 0.2),
        weekday_lunch_center_range=(12.2, 13.4),
        weekday_lunch_width_range=(0.45, 0.9),
        weekday_noise_fraction_range=(0.015, 0.04),
        weekday_ramp_up_k_range=(2.0, 3.2),
        weekday_ramp_down_k_range=(1.8, 3.0),
        weekday_early_tail_fraction_range=(0.0, 0.05),
        weekday_early_tail_width_range=(0.8, 1.3),
        weekday_late_tail_fraction_range=(0.01, 0.05),
        weekday_late_tail_width_range=(0.8, 1.3),
        weekday_cutoff_threshold=0.2,
        weekend_active_prob=0.72,
        weekend_peak_range=(2.0, 5.0),
        weekend_start_range=(9.0, 12.0),
        weekend_peak_hour_range=(10.5, 13.5),
        weekend_end_range=(13.5, 17.0),
        weekend_width_range=(0.8, 1.4),
        weekend_noise_fraction_range=(0.01, 0.035),
        weekend_ramp_up_k_range=(1.5, 2.4),
        weekend_ramp_down_k_range=(1.5, 2.4),
        weekend_lunch_dip_fraction_range=(0.0, 0.12),
        weekend_lunch_center_range=(12.0, 13.0),
        weekend_lunch_width_range=(0.4, 0.8),
        weekend_cutoff_threshold=0.12,
        weekend_early_tail_fraction_range=(0.0, 0.03),
        weekend_early_tail_width_range=(0.7, 1.1),
        weekend_late_tail_fraction_range=(0.0, 0.04),
        weekend_late_tail_width_range=(0.7, 1.1),
    ),
}


BASELINE_WEEKDAY = {
    zone: np.array(values, dtype=float) for zone, values in DEFAULT_ZONE_OCCUPANCY_WEEKDAY.items()
}
BASELINE_WEEKEND = {
    zone: np.array(values, dtype=float) for zone, values in DEFAULT_ZONE_OCCUPANCY_WEEKEND.items()
}
ACTIVE_HOUR_MASK = np.zeros(24, dtype=float)
ACTIVE_HOUR_MASK[8:18] = 1.0


def _zone_slug(zone_name: str) -> str:
    return zone_name.lower().replace(" ", "")


def _logistic(x: np.ndarray, midpoint: float, sharpness: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-sharpness * (x - midpoint)))


def _logistic_down(x: np.ndarray, midpoint: float, sharpness: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(sharpness * (x - midpoint)))


def _gaussian(x: np.ndarray, center: float, width: float) -> np.ndarray:
    width = max(width, 0.35)
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def _ensure_bounds(start: float, peak: float, end: float) -> tuple[float, float, float]:
    # Guarantee chronological ordering with some margin.
    peak = max(peak, start + 0.35)
    end = max(end, peak + 0.75)
    return start, peak, end


def _add_tail(
    base: np.ndarray,
    hours: np.ndarray,
    peak_value: float,
    fraction_range: tuple[float, float],
    width_range: tuple[float, float],
    center: float,
    rng: np.random.Generator,
) -> None:
    frac = rng.uniform(*fraction_range)
    if frac <= 0.0:
        return
    width = rng.uniform(*width_range)
    base += frac * peak_value * _gaussian(hours, center=center, width=width)


def generate_profile(
    rng: np.random.Generator,
    params: ZoneProfileParams,
    day_type: str,
) -> np.ndarray:
    """Create a 24-hour occupancy profile for the requested day type."""

    prefix = "weekday" if day_type == "weekday" else "weekend"

    if day_type == "weekend" and rng.random() > params.weekend_active_prob:
        return np.zeros(24, dtype=float)

    hours = np.arange(24, dtype=float) + 0.5

    start = rng.uniform(*getattr(params, f"{prefix}_start_range"))
    peak_hour = rng.uniform(*getattr(params, f"{prefix}_peak_hour_range"))
    end = rng.uniform(*getattr(params, f"{prefix}_end_range"))
    start, peak_hour, end = _ensure_bounds(start, peak_hour, end)

    peak_value = rng.uniform(*getattr(params, f"{prefix}_peak_range"))
    width = rng.uniform(*getattr(params, f"{prefix}_width_range"))
    ramp_up_k = rng.uniform(*getattr(params, f"{prefix}_ramp_up_k_range"))
    ramp_down_k = rng.uniform(*getattr(params, f"{prefix}_ramp_down_k_range"))

    profile = peak_value * _gaussian(hours, center=peak_hour, width=width)
    profile *= _logistic(hours, midpoint=start, sharpness=ramp_up_k)
    profile *= _logistic_down(hours, midpoint=end, sharpness=ramp_down_k)

    # Apply midday reduction to mimic lunch or event gaps.
    dip_range = getattr(params, f"{prefix}_lunch_dip_fraction_range")
    dip_fraction = rng.uniform(*dip_range)
    if dip_fraction > 0.0:
        dip_center_range = getattr(params, f"{prefix}_lunch_center_range")
        dip_width_range = getattr(params, f"{prefix}_lunch_width_range")
        dip_center = rng.uniform(*dip_center_range)
        dip_width = rng.uniform(*dip_width_range)
    dip = dip_fraction * peak_value * _gaussian(hours, center=dip_center, width=dip_width)
    profile = np.maximum(profile - dip, 0.0)

    # Early and late tails let a few people linger outside core hours.
    early_tail_fraction_range = getattr(params, f"{prefix}_early_tail_fraction_range")
    early_tail_width_range = getattr(params, f"{prefix}_early_tail_width_range")
    _add_tail(
        profile,
        hours,
        peak_value,
        early_tail_fraction_range,
        early_tail_width_range,
        center=start - 1.1,
        rng=rng,
    )

    late_tail_fraction_range = getattr(params, f"{prefix}_late_tail_fraction_range")
    late_tail_width_range = getattr(params, f"{prefix}_late_tail_width_range")
    _add_tail(
        profile,
        hours,
        peak_value,
        late_tail_fraction_range,
        late_tail_width_range,
        center=end + 1.4,
        rng=rng,
    )

    # Blend with baseline profile to keep overall trend realistic.
    baseline_map = BASELINE_WEEKDAY if day_type == "weekday" else BASELINE_WEEKEND
    baseline = baseline_map.get(params.name, np.zeros(24, dtype=float)).copy()
    baseline *= ACTIVE_HOUR_MASK

    blend_weight = 0.0
    if baseline.max() > 0.0:
        if day_type == "weekday":
            blend_weight = rng.uniform(0.5, 0.85)
        else:
            blend_weight = rng.uniform(0.35, 0.7)
        profile = blend_weight * baseline + (1.0 - blend_weight) * profile

    # Add smooth noise for variability.
    noise_fraction_range = getattr(params, f"{prefix}_noise_fraction_range")
    noise_fraction = rng.uniform(*noise_fraction_range)
    if noise_fraction > 0:
        noise = rng.normal(loc=0.0, scale=noise_fraction * max(peak_value, 1.0), size=profile.size)
        profile = np.maximum(profile + noise, 0.0)

    # Encourage distinct morning/midday/afternoon tendencies while staying smooth.
    morning_factor = rng.uniform(0.85, 1.2)
    midday_factor = rng.uniform(0.9, 1.15)
    afternoon_factor = rng.uniform(0.75, 1.15)
    profile[8:11] *= morning_factor
    profile[11:14] *= midday_factor
    profile[14:18] *= afternoon_factor

    # Soft smoothing to avoid hour-to-hour spikes.
    kernel = np.array([0.2, 0.6, 0.2])
    padded = np.pad(profile, (1, 1), mode="edge")
    profile = np.convolve(padded, kernel, mode="valid")

    cutoff_threshold = getattr(params, f"{prefix}_cutoff_threshold")
    profile[profile < cutoff_threshold] = 0.0

    # Restrict occupancy to active hours (08:00-18:00).
    profile *= ACTIVE_HOUR_MASK

    # Re-normalise so the profile reaches the sampled peak (after masking and smoothing).
    max_val = profile.max()
    if max_val > 0:
        profile *= peak_value / max_val

    profile = np.clip(profile, 0.0, None)

    # Minor re-smoothing to iron out edges from masking.
    padded = np.pad(profile, (1, 1), mode="edge")
    profile = np.convolve(padded, kernel, mode="valid")
    profile *= ACTIVE_HOUR_MASK

    return np.round(profile, 1)


def scenario_iterator(
    rng: np.random.Generator,
    zone_params: Dict[str, ZoneProfileParams],
    num_scenarios: int,
) -> Iterable[dict]:
    for idx in range(num_scenarios):
        scenario_id = f"scenario_{idx + 1:03d}"
        scenario = {"id": scenario_id, "weekday": {}, "weekend": {}}
        for zone_name, params in zone_params.items():
            weekday_profile = generate_profile(rng, params, day_type="weekday")
            weekend_profile = generate_profile(rng, params, day_type="weekend")
            scenario["weekday"][zone_name] = weekday_profile.tolist()
            scenario["weekend"][zone_name] = weekend_profile.tolist()
        yield scenario


def write_zone_csvs(
    scenarios: list[dict],
    base_dir: Path,
) -> None:
    if not scenarios:
        return

    base_dir.mkdir(parents=True, exist_ok=True)
    zones = sorted(scenarios[0]["weekday"].keys())

    for zone in zones:
        (base_dir / _zone_slug(zone)).mkdir(parents=True, exist_ok=True)

    for idx, scenario in enumerate(scenarios, start=1):
        pattern_id = f"{idx:03d}"
        for zone in zones:
            slug = _zone_slug(zone)
            path = base_dir / slug / f"{slug}_pattern_{pattern_id}.csv"
            weekday_vals = scenario["weekday"][zone]
            weekend_vals = scenario["weekend"][zone]
            with path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["hour", "weekday", "weekend"])
                for hour in range(24):
                    writer.writerow([hour, weekday_vals[hour], weekend_vals[hour]])


def write_outputs(
    scenarios: list[dict],
    output_dir: Path,
    seed: int,
    zone_output_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "occupancy_scenarios.json"
    json_content = {
        "metadata": {
            "seed": seed,
            "num_scenarios": len(scenarios),
            "zones": list(ZONE_PARAMS.keys()),
            "description": "Synthetic occupancy schedules generated for RL training and simulations.",
        },
        "scenarios": scenarios,
    }
    json_path.write_text(json.dumps(json_content, indent=2), encoding="utf-8")

    csv_path = output_dir / "occupancy_scenarios.csv"
    fieldnames = ["scenario_id", "day_type", "hour", "zone", "occupancy"]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in scenarios:
            scenario_id = scenario["id"]
            for day_type in ("weekday", "weekend"):
                for zone, values in scenario[day_type].items():
                    for hour, occupancy in enumerate(values):
                        writer.writerow(
                            {
                                "scenario_id": scenario_id,
                                "day_type": day_type,
                                "hour": hour,
                                "zone": zone,
                                "occupancy": occupancy,
                            }
                        )

    if zone_output_dir is not None:
        write_zone_csvs(scenarios, zone_output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-scenarios", type=int, default=20, help="Number of occupancy scenarios to generate.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/occupancy_scenarios"),
        help="Directory where output JSON and CSV files will be written.",
    )
    parser.add_argument(
        "--zone-output-dir",
        type=Path,
        default=Path("people_data"),
        help="Base directory for per-zone pattern CSVs (zone subdirectories are created automatically).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    scenarios = list(scenario_iterator(rng, ZONE_PARAMS, args.num_scenarios))
    write_outputs(
        scenarios,
        args.output_dir,
        seed=args.seed,
        zone_output_dir=args.zone_output_dir,
    )

    print(
        f"Generated {len(scenarios)} occupancy scenarios across {len(ZONE_PARAMS)} zones. "
        f"Global files: '{args.output_dir}'. Per-zone CSVs: '{args.zone_output_dir}'."
    )


if __name__ == "__main__":
    main()
