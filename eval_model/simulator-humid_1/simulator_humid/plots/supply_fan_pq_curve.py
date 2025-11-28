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
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from simulator_humid.simulation import (
    ZoneBranch,
    ZoneConfig,
    VariableSpeedFan,
    RHO_AIR,
    build_default_zones as simulation_build_default_zones,
)

plt.switch_backend("Agg")


def build_default_zones() -> list[ZoneConfig]:
    return list(simulation_build_default_zones())


def design_fan_from_zones(
    zones: list[ZoneConfig],
    *,
    coil_dp_pa: float = 180.0,
    fan_nominal_flow_m3_min: float = 180.0,
    default_fan_inv: float = 0.95,
    static_pressure_limit: float = 900.0,
) -> tuple[VariableSpeedFan, float, float]:
    branches = [ZoneBranch(zone) for zone in zones]
    design_volumes: list[float] = []
    design_branch_dps: list[float] = []

    for branch, zone in zip(branches, zones):
        design_mdot = zone.design_flow_kg_s if zone.design_flow_kg_s is not None else zone.flow_max
        design_vol = max(design_mdot / RHO_AIR * 60.0, 1e-6)
        design_volumes.append(design_vol)
        design_branch_dps.append(branch.pressure_drop(design_vol, zone.design_damper_pos))

    total_design_flow = float(sum(design_volumes))
    nominal_cap_flow = max(fan_nominal_flow_m3_min, 1e-3)
    if total_design_flow > 1e-6:
        design_flow_target = min(nominal_cap_flow * 0.95, total_design_flow)
    else:
        design_flow_target = nominal_cap_flow * 0.95
    design_flow_target = max(design_flow_target, 0.05)

    if design_branch_dps:
        design_supply_static = float(max(design_branch_dps))
    else:
        design_supply_static = max(static_pressure_limit * 0.5, 200.0)

    design_total_head = coil_dp_pa + design_supply_static

    fan = VariableSpeedFan(
        design_flow_m3_min=design_flow_target,
        design_head_pa=design_total_head,
        reference_inv=max(default_fan_inv, 0.1),
    )

    return fan, design_flow_target, design_total_head


def _sanitize_speeds(speeds: Iterable[float], include_reference: bool, reference: float) -> list[float]:
    sanitized = {float(s) for s in speeds if float(s) > 0.0}
    if include_reference:
        sanitized.add(float(reference))
    if not sanitized:
        raise ValueError("At least one positive inverter speed must be provided.")
    return sorted(sanitized)


def plot_pq_curve(
    output_path: Path,
    *,
    speeds: Iterable[float],
    include_reference: bool = True,
) -> None:
    zones = build_default_zones()
    fan, design_flow, design_head = design_fan_from_zones(zones)

    speed_list = _sanitize_speeds(speeds, include_reference, fan.reference_inv)

    max_inv = max(speed_list)
    flows = np.linspace(0.0, fan.g_zero * max_inv * 1.05, 250)

    fig, ax = plt.subplots(figsize=(8, 5))
    for inv in speed_list:
        fan.inv = inv
        dp_vals = np.array([fan.f2p(flow) for flow in flows])
        ax.plot(flows, dp_vals, label=f"inv = {inv:.2f}")

    ax.set_xlabel("Flow [m^3/min]")
    ax.set_ylabel("Static Pressure [Pa]")
    ax.set_title("Supply Fan P-Q Curve")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.legend(title="Fan inverter")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved P-Q curve to {output_path}")
    print(f"Design flow ≈ {design_flow:.2f} m^3/min, design head ≈ {design_head:.1f} Pa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot supply fan P-Q curves for selected inverter speeds.")
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=None,
        help="Inverter fractions to evaluate, e.g. --speeds 0.6 0.75 1.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fan_pq_curve.png"),
        help="Destination image file. Defaults to fan_pq_curve.png",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Exclude the fan reference inverter speed from the plotted curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speeds = args.speeds if args.speeds is not None else [0.60, 0.75, 1.00, 1.20]
    plot_pq_curve(
        args.output,
        speeds=speeds,
        include_reference=not args.no_reference,
    )


if __name__ == "__main__":
    main()
