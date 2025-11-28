from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


plt.switch_backend("Agg")


def _quadratic_positive_root(a: float, b: float, c: float) -> float | None:
    """Return the positive real root of a + b x + c x^2 = 0 if it exists.

    If both roots are negative or imaginary, return None.
    """
    a = float(a)
    b = float(b)
    c = float(c)
    # Handle near-zero quadratic term by linear fallback
    if abs(c) < 1e-12:
        if abs(b) < 1e-12:
            return None
        x = -a / b
        return x if x > 0.0 else None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = float(np.sqrt(disc))
    x1 = (-b + sqrt_disc) / (2.0 * c)
    x2 = (-b - sqrt_disc) / (2.0 * c)
    candidates = [x for x in (x1, x2) if x > 0.0]
    if not candidates:
        return None
    return float(min(candidates))


class VariableSpeedPump:
    """Simple quadratic pump head model with inverter scaling.

    Head equation (kPa): dp = (a + b*(g/inv) + c*(g/inv)^2) * inv^2
    where g is flow [m^3/min], inv is inverter fraction [-].
    """

    def __init__(self, pg_coeffs: Sequence[float], design_flow_m3_min: float, reference_inv: float = 1.0) -> None:
        if len(pg_coeffs) != 3:
            raise ValueError("pg_coeffs must have 3 elements: [a, b, c]")
        self.a = float(pg_coeffs[0])
        self.b = float(pg_coeffs[1])
        self.c = float(pg_coeffs[2])
        self.design_flow = float(max(design_flow_m3_min, 1e-6))
        self.reference_inv = float(max(reference_inv, 1e-6))

    def head_kpa(self, g_m3_min: float, inv: float) -> float:
        g = float(max(g_m3_min, 0.0))
        inv = float(max(inv, 0.0))
        if inv <= 0.0:
            return 0.0
        bracket = self.a + self.b * (g / inv) + self.c * (g / inv) ** 2
        dp = bracket * inv * inv
        return float(max(dp, 0.0))

    def zero_head_flow(self, inv: float) -> float | None:
        inv = float(max(inv, 0.0))
        if inv <= 0.0:
            return 0.0
        root = _quadratic_positive_root(self.a, self.b, self.c)
        return None if root is None else float(root * inv)


def _sanitize_speeds(speeds: Iterable[float], include_reference: bool, reference: float) -> list[float]:
    sanitized = {float(s) for s in speeds if float(s) > 0.0}
    if include_reference:
        sanitized.add(float(reference))
    if not sanitized:
        raise ValueError("At least one positive inverter speed must be provided.")
    return sorted(sanitized)


def plot_hq_curve(
    output_path: Path,
    *,
    pg_coeffs: Sequence[float] = (100.0, -30.0, -20.0),
    design_flow_m3_min: float = 0.6,
    speeds: Iterable[float] = (0.60, 0.80, 1.00, 1.20),
    include_reference: bool = True,
) -> None:
    pump = VariableSpeedPump(pg_coeffs=pg_coeffs, design_flow_m3_min=design_flow_m3_min, reference_inv=1.0)

    speed_list = _sanitize_speeds(speeds, include_reference, pump.reference_inv)

    # Determine common x-axis upper bound from the largest shut-off flow among selected speeds
    g_max_candidates = []
    for inv in speed_list:
        g0 = pump.zero_head_flow(inv)
        if g0 is not None and g0 > 0.0:
            g_max_candidates.append(g0)
    g_max = float(max(g_max_candidates)) if g_max_candidates else max(2.0 * pump.design_flow, 1.0)
    flows = np.linspace(0.0, g_max * 1.02, 300)

    fig, ax = plt.subplots(figsize=(8, 5))
    for inv in speed_list:
        dp_vals = np.array([pump.head_kpa(flow, inv) for flow in flows])
        ax.plot(flows, dp_vals, label=f"inv = {inv:.2f}")

    # (design point marker removed by user request)

    ax.set_xlabel("Flow [m^3/min]")
    ax.set_ylabel("Head [kPa]")
    ax.set_title("Pump H-Q Curve")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.legend(title="Pump inverter")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved H-Q curve to {output_path}")
    # (design point console output removed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pump H-Q curves for selected inverter speeds.")
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=None,
        help="Inverter fractions to evaluate, e.g. --speeds 0.6 0.8 1.0 1.2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pump_hq_curve.png"),
        help="Destination image file. Defaults to pump_hq_curve.png",
    )
    parser.add_argument(
        "--pg",
        type=float,
        nargs=3,
        metavar=("A", "B", "C"),
        default=(100.0, -30.0, -20.0),
        help="Pump head coefficients [kPa]: A B C for dp = (A + B*g + C*g^2) at inv=1",
    )
    parser.add_argument(
        "--design-flow",
        type=float,
        default=0.6,
        help="Design flow [m^3/min] at inv=1.0",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Exclude the reference inverter speed (1.00) from the plotted curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speeds = args.speeds if args.speeds is not None else [0.60, 0.80, 1.00, 1.20]
    plot_hq_curve(
        args.output,
        pg_coeffs=args.pg,
        design_flow_m3_min=args.design_flow,
        speeds=speeds,
        include_reference=not args.no_reference,
    )


if __name__ == "__main__":
    main()


