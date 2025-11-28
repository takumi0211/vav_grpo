"""Optimize zone PID gains before running the VAV simulation."""
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

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from simulator_humid.simulation import (
    ZoneConfig,
    build_default_zones,
    compute_zone_pid_metrics,
    create_plots,
    run_simulation,
    zone_pid_cost,
)
from simulator_humid.utils.paths import REFERENCE_OUTPUT_DIR


@dataclass
class CandidateResult:
    kp: float
    ti: float
    cost: float
    metrics: Dict[str, float]


START_TIME = datetime(2025, 7, 29, 0, 0)
SETPOINT = 26.0
SUPPLY_AIR_SETPOINT = 16.0
MOD_SP_FLOOR = 200.0
MOD_SP_CEILING = 500.0


def build_zones() -> Tuple[ZoneConfig, ...]:
    """Return the default 4-zone configuration used in the baseline script."""
    return build_default_zones()


def base_simulation_kwargs() -> Dict[str, object]:
    """Baseline kwargs matching simulator_humid.simulation.main."""
    return {
        "start": START_TIME,
        "minutes": 24 * 60,
        "timestep_s": 60,
        "zones": list(build_zones()),
        "oa_frac_min": 0.3,
        "coil_approach": 1.0,
        "coil_ua": 3500.0,
        "default_fan_inv": 0.95,
        "static_pressure_limit": 900.0,
        "mod_sp_floor": MOD_SP_FLOOR,
        "mod_sp_ceiling": MOD_SP_CEILING,
        "supply_air_setpoint": SUPPLY_AIR_SETPOINT,
        "setpoint": SETPOINT,
        "zone_pid_t_reset": 30,
        "zone_pid_t_step": 1,
        "zone_pid_max_step": 0.10,
        "hvac_start_hour": 8.0,
        "hvac_stop_hour": 18,
    }


def evaluate_candidate(kp: float, ti: float) -> CandidateResult:
    """Run one simulation and return the resulting KPI bundle."""
    sim_kwargs = base_simulation_kwargs()
    df = run_simulation(zone_pid_kp=kp, zone_pid_ti=ti, **sim_kwargs)
    metrics = compute_zone_pid_metrics(df, setpoint=SETPOINT)
    cost = zone_pid_cost(metrics)
    return CandidateResult(kp=kp, ti=ti, cost=cost, metrics=metrics)


def grid_search(
    kp_values: Iterable[float],
    ti_values: Iterable[float],
    cache: Dict[Tuple[float, float], CandidateResult],
    current_best: CandidateResult | None = None,
) -> CandidateResult:
    """Evaluate a grid of PID pairs and return the best candidate."""
    best = current_best

    for kp in kp_values:
        for ti in ti_values:
            key = (round(kp, 6), round(ti, 6))
            if key in cache:
                result = cache[key]
            else:
                print(f"Evaluating kp={kp:.4f}, ti={ti:.2f} ...", flush=True)
                result = evaluate_candidate(kp, ti)
                cache[key] = result
            if best is None or result.cost < best.cost:
                best = result
                print(
                    f"  -> new best: cost={best.cost:.4f}, kp={best.kp:.4f}, ti={best.ti:.2f}",
                    flush=True,
                )
    if best is None:
        raise RuntimeError("Grid search produced no candidates.")
    return best


def refine_search(initial_best: CandidateResult) -> CandidateResult:
    """Perform a two-stage grid search around the best coarse result."""
    cache: Dict[Tuple[float, float], CandidateResult] = {}
    baseline = evaluate_candidate(initial_best.kp, initial_best.ti)
    cache[(round(baseline.kp, 6), round(baseline.ti, 6))] = baseline
    print(
        f"Baseline cost={baseline.cost:.4f} at kp={baseline.kp:.4f}, ti={baseline.ti:.2f}",
        flush=True,
    )

    coarse_kp = np.linspace(0.3, 1.1, num=7)
    coarse_ti = np.linspace(12.0, 60.0, num=9)
    best = grid_search(coarse_kp, coarse_ti, cache, current_best=baseline)

    kp_min = max(0.1, best.kp - 0.2)
    kp_max = min(1.5, best.kp + 0.2)
    ti_min = max(6.0, best.ti - 12.0)
    ti_max = min(90.0, best.ti + 12.0)

    fine_kp = np.linspace(kp_min, kp_max, num=7)
    fine_ti = np.linspace(ti_min, ti_max, num=9)
    best = grid_search(fine_kp, fine_ti, cache, current_best=best)
    return best


def run_final_simulation(best: CandidateResult) -> None:
    """Re-run the simulation with the optimal PID pair and write outputs."""
    sim_kwargs = base_simulation_kwargs()
    df = run_simulation(zone_pid_kp=best.kp, zone_pid_ti=best.ti, **sim_kwargs)

    output_dir = REFERENCE_OUTPUT_DIR
    csv_path = output_dir / "simulation_results.csv"
    df.to_csv(csv_path, index=False)

    plot_path = output_dir / "simulation_results.png"
    damper_plot_path = output_dir / "damper_positions.png"
    create_plots(df, plot_path, damper_plot_path)

    metrics = compute_zone_pid_metrics(df, setpoint=SETPOINT)
    cost = zone_pid_cost(metrics)

    # ベースライン制御と同じ項目でメトリクスを計算
    hvac_df = df[df["hvac_on"]] if "hvac_on" in df.columns else df
    if hvac_df.empty:
        hvac_df = df
    
    zone_temp_cols = sorted(
        col for col in hvac_df.columns if col.startswith("zone") and col.endswith("_temp")
    )
    
    # 温度誤差の計算
    mean_temp_error = 0.0
    max_temp_error = 0.0
    comfort_violation_ratio = 0.0
    if zone_temp_cols:
        temps = hvac_df[zone_temp_cols].to_numpy()
        temp_errors = temps - SETPOINT
        if temp_errors.size:
            mean_temp_error = float(np.mean(np.abs(temp_errors)))
            max_temp_error = float(np.max(np.abs(temp_errors)))
            # 快適域違反率（25-27℃の範囲外）
            comfort_violations = (temps < 25.0) | (temps > 27.0)
            comfort_violation_ratio = float(np.mean(comfort_violations))
    
    # 電力消費の計算
    power_cols = ["fan_power_kw", "chw_pump_power_kw", "chiller_power_kw"]
    power_data = hvac_df[power_cols].sum(axis=1)
    mean_power_kw = 0.0
    total_power_kwh = 0.0
    if not power_data.empty:
        mean_power_kw = float(power_data.mean())
        # 1分間の電力消費をkWhに変換（1分 = 1/60時間）
        total_power_kwh = float(power_data.sum()) / 60.0
    
    # CO2濃度の計算
    co2_cols = sorted(
        col for col in hvac_df.columns if col.startswith("zone") and col.endswith("_co2_ppm")
    )
    mean_co2_ppm = 0.0
    max_co2_ppm = 0.0
    if co2_cols:
        co2_data = hvac_df[co2_cols].to_numpy()
        if co2_data.size:
            mean_co2_ppm = float(np.mean(co2_data))
            max_co2_ppm = float(np.max(co2_data))

    # テキストファイルでの結果保存
    txt_path = output_dir / "simulation_results.txt"
    with txt_path.open('w', encoding='utf-8') as f:
        f.write("=== 最適化シミュレーション結果 ===\n")
        f.write(f"平均温度誤差: {mean_temp_error:.3f}°C\n")
        f.write(f"最大温度誤差: {max_temp_error:.3f}°C\n")
        f.write(f"快適域違反率: {comfort_violation_ratio:.3f}\n")
        f.write(f"平均電力消費: {mean_power_kw:.3f}kW\n")
        f.write(f"総電力消費: {total_power_kwh:.3f}kWh\n")
        f.write(f"平均CO2濃度: {mean_co2_ppm:.1f}ppm\n")
        f.write(f"最大CO2濃度: {max_co2_ppm:.1f}ppm\n")
    print(f"結果をテキストファイルに保存: {txt_path}")

    print("Finished optimized run:")
    print(
        f"  kp={best.kp:.4f}, ti={best.ti:.2f}, cost={cost:.4f}, "
        f"max_overshoot={metrics['max_overshoot']:.3f}°C, "
        f"mean_abs_error={metrics['mean_abs_error']:.3f}°C, "
        f"damper_chatter={metrics['damper_chatter']:.4f}"
    )
    print(f"CSV -> {csv_path}")
    print(f"Plots -> {plot_path}, {damper_plot_path}")


def main() -> None:
    initial_guess = CandidateResult(kp=0.6, ti=25.0, cost=0.0, metrics={})
    best = refine_search(initial_guess)
    run_final_simulation(best)


if __name__ == "__main__":
    main()
