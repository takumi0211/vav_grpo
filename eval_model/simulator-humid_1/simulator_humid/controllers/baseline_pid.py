from __future__ import annotations

"""Baseline PID simulation runner with outputs isolated per spec."""

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

from datetime import datetime
from typing import Iterable, List

import numpy as np

from simulator_humid.simulation import (
    ZoneConfig,
    build_default_zones,
    compute_zone_pid_metrics,
    create_plots,
    run_simulation,
    zone_pid_cost,
)
from simulator_humid.utils.paths import BASELINE_OUTPUT_DIR

BASELINE_PID_DIR = BASELINE_OUTPUT_DIR / "baseline_pid"
BASELINE_PID_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILENAME = "simulation_results.csv"
TXT_FILENAME = "simulation_results.txt"
PLOT_FILENAME = "simulation_results.png"
DAMPER_PLOT_FILENAME = "damper_positions.png"


def _build_zones() -> List[ZoneConfig]:
    return list(build_default_zones())


def _save_metrics(df, setpoint: float) -> None:
    hvac_df = df[df["hvac_on"]] if "hvac_on" in df.columns else df
    if hvac_df.empty:
        hvac_df = df

    zone_temp_cols = sorted(
        col for col in hvac_df.columns if col.startswith("zone") and col.endswith("_temp")
    )
    mean_temp_error = 0.0
    max_temp_error = 0.0
    comfort_violation_ratio = 0.0
    if zone_temp_cols:
        temps = hvac_df[zone_temp_cols].to_numpy()
        temp_errors = temps - setpoint
        if temp_errors.size:
            mean_temp_error = float(np.mean(np.abs(temp_errors)))
            max_temp_error = float(np.max(np.abs(temp_errors)))
            comfort_violation_ratio = float(np.mean((temps < 25.0) | (temps > 27.0)))

    power_cols = ["fan_power_kw", "chw_pump_power_kw", "chiller_power_kw"]
    power_data = hvac_df[power_cols].sum(axis=1)
    mean_power_kw = float(power_data.mean()) if not power_data.empty else 0.0
    total_power_kwh = float(power_data.sum()) * (1.0 / 60.0) if not power_data.empty else 0.0

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

    rh_cols = sorted(
        col for col in hvac_df.columns if col.startswith("zone") and col.endswith("_rh")
    )
    mean_rh = 0.0
    max_rh = 0.0
    min_rh = 0.0
    if rh_cols:
        rh_data = hvac_df[rh_cols].to_numpy()
        if rh_data.size:
            mean_rh = float(np.mean(rh_data))
            max_rh = float(np.max(rh_data))
            min_rh = float(np.min(rh_data))

    txt_path = BASELINE_PID_DIR / TXT_FILENAME
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=== シミュレーション結果 ===\n")
        f.write(f"平均温度誤差: {mean_temp_error:.3f}°C\n")
        f.write(f"最大温度誤差: {max_temp_error:.3f}°C\n")
        f.write(f"快適域違反率: {comfort_violation_ratio:.3f}\n")
        f.write(f"平均電力消費: {mean_power_kw:.3f}kW\n")
        f.write(f"総電力消費: {total_power_kwh:.3f}kWh\n")
        f.write(f"平均CO2濃度: {mean_co2_ppm:.1f}ppm\n")
        f.write(f"最大CO2濃度: {max_co2_ppm:.1f}ppm\n")
        f.write(f"平均相対湿度: {mean_rh:.1f}%\n")
        f.write(f"最大相対湿度: {max_rh:.1f}%\n")
        f.write(f"最小相対湿度: {min_rh:.1f}%\n")
    print(f"結果をテキストファイルに保存: {txt_path}")


def main() -> None:
    start = datetime(2025, 7, 29, 0, 0)
    zones = _build_zones()

    minutes = 24 * 60
    timestep_s = 60
    setpoint = 26.0

    sim_kwargs = dict(
        start=start,
        minutes=minutes,
        timestep_s=timestep_s,
        zones=zones,
        coil_approach=1.0,
        coil_ua=3500.0,
        coil_bypass_factor=0.1,
        default_fan_inv=0.95,
        static_pressure_limit=900.0,
        mod_sp_floor=250.0,
        mod_sp_ceiling=500.0,
        # ASHRAE G36 Trim & Respond を有効化
        use_trim_respond=True,
        tr_initial_sp=None,          # 未指定なら上限寄りで開始しトリムダウン
        tr_min_sp=None,              # None で mod_sp_floor を自動使用
        tr_max_sp=None,              # None で static_pressure_limit を自動使用
        tr_trim_pa=-15.0,            # 毎サイクル下げる基準量（Pa）
        tr_respond_pa=25.0,          # リクエスト超過1本あたり上げる量（Pa）
        tr_max_step_pa=75.0,         # 1回の更新で動かす最大幅（Pa）
        tr_sample_s=120,             # リセットのサンプリング間隔（s）
        tr_stability_wait_s=600,     # 立ち上がり安定化待ち時間（s）
        tr_request_threshold=0.95,   # ダンパー開度95%以上をリクエストと判定
        tr_ignore_requests=2,        # 無視する本数。これ超過分だけ Respond する
        setpoint=setpoint,
        zone_pid_t_reset=30,
        zone_pid_t_step=1,
        zone_pid_max_step=0.10,
    )

    selected_kp = 1.1667
    selected_ti = 100.0
    print(
        f"Running baseline PID simulation (G36 Trim & Respond ON) -> "
        f"kp={selected_kp:.3f}, ti={selected_ti:.1f}"
    )

    df = run_simulation(
        zone_pid_kp=selected_kp,
        zone_pid_ti=selected_ti,
        **sim_kwargs,
    )

    metrics = compute_zone_pid_metrics(df, setpoint=setpoint)
    cost = zone_pid_cost(metrics)
    print(
        f"Zone PID metrics -> cost={cost:.3f}, "
        f"max_overshoot={metrics['max_overshoot']:.3f}°C, "
        f"mean_abs_error={metrics['mean_abs_error']:.3f}°C, chatter={metrics['damper_chatter']:.3f}"
    )

    csv_path = BASELINE_PID_DIR / CSV_FILENAME
    df.to_csv(csv_path, index=False)
    print(f"結果をCSVに保存: {csv_path}")

    _save_metrics(df, setpoint=setpoint)

    plot_path = BASELINE_PID_DIR / PLOT_FILENAME
    damper_plot_path = BASELINE_PID_DIR / DAMPER_PLOT_FILENAME
    create_plots(df, plot_path, damper_plot_path)
    print(f"プロットを保存: {plot_path}, {damper_plot_path}")


if __name__ == "__main__":
    main()
