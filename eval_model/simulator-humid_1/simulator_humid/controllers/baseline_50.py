#!/usr/bin/env python3
"""
ベースライン制御システム

外気ダンパー50%固定、ゾーンダンパー50%固定、
コイルバルブはPI制御、ファン速度は最小化ロジック（弁飽和時のみ増速）によるベースライン制御
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

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

import phyvac as pv

from simulator_humid.simulation import (
    HVACActions,
    ZoneConfig,
    build_default_zones as simulation_build_default_zones,
    create_plots,
    outdoor_temperature,
    run_simulation,
)
from simulator_humid.utils.paths import BASELINE_OUTPUT_DIR


@dataclass
class BaselineConfig:
    """ベースライン制御システムの設定"""
    
    # 固定制御値
    oa_damper_fixed: float = 0.4  # 外気ダンパー40%固定
    zone_damper_fixed: float = 0.5  # ゾーンダンパー50%固定
    
    # PID制御パラメータ
    coil_pid_kp: float = 0.2  # コイルバルブ比例ゲイン
    coil_pid_ti: float = 1200.0  # コイルバルブ積分時間 [s]
    coil_valve_min: float = 0.06  # コイルバルブ最小開度
    
    fan_pid_kp: float = 1.2  # ファン速度比例ゲイン
    fan_pid_ti: float = 180.0  # ファン速度積分時間 [s]
    fan_min: float = 0.45  # ファン最小速度
    fan_max: float = 1.4  # ファン最大速度
    
    # 制御更新間隔
    control_interval_minutes: int = 1  # 制御更新間隔（分）
    
    # シミュレーション設定
    timestep_s: int = 60  # タイムステップ（秒）
    episode_minutes: int = 24 * 60  # エピソード長（分）
    start_time: datetime = datetime(2025, 7, 29, 0, 0)  # 開始時刻
    setpoint: float = 26.0  # 温度設定値
    
    # HVAC運転スケジュール
    hvac_start_hour: float = 8.0  # HVAC開始時刻（時）
    hvac_stop_hour: int = 18  # HVAC終了時刻（時）
    
    # 出力設定
    output_dir: Path = BASELINE_OUTPUT_DIR  # 出力ディレクトリ


class BaselineController:
    """ベースライン制御システムのコントローラー"""
    
    def __init__(self, config: BaselineConfig, zones: Sequence[ZoneConfig]):
        self.config = config
        self.zones = zones
        self.zone_count = len(zones)
        
        # PID制御器の初期化
        self.coil_pid = pv.PID(
            kp=config.coil_pid_kp,
            ti=config.coil_pid_ti,
            a_min=config.coil_valve_min,
            a_max=1.0,
            t_step=1,
            kg=-1,
            t_reset=300,
            a=config.coil_valve_min,
        )
        
        self.fan_pid = pv.PID(
            kp=config.fan_pid_kp,
            ti=config.fan_pid_ti,
            a_min=config.fan_min,
            a_max=config.fan_max,
            t_step=1,
            kg=-1,
            t_reset=300,
            a=config.fan_min,
        )
        
        # 制御履歴
        self.control_period_s = max(1, int(config.control_interval_minutes * 60))
        self.last_command_time: Optional[datetime] = None
        self.obs_history: List[np.ndarray] = []
        self.action_history: List[np.ndarray] = []
        self.timestamps: List[datetime] = []
        
        # 前回の状態
        self.prev_zone_temps: Optional[np.ndarray] = None
        self.prev_zone_co2: Optional[np.ndarray] = None
        # ファン速度コマンド（最小化ロジック用）
        self.fan_speed_cmd: float = float(config.fan_min)
    
    def build_observation(
        self,
        timestamp: datetime,
        zone_temps: np.ndarray,
        zone_co2: np.ndarray,
        zone_rh: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """観測値の構築"""
        zone_temps = zone_temps.astype(np.float32)
        zone_co2 = zone_co2.astype(np.float32)
        
        # 温度・CO2の変化量を計算
        if self.prev_zone_temps is None:
            temp_delta = np.zeros_like(zone_temps)
        else:
            temp_delta = zone_temps - self.prev_zone_temps
            
        if self.prev_zone_co2 is None:
            co2_delta = np.zeros_like(zone_co2)
        else:
            co2_delta = zone_co2 - self.prev_zone_co2
        
        self.prev_zone_temps = zone_temps.copy()
        self.prev_zone_co2 = zone_co2.copy()
        
        # 観測値の構築
        temp_error = zone_temps - self.config.setpoint
        co2_error = zone_co2 - 1000.0  # CO2目標値1000ppm
        outdoor = float(outdoor_temperature(timestamp))
        minute = timestamp.hour * 60 + timestamp.minute
        angle = 2.0 * math.pi * (minute / 1440.0)
        sin_time = math.sin(angle)
        cos_time = math.cos(angle)
        current_hour = timestamp.hour + timestamp.minute / 60.0
        hvac_on_flag = 1.0 if self.config.hvac_start_hour <= current_hour < self.config.hvac_stop_hour else 0.0
        
        obs_parts = [
            temp_error,
            temp_delta,
            co2_error,
            co2_delta,
            np.array([outdoor, sin_time, cos_time, hvac_on_flag], dtype=np.float32),
        ]
        if zone_rh is not None:
            zone_rh = zone_rh.astype(np.float32)
            obs_parts.insert(4, zone_rh)
        obs = np.concatenate(obs_parts)

        return np.nan_to_num(obs, nan=0.0)

    def __call__(
        self,
        timestamp: datetime,
        zone_temps: np.ndarray,
        zone_co2: np.ndarray,
        zone_rh: Optional[np.ndarray] = None,
    ) -> HVACActions:
        """制御アクションの計算"""
        obs_vec = self.build_observation(timestamp, zone_temps, zone_co2, zone_rh)
        
        current_hour = timestamp.hour + timestamp.minute / 60.0
        hvac_on = self.config.hvac_start_hour <= current_hour < self.config.hvac_stop_hour
        
        # 制御更新の判定
        should_update = hvac_on and (
            self.last_command_time is None
            or (timestamp - self.last_command_time).total_seconds() >= self.control_period_s - 1e-9
        )
        
        if should_update:
            # コイルバルブのPID制御
            mean_zone_temp = float(np.mean(zone_temps))
            coil_valve = self.coil_pid.control(sp=self.config.setpoint, mv=mean_zone_temp)
            coil_valve = float(np.clip(coil_valve, self.config.coil_valve_min, 1.0))
            
            # ファン速度最小化ロジック
            # 目的：温度は主にコイルで追従。コイル弁が飽和してもなお高温時のみファンを段階的に増速。
            # 余裕がある（弁が閉方向・温度誤差小さい）ときは微減して最小必要流量を探索。
            temp_error = float(mean_zone_temp - self.config.setpoint)
            deadband = 0.2  # [°C] デッドバンド
            ramp_up = 0.03  # 1分あたりの増速上限
            ramp_down = 0.02  # 1分あたりの減速上限

            new_fan = self.fan_speed_cmd

            # 立ち上がり時など、弁が高開度かつ暑い → 送風を増やす
            if temp_error > deadband and coil_valve >= 0.95:
                new_fan = min(new_fan + ramp_up, self.config.fan_max)
            # 冷え過ぎ・弁が閉気味 → 送風を下げる
            elif temp_error < -deadband and coil_valve <= 0.10:
                new_fan = max(new_fan - ramp_down, self.config.fan_min)
            else:
                # 余裕時は微減して最小を探索（ハンチングを避ける軽いニブリング）
                if abs(temp_error) <= deadband and coil_valve < 0.8:
                    new_fan = max(new_fan - 0.005, self.config.fan_min)

            self.fan_speed_cmd = float(np.clip(new_fan, self.config.fan_min, self.config.fan_max))
            fan_speed = self.fan_speed_cmd
            
            self.last_command_time = timestamp
        else:
            # 前回の値を維持
            coil_valve = self.coil_pid.a
            fan_speed = self.fan_speed_cmd
        
        # 固定制御値
        zone_dampers = [self.config.zone_damper_fixed] * self.zone_count
        oa_damper = self.config.oa_damper_fixed
        
        # 履歴の記録
        action_vec = np.array(zone_dampers + [oa_damper, coil_valve, fan_speed], dtype=np.float32)
        self.obs_history.append(obs_vec.astype(np.float32, copy=False))
        self.action_history.append(action_vec.astype(np.float32, copy=False))
        self.timestamps.append(timestamp)
        
        return HVACActions(
            zone_dampers=zone_dampers,
            oa_damper=oa_damper,
            coil_valve=coil_valve,
            fan_speed=fan_speed,
        )
    
    def trajectory(self) -> dict[str, np.ndarray | List[datetime]]:
        """制御履歴の取得"""
        obs = np.asarray(self.obs_history, dtype=np.float32)
        actions = np.asarray(self.action_history, dtype=np.float32)
        return {
            "obs": obs,
            "actions": actions,
            "timestamps": list(self.timestamps),
        }


def build_default_zones() -> Sequence[ZoneConfig]:
    """デフォルトゾーン設定の構築"""
    return simulation_build_default_zones()


def compute_baseline_metrics(df: pd.DataFrame, setpoint: float) -> dict[str, float]:
    """ベースライン制御の性能指標を計算"""
    hvac_df = df[df["hvac_on"]] if "hvac_on" in df.columns else df
    if hvac_df.empty:
        hvac_df = df
    
    zone_temp_cols = sorted(
        col for col in hvac_df.columns if col.startswith("zone") and col.endswith("_temp")
    )
    
    metrics = {
        "mean_temp_error": 0.0,
        "max_temp_error": 0.0,
        "comfort_violation_ratio": 0.0,
        "mean_power_kw": 0.0,
        "total_power_kw": 0.0,
        "mean_co2_ppm": 0.0,
        "max_co2_ppm": 0.0,
    }
    
    if zone_temp_cols:
        temps = hvac_df[zone_temp_cols].to_numpy()
        temp_errors = temps - setpoint
        
        if temp_errors.size:
            metrics["mean_temp_error"] = float(np.mean(np.abs(temp_errors)))
            metrics["max_temp_error"] = float(np.max(np.abs(temp_errors)))
            
            # 快適域違反率（25-27℃の範囲外）
            comfort_violations = (temps < 25.0) | (temps > 27.0)
            metrics["comfort_violation_ratio"] = float(np.mean(comfort_violations))
    
    # 電力消費
    power_cols = ["fan_power_kw", "chw_pump_power_kw", "chiller_power_kw"]
    power_data = hvac_df[power_cols].sum(axis=1)
    if not power_data.empty:
        metrics["mean_power_kw"] = float(power_data.mean())
        # 1分間の電力消費をkWhに変換（1分 = 1/60時間）
        metrics["total_power_kwh"] = float(power_data.sum()) / 60.0
    
    # CO2濃度
    co2_cols = sorted(
        col for col in hvac_df.columns if col.startswith("zone") and col.endswith("_co2_ppm")
    )
    if co2_cols:
        co2_data = hvac_df[co2_cols].to_numpy()
        if co2_data.size:
            metrics["mean_co2_ppm"] = float(np.mean(co2_data))
            metrics["max_co2_ppm"] = float(np.max(co2_data))
    
    return metrics


def run_baseline_simulation(config: BaselineConfig) -> pd.DataFrame:
    """ベースライン制御シミュレーションの実行"""
    zones = build_default_zones()
    controller = BaselineController(config, zones)
    
    sim_kwargs = {
        "start": config.start_time,
        "minutes": config.episode_minutes,
        "timestep_s": config.timestep_s,
        "zones": list(zones),
        "coil_approach": 1.0,
        "coil_ua": 3500.0,
        "default_fan_inv": 0.95,
        "static_pressure_limit": 900.0,
        "setpoint": config.setpoint,
        "zone_pid_kp": 0.6,
        "zone_pid_ti": 25.0,
        "zone_pid_t_reset": 30,
        "zone_pid_t_step": 2,
        "hvac_start_hour": config.hvac_start_hour,
        "hvac_stop_hour": config.hvac_stop_hour,
        "action_callback": controller,
    }
    
    df = run_simulation(**sim_kwargs)
    return df


def main() -> None:
    """メイン実行関数"""
    config = BaselineConfig()
    
    # 出力ディレクトリを baseline_50 に変更
    config.output_dir = BASELINE_OUTPUT_DIR / "baseline_50"
    
    # 出力ディレクトリの作成
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = config.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print("ベースライン制御シミュレーションを開始します...")
    print(f"外気ダンパー: {config.oa_damper_fixed*100:.0f}% 固定")
    print(f"ゾーンダンパー: {config.zone_damper_fixed*100:.0f}% 固定")
    print(f"コイルバルブ: PID制御 (Kp={config.coil_pid_kp}, Ti={config.coil_pid_ti}s)")
    print("ファン速度: 最小化ロジック（弁飽和時のみ段階増速、余裕時は微減）")
    
    # シミュレーション実行
    df = run_baseline_simulation(config)
    
    # 性能指標の計算
    metrics = compute_baseline_metrics(df, config.setpoint)
    
    print("\n=== ベースライン制御結果 ===")
    print(f"平均温度誤差: {metrics['mean_temp_error']:.3f}°C")
    print(f"最大温度誤差: {metrics['max_temp_error']:.3f}°C")
    print(f"快適域違反率: {metrics['comfort_violation_ratio']:.3f}")
    print(f"平均電力消費: {metrics['mean_power_kw']:.3f}kW")
    print(f"総電力消費: {metrics['total_power_kwh']:.3f}kWh")
    print(f"平均CO2濃度: {metrics['mean_co2_ppm']:.1f}ppm")
    print(f"最大CO2濃度: {metrics['max_co2_ppm']:.1f}ppm")
    
    # 結果の保存
    csv_path = config.output_dir / "baseline_results.csv"
    df.to_csv(csv_path)
    print(f"\n結果をCSVに保存: {csv_path}")
    
    # テキストファイルでの結果保存
    txt_path = config.output_dir / "baseline_results.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== ベースライン制御結果 ===\n")
        f.write(f"平均温度誤差: {metrics['mean_temp_error']:.3f}°C\n")
        f.write(f"最大温度誤差: {metrics['max_temp_error']:.3f}°C\n")
        f.write(f"快適域違反率: {metrics['comfort_violation_ratio']:.3f}\n")
        f.write(f"平均電力消費: {metrics['mean_power_kw']:.3f}kW\n")
        f.write(f"総電力消費: {metrics['total_power_kwh']:.3f}kWh\n")
        f.write(f"平均CO2濃度: {metrics['mean_co2_ppm']:.1f}ppm\n")
        f.write(f"最大CO2濃度: {metrics['max_co2_ppm']:.1f}ppm\n")
    print(f"結果をテキストファイルに保存: {txt_path}")
    
    # プロットの作成
    main_plot_path = plot_dir / "baseline_results.png"
    damper_plot_path = plot_dir / "baseline_dampers.png"
    create_plots(df, main_plot_path, damper_plot_path)
    print(f"プロットを保存: {main_plot_path}, {damper_plot_path}")
    
    print("\nベースライン制御シミュレーションが完了しました。")


if __name__ == "__main__":
    main()
