from __future__ import annotations
# ---------------------------------------------------------------------------
# 外部ライブラリや標準ライブラリを読み込み、シミュレーションに必要な機能を準備
# ---------------------------------------------------------------------------
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import phyvac as pv

from simulator_humid.utils.paths import (
    WEATHER_DATA_DIR,
    FIGURES_OUTPUT_DIR,
    SIMULATION_OUTPUT_DIR,
)
from simulator_humid.config.duct_geometry import (
    ZONE_BRANCH_GEOMETRY,
    SUPPLY_TRUNK_GEOMETRY,
    RETURN_TRUNK_GEOMETRY,
    DEFAULT_BRANCH_GEOMETRY,
    BranchGeometry,
    SUPPLY_TAP_INDEX,
)

# ---------------------------------------------------------------------------
# 気象庁の観測CSVを読み込み、外気温・外気湿度の連続時系列を整備
# ---------------------------------------------------------------------------
# JMAの10分値を元に外気温・湿度時系列を読み込み、全時刻で補間できるよう整備
OUTDOOR_DATA_PATH = WEATHER_DATA_DIR / "outdoor_temp_20250729.csv"
if not OUTDOOR_DATA_PATH.exists():
    raise FileNotFoundError(f"Missing outdoor temperature data: {OUTDOOR_DATA_PATH}")

_outdoor_df = pd.read_csv(OUTDOOR_DATA_PATH)
required_cols = ["minutes", "temp_c", "relative_humidity"]
missing_cols = [col for col in required_cols if col not in _outdoor_df.columns]
if missing_cols:
    raise ValueError(f"Outdoor data file must include columns: {missing_cols}")
_outdoor_df = _outdoor_df.sort_values("minutes").reset_index(drop=True)
# 日付区間の先頭が0分で始まらないCSVも想定し、最初の行を複製して基準を作る
if _outdoor_df.loc[0, "minutes"] > 0:
    first_row = _outdoor_df.loc[0].copy()
    first_row["minutes"] = 0
    _outdoor_df = pd.concat([pd.DataFrame([first_row]), _outdoor_df], ignore_index=True)

OUTDOOR_MINUTES = _outdoor_df["minutes"].to_numpy()
OUTDOOR_TEMPS = _outdoor_df["temp_c"].to_numpy()
OUTDOOR_RHS = _outdoor_df["relative_humidity"].to_numpy()
# minutes列は線形補間時の参照軸（分単位）、temp_c列は外気乾球温度[℃]、relative_humidity列は外気相対湿度[%]として扱う

# 室内空気質評価用の代表外気CO2濃度[ppm]
OUTDOOR_CO2_PPM = 420.0

# CLI環境で描画するため、GUI非依存のバックエンドへ切り替え
plt.switch_backend("Agg")

# Physical constants (reference: phyvac CA constant [kJ/kgK])
# 冷房日シナリオで頻繁に使う物性値を先に定義
CP_AIR = pv.CA * 1000.0  # convert to J/(kg*K)
CP_WATER = 4180.0  # J/(kg*K)
RHO_AIR = 1.2  # kg/m3, assumed constant
RHO_WATER = 998.0  # kg/m3, approximate density of chilled water
DEFAULT_CHILLER_COP = 4.0
DEFAULT_INITIAL_REL_HUM = 50.0  # [%] initial relative humidity for all zones
OCCUPANCY_CUTOFF_HOUR = 18  # Occupants leave after 18:00


def occupancy_with_cutoff(
    values: Sequence[float], *, cutoff_hour: int = OCCUPANCY_CUTOFF_HOUR
) -> List[float]:
    """Return a copy of the occupancy profile with values zeroed after the cutoff hour."""

    arr = list(values)
    if len(arr) != 24:
        raise ValueError(
            f"Occupancy schedule must provide 24 hourly values, got {len(arr)}."
        )
    cutoff = max(0, min(int(cutoff_hour), len(arr)))
    for hour in range(cutoff, len(arr)):
        arr[hour] = 0.0
    return arr


def internal_gain_profile_from_occupancy(
    values: Sequence[float], *, day_gain: float, night_gain: float, cutoff_hour: int = OCCUPANCY_CUTOFF_HOUR
) -> np.ndarray:
    """Derive a 24-hour internal gain profile that respects the occupancy cutoff."""

    profile = np.array(occupancy_with_cutoff(values, cutoff_hour=cutoff_hour), dtype=float)
    if profile.size != 24:
        raise ValueError(
            f"Internal gain profile expects 24 hourly values, got {profile.size}."
        )

    gains = np.full(profile.shape, float(night_gain), dtype=float)
    max_occ = float(np.max(profile))
    if max_occ > 0.0:
        scaled = (profile / max_occ) * float(day_gain)
        day_mask = profile > 0.0
        hour_mask = np.arange(profile.size) < int(cutoff_hour)
        gains = np.where(day_mask & hour_mask, scaled, gains)

    return gains

# ここでは冷水一次側の簡易特性を設定し、ポンプや弁の圧力応答をおおまかに再現する
CHW_PUMP_PG = [100.0, -30.0, -20.0]  
CHW_PUMP_G_DESIGN = 0.6  
CHW_VALVE_CV_MAX = 70.0
CHW_VALVE_RANGEABILITY = 50.0
CHW_BRANCH_KR_EQ = 8.0  
CHW_BRANCH_KR_PIPE = 5.0  
CHW_BRANCH_FLOW_MAX = 0.7  
CHW_PUMP_INV_MIN = 0.45
CHW_PUMP_INV_MAX = 1.2
CHW_FLOW_TOL_KPA = 0.05


# ---------------------------------------------------------------------------
# 典型的な平日・週末の在室スケジュールをゾーン別に定義
# ---------------------------------------------------------------------------
# 典型的なオフィス平日・週末の時間別在室人数プロファイル（1時間ごと、0=0時）
_DEFAULT_ZONE_OCCUPANCY_WEEKDAY: Dict[str, List[float]] = {
    "Zone 1": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        14.0,
        17.0,
        15.0,
        16.0,
        17.0,
        13.0,
        13.0,
        15.0,
        12.0,
        11.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "Zone 2": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        5.0,
        8.0,
        10.0,
        13.0,
        12.0,
        9.0,
        11.0,
        13.0,
        8.0,
        8.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "Zone 3": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.0,
        10.0,
        12.0,
        11.0,
        12.0,
        11.0,
        9.0,
        8.0,
        7.0,
        8.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "Zone 4": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        6.0,
        7.0,
        6.0,
        5.0,
        5.0,
        4.0,
        3.0,
        3.0,
        3.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
}

DEFAULT_ZONE_OCCUPANCY_WEEKDAY: Dict[str, List[float]] = {
    zone: occupancy_with_cutoff(profile) for zone, profile in _DEFAULT_ZONE_OCCUPANCY_WEEKDAY.items()
}


_DEFAULT_ZONE_OCCUPANCY_WEEKEND: Dict[str, List[float]] = {
    "Zone 1": [0.0] * 24,
    "Zone 2": [0.0] * 24,
    "Zone 3": [0.0] * 24,
    "Zone 4": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        3.0,
        3.0,
        3.0,
        2.0,
        2.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
}

DEFAULT_ZONE_OCCUPANCY_WEEKEND: Dict[str, List[float]] = {
    zone: occupancy_with_cutoff(profile) for zone, profile in _DEFAULT_ZONE_OCCUPANCY_WEEKEND.items()
}

# ゾーンごとの機器特性の設定（全ファイルで共通）
_DEFAULT_ZONE_PARAMETER_ROWS: Tuple[Dict[str, float | str], ...] = (
    {
        "name": "Zone 1",  # ゾーン名
        "thermal_cap": 3.2e6,  # 熱容量 [J/K] - ゾーンの蓄熱容量
        "ua_env": 170.0,  # 外壁熱貫流率 [W/K] - 外気との熱伝達係数
        "infil_mdot": 0.018,  # 外気侵入量 [kg/s] - 隙間風などによる外気流入
        "flow_min": 0.08,  # 最小風量 [kg/s] - VAVダンパーの最小開度時の風量
        "flow_max": 0.42,  # 最大風量 [kg/s] - VAVダンパーの最大開度時の風量
        "initial_temp": 28.0,  # 初期温度 [°C] - シミュレーション開始時の室温
        "day_internal_gain": 3200.0,  # 昼間内部発熱 [W] - 人体・機器からの発熱（稼働時）
        "night_internal_gain": 300.0,  # 夜間内部発熱 [W] - 待機電力などの発熱（非稼働時）
    },
    {
        "name": "Zone 2",
        "thermal_cap": 2.8e6,
        "ua_env": 150.0,
        "infil_mdot": 0.016,
        "flow_min": 0.07,
        "flow_max": 0.36, # 設計風量 [kg/s]
        "initial_temp": 27.5,
        "day_internal_gain": 2800.0,
        "night_internal_gain": 280.0,
    },
    {
        "name": "Zone 3",
        "thermal_cap": 2.6e6,
        "ua_env": 145.0,
        "infil_mdot": 0.015,
        "flow_min": 0.065,
        "flow_max": 0.34, # 設計風量 [kg/s]
        "initial_temp": 27.2,
        "day_internal_gain": 2500.0,
        "night_internal_gain": 250.0,
    },
    {
        "name": "Zone 4",
        "thermal_cap": 3.0e6,
        "ua_env": 160.0,
        "infil_mdot": 0.017,
        "flow_min": 0.055,
        "flow_max": 0.32, # 設計風量 [kg/s]
        "initial_temp": 28.3,
        "day_internal_gain": 3000.0,
        "night_internal_gain": 300.0,
    },
)

@dataclass
class ZoneConfig:
    """ゾーンの熱容量・空調条件を集約した設定コンテナ。

    各ゾーン特有の熱負荷や風量制約を事前に登録し、時間ステップごとの状態計算で参照する。
    """

    # --- ゾーン物理特性や初期条件を保持するフィールド定義 ---
    name: str
    thermal_cap: float
    ua_env: float
    infil_mdot: float
    flow_min: float
    flow_max: float
    initial_temp: float
    day_internal_gain: float
    night_internal_gain: float
    design_flow_kg_s: Optional[float] = None
    design_static_pa: float = 700.0
    design_damper_pos: float = 0.5
    damper_authority: float = 0.5
    damper_min: float = 0.4
    leakage_m3_min: float = 0.02
    terminal_dp_pa: float = 0.0
    volume_m3: float = 250.0
    initial_co2_ppm: float = 420.0
    initial_rh: float = 50.0  # 初期相対湿度 [%]
    day_occupants: float = 6.0
    night_occupants: float = 0.0
    co2_gen_lps_per_person: float = 0.0055
    latent_gain_per_person: float = 0.06  # 人体からの潜熱負荷 [kW/person]
    occupant_schedule: Optional[Sequence[float]] = None
    occupant_schedule_weekend: Optional[Sequence[float]] = None
    internal_gain_schedule: Optional[Sequence[float]] = None
    internal_gain_schedule_weekend: Optional[Sequence[float]] = None  
    duct_key: Optional[str] = None
    damper_curve_key: Optional[str] = None


def build_default_zones() -> Tuple[ZoneConfig, ...]:
    """Return the canonical 4-zone configuration shared by all entry points."""

    zones: List[ZoneConfig] = []
    for params in _DEFAULT_ZONE_PARAMETER_ROWS:
        name = str(params["name"])
        weekday_schedule = DEFAULT_ZONE_OCCUPANCY_WEEKDAY[name]
        weekend_schedule = DEFAULT_ZONE_OCCUPANCY_WEEKEND[name]
        day_gain = float(params["day_internal_gain"])
        night_gain = float(params["night_internal_gain"])

        zones.append(
            ZoneConfig(
                name,
                thermal_cap=float(params["thermal_cap"]),
                ua_env=float(params["ua_env"]),
                infil_mdot=float(params["infil_mdot"]),
                flow_min=float(params["flow_min"]),
                flow_max=float(params["flow_max"]),
                initial_temp=float(params["initial_temp"]),
                day_internal_gain=day_gain,
                night_internal_gain=night_gain,
                design_static_pa=700.0,
                damper_authority=0.55,
                design_damper_pos=0.6,
                damper_min=0.4,
                leakage_m3_min=0.01,
                day_occupants=max(weekday_schedule),
                night_occupants=0.0,
                occupant_schedule=weekday_schedule,
                occupant_schedule_weekend=weekend_schedule,
                internal_gain_schedule=internal_gain_profile_from_occupancy(
                    weekday_schedule, day_gain=day_gain, night_gain=night_gain
                ),
                internal_gain_schedule_weekend=internal_gain_profile_from_occupancy(
                    weekend_schedule, day_gain=day_gain, night_gain=night_gain
                ),
            )
        )
    return tuple(zones)


@dataclass
class HVACActions:
    """各タイムステップで外部から与えるアクチュエータ指令値を保持する。"""

    # --- 供給ファンやダンパーなどの操作量をまとめた入力群 ---
    zone_dampers: Optional[List[float]] = None  
    oa_damper: Optional[float] = None  
    fan_speed: Optional[float] = None  
    coil_valve: Optional[float] = None  


_MISSING_GEOMETRY_KEYS: set[str] = set()
_MISSING_TAP_KEYS: set[str] = set()


def _branch_geometry_for_zone(config: ZoneConfig) -> BranchGeometry:
    key = config.duct_key or config.name
    geometry = ZONE_BRANCH_GEOMETRY.get(key)
    if geometry is not None:
        return geometry

    if key not in _MISSING_GEOMETRY_KEYS:
        warnings.warn(
            f"No duct geometry found for zone '{config.name}' (key='{key}'); "
            "using DEFAULT_BRANCH_GEOMETRY.",
            RuntimeWarning,
        )
        _MISSING_GEOMETRY_KEYS.add(key)
    return DEFAULT_BRANCH_GEOMETRY


def _resistance_from_geometry(geometry: Optional[BranchGeometry]) -> float:
    if geometry is None:
        return 0.0
    return geometry.resistance_coefficient(rho_air=RHO_AIR)


DAMPER_CURVES: Dict[str, tuple[tuple[float, float], ...]] = {
    "default": (
        (1.0, 0.020),
        (0.8, 0.200),
        (0.6, 1.000),
        (0.4, 2.000),
        (0.2, 5.000),
        (0.0, 999.9),
    ),
}

DEFAULT_DAMPER_CURVE_KEY = "default"


def _branch_components_from_geometry(geometry: Optional[BranchGeometry]) -> tuple[float, float]:
    if geometry is None:
        return 0.0, 0.0
    return geometry.resistance_components(rho_air=RHO_AIR)


def _trunk_segment_coefficients(geometry: BranchGeometry) -> np.ndarray:
    return np.array([seg.resistance_coefficient(rho_air=RHO_AIR) for seg in geometry.segments], dtype=float)


def _supply_tap_index_for_zone(config: ZoneConfig, *, max_index: int) -> int:
    key = config.duct_key or config.name
    tap = SUPPLY_TAP_INDEX.get(key)
    if tap is None:
        if key not in _MISSING_TAP_KEYS:
            warnings.warn(
                f"No supply tap index defined for zone '{config.name}' (key='{key}'); assigning to furthest node.",
                RuntimeWarning,
            )
            _MISSING_TAP_KEYS.add(key)
        tap = max_index
    return int(np.clip(tap, 0, max_index))


def _damper_curve_for_zone(config: ZoneConfig) -> tuple[tuple[float, float], ...]:
    key = config.damper_curve_key or DEFAULT_DAMPER_CURVE_KEY
    curve = DAMPER_CURVES.get(key)
    if curve is not None:
        return curve
    warnings.warn(
        f"Unknown damper_curve_key '{key}' for zone '{config.name}'; using '{DEFAULT_DAMPER_CURVE_KEY}'.",
        RuntimeWarning,
    )
    return DAMPER_CURVES[DEFAULT_DAMPER_CURVE_KEY]


def _damper_coef_at_position(curve: Sequence[Sequence[float]], position: float) -> float:
    if not curve:
        return 0.0
    pos = float(np.clip(position, 0.0, 1.0))
    n = len(curve)
    if pos >= curve[0][0]:
        return float(curve[0][1])
    if pos <= curve[-1][0]:
        return float(curve[-1][1])
    for idx in range(1, n):
        hi = curve[idx - 1]
        lo = curve[idx]
        if lo[0] <= pos <= hi[0]:
            span = max(hi[0] - lo[0], 1e-9)
            weight = (pos - lo[0]) / span
            return float(lo[1] + (hi[1] - lo[1]) * weight)
    return float(curve[-1][1])


class ZoneBranch:
    """Zone air branch backed by `phyvac.BranchA` and `phyvac.Damper`."""

    def __init__(self, config: ZoneConfig):
        self.config = config
        self.geometry = _branch_geometry_for_zone(config)
        self.leakage_m3_min = max(float(config.leakage_m3_min), 0.0)

        design_mdot = config.design_flow_kg_s if config.design_flow_kg_s is not None else config.flow_max
        self.design_vol_flow_m3_min = max(float(design_mdot) / RHO_AIR * 60.0, 1e-6)
        self.design_static_pa = max(float(config.design_static_pa), 5.0)
        self.damper_min = float(np.clip(config.damper_min, 0.0, 1.0))
        self.design_damper_pos = float(np.clip(config.design_damper_pos, self.damper_min, 1.0))
        self.damper_authority = float(np.clip(config.damper_authority, 0.01, 0.99))

        kr_duct_geom, kr_eq_geom = _branch_components_from_geometry(self.geometry)
        self.kr_duct = max(kr_duct_geom, 0.0)
        self.kr_eq_geom = max(kr_eq_geom, 0.0)
        self.kr_terminal = self._terminal_coefficient()
        self.kr_eq = self.kr_eq_geom + self.kr_terminal
        self.r_fixed = max(self.kr_duct + self.kr_eq, 1e-9)

        total_coeff = self.design_static_pa / (self.design_vol_flow_m3_min ** 2)
        min_fixed = (1.0 - self.damper_authority) * total_coeff
        if self.r_fixed < min_fixed:
            self.r_fixed = max(min_fixed, 1e-9)
        if self.r_fixed > total_coeff:
            warnings.warn(
                f"Zone '{config.name}' duct losses exceed design static pressure; "
                "increasing design_static_pa to maintain feasibility.",
                RuntimeWarning,
            )
            total_coeff = self.r_fixed
            self.design_static_pa = total_coeff * (self.design_vol_flow_m3_min ** 2)

        remaining_coeff = max(total_coeff - self.r_fixed, 0.0)

        base_curve = _damper_curve_for_zone(config)
        base_coef = _damper_coef_at_position(base_curve, self.design_damper_pos)
        if base_coef <= 0.0:
            base_coef = 1.0
        if remaining_coeff <= 0.0:
            damper_scale = 0.0
        else:
            damper_scale = remaining_coeff / base_coef
        self.damper_curve_key = config.damper_curve_key or DEFAULT_DAMPER_CURVE_KEY
        scaled_curve = [
            [float(pos), float(coeff * damper_scale)]
            for (pos, coeff) in base_curve
        ]
        self.damper = pv.Damper(coef=scaled_curve)
        self.branch = pv.BranchA(
            damper=self.damper,
            kr_eq=self.kr_eq,
            kr_duct=self.kr_duct,
        )

    def _terminal_coefficient(self) -> float:
        if self.config.terminal_dp_pa <= 0.0:
            return 0.0
        if self.design_vol_flow_m3_min <= 0.0:
            return 0.0
        return float(self.config.terminal_dp_pa) / (self.design_vol_flow_m3_min ** 2)

    def _apply_damper_position(self, position: float) -> float:
        eff = float(np.clip(position, 0.0, 1.0))
        eff = max(eff, self.damper_min)
        self.damper.damp = eff
        return eff

    def pressure_drop(self, g_m3_min: float, position: float) -> float:
        flow = float(max(g_m3_min, 0.0))
        if flow <= 0.0:
            return 0.0
        self._apply_damper_position(position)
        dp = self.branch.f2p(flow)
        return abs(float(dp))

    def flow_for_pressure(self, dp_target: float, position: float) -> float:
        if dp_target <= 0.0:
            self.branch.p2f(0.0)
            return 0.0
        eff = self._apply_damper_position(position)
        flow = float(max(self.branch.p2f(-float(dp_target)), 0.0))
        if eff <= self.damper_min + 1e-9 and self.leakage_m3_min > 0.0:
            flow = max(flow, self.leakage_m3_min)
        return flow


class VariableSpeedFan:
    """アフィニティ則に従う可変速ファンの簡略化モデル。

    標準のphyvacファンモデルはユーザー指定の設計点を十分に反映しないため、
    本クラスでは analytically 定義した二次曲線を用いて次の条件を満たすよう調整している。

    * 基準インバータ時に `design_flow_m3_min` で `design_head_pa` を発生させる。
    * 風量はインバータ比に一次比例、静圧は二乗比例させ、ファンの相似則を維持する。
    * ゼロ風量時でも有限の静圧を確保し、低風量域の静圧制御を破綻させない。
    """

    def __init__(
        self,
        design_flow_m3_min: float,
        design_head_pa: float,
        *,
        reference_inv: float = 1.0,
        zero_flow_margin: float = 1.35,
        peak_efficiency: float = 0.62,
        motor_efficiency: float = 0.88,
    ) -> None:
        # --- 設計点や効率パラメータを登録し、相似則ベースの特性曲線を生成 ---
        self.design_flow = float(max(design_flow_m3_min, 1e-3))
        self.design_head = float(max(design_head_pa, 1.0))
        self.reference_inv = float(max(reference_inv, 0.1))
        self.zero_flow_margin = float(max(zero_flow_margin, 1.05))
        self.peak_efficiency = float(np.clip(peak_efficiency, 0.1, 0.9))
        self.motor_efficiency = float(np.clip(motor_efficiency, 0.1, 0.99))

        # 実機に近い静圧-風量曲線を与えることで、ファン制御が物理的に破綻しないようにする
        self.g_zero = self.design_flow * self.zero_flow_margin
        ratio_ref = min(self.design_flow / max(self.g_zero * self.reference_inv, 1e-6), 0.9999)
        denom = (self.reference_inv ** 2) * max(1.0 - ratio_ref ** 2, 1e-6)
        self.dp0 = self.design_head / denom

        self.inv = self.reference_inv
        self.g = 0.0
        self.dp = 0.0
        self.pw = 0.0  
        self.flag = 0

    def f2p(self, flow_m3_min: float) -> float:
        # --- 現在の風量から静圧を計算し、ファン挙動を更新 ---
        flow_m3_min = float(max(flow_m3_min, 0.0))
        self.g = flow_m3_min

        if self.inv <= 0.0:
            self.dp = 0.0
            return self.dp

        # アフィニティ則に従い、ゼロ風量時圧力から二次曲線で現在の静圧を計算
        g_zero_scaled = self.g_zero * self.inv
        flow_ratio = flow_m3_min / max(g_zero_scaled, 1e-6)
        flow_ratio = min(flow_ratio, 1.0)

        dp_available = self.dp0 * self.inv ** 2
        self.dp = dp_available * max(0.0, 1.0 - flow_ratio ** 2)
        return self.dp

    def cal(self) -> None:
        # --- 最新の風量・静圧からファン電力を推定 ---
        if self.inv <= 0.0 or self.g <= 0.0:
            self.pw = 0.0
            self.flag = 0
            return

        flow_m3_s = self.g / 60.0
        air_power_w = flow_m3_s * self.dp

        design_flow_at_inv = self.design_flow * self.inv
        design_flow_at_inv = max(design_flow_at_inv, 1e-6)
        flow_ratio = self.g / design_flow_at_inv
        efficiency = self.peak_efficiency * max(0.0, 1.0 - 0.6 * (flow_ratio - 1.0) ** 2)
        efficiency = float(np.clip(efficiency, 0.08, self.peak_efficiency))

        # ファン効率とモーター効率を組み合わせ、必要電力[kW]を見積もる
        total_eff = max(efficiency * self.motor_efficiency, 0.08)
        self.pw = air_power_w / (total_eff * 1000.0)
        self.flag = 0


def _solve_branch_flow(branch: pv.BranchW, *, g_cap: float, tol: float) -> float:
    """枝配管内の圧力バランスが0kPa付近になる流量[m3/min]を探索する。"""

    # --- ポンプと配管損失の釣り合い点を反復計算で求める ---
    if g_cap <= 0.0:
        branch.f2p(0.0)
        return 0.0

    g_low = 0.0
    g_high = min(max(g_cap, 1e-3), CHW_BRANCH_FLOW_MAX)

    # まず上限流量を広げ、ポンプ揚程と配管損失が釣り合う領域を見つける
    dp_high = branch.f2p(g_high)
    attempts = 0
    while dp_high > 0.0 and g_high < CHW_BRANCH_FLOW_MAX:
        g_low = g_high
        g_high = min(g_high * 1.5 + 1e-6, CHW_BRANCH_FLOW_MAX)
        dp_high = branch.f2p(g_high)
        attempts += 1
        if attempts > 32:
            break

    if dp_high > 0.0:
        # ポンプ揚程が上限流量でもまだ支配的な場合、上限で固定して返す。
        branch.f2p(g_high)
        return g_high

    g_low = max(g_low, 0.0)
    g_high = max(g_high, g_low + 1e-6)

    for _ in range(48):
        g_mid = 0.5 * (g_low + g_high)
        dp_mid = branch.f2p(g_mid)
        if abs(dp_mid) <= tol:
            return g_mid
        if dp_mid > 0.0:
            g_low = g_mid
        else:
            g_high = g_mid

    # 収束しきらない場合でも、最後に評価した流量を返す
    branch.f2p(g_mid)
    return g_mid


def outdoor_temperature(ts: datetime) -> float:
    """気象庁10分値データを線形補間し、指定時刻の外気乾球温度を返す。"""
    minute_of_day = ts.hour * 60 + ts.minute
    minute_of_day = np.clip(minute_of_day, OUTDOOR_MINUTES[0], OUTDOOR_MINUTES[-1])
    # --- 時刻を分単位に変換して線形補間で外気温を取得 ---
    # numpy.interpで対象分の外気温を取得（外挿は避ける）
    return float(np.interp(minute_of_day, OUTDOOR_MINUTES, OUTDOOR_TEMPS))


def outdoor_relative_humidity(ts: datetime) -> float:
    """気象庁10分値データを線形補間し、指定時刻の外気相対湿度を返す。"""
    minute_of_day = ts.hour * 60 + ts.minute
    minute_of_day = np.clip(minute_of_day, OUTDOOR_MINUTES[0], OUTDOOR_MINUTES[-1])
    # --- 分単位の時刻を元に線形補間で外気相対湿度を推定 ---
    # numpy.interpで対象分の外気相対湿度を取得（外挿は避ける）
    return float(np.interp(minute_of_day, OUTDOOR_MINUTES, OUTDOOR_RHS))


def outdoor_absolute_humidity(ts: datetime) -> float:
    """指定時刻の外気絶対湿度[kg/kg(DA)]を返す。"""
    # --- 乾球温度と相対湿度を組み合わせて絶対湿度へ変換 ---
    return float(pv.tdb_rh2w(outdoor_temperature(ts), outdoor_relative_humidity(ts)))


def run_simulation(
    start: datetime,
    minutes: int,
    timestep_s: int,
    zones: List[ZoneConfig],
    chilled_water_temp: float = 7.0,
    coil_approach: float = 1.5,
    coil_ua: float = 3500.0,
    coil_dp_pa: float = 260.0,
    coil_bypass_factor: float = 0.1,
    oa_frac_min: float = 0.35,
    oa_flow_min: float = 0.05,
    fan_nominal_flow_m3_min: float = 100.0,
    static_pressure_limit: float = 800.0,
    mod_sp_floor: float = 250.0,
    mod_sp_ceiling: float = 500.0,
    # --- ASHRAE G36 Trim & Respond パラメータ ---
    # use_trim_respond: Trim & Respond を有効化するか（True で静圧SPをG36方式でリセット）
    use_trim_respond: bool = True,
    # tr_initial_sp: 空調開始時の静圧SP初期値（Noneなら上限寄りで開始しトリムダウン）
    tr_initial_sp: Optional[float] = None,
    # tr_min_sp / tr_max_sp: 静圧SPの下限 / 上限（未指定は mod_sp_floor / static_pressure_limit）
    tr_min_sp: Optional[float] = None,
    tr_max_sp: Optional[float] = None,
    # tr_trim_pa: サンプリング毎に常時かけるトリム量（負値で下げる）
    tr_trim_pa: float = -15.0,
    # tr_respond_pa: リクエスト超過1件あたり上げる静圧量（Respond成分）
    tr_respond_pa: float = 25.0,
    # tr_max_step_pa: 1回の更新で動かせる静圧ステップの絶対上限
    tr_max_step_pa: float = 75.0,
    # tr_sample_s: Trim & Respond を実行するサンプリング間隔 [s]
    tr_sample_s: int = 120,
    # tr_stability_wait_s: 立ち上がり安定化のためリセット開始を遅らせる時間 [s]
    tr_stability_wait_s: int = 600,
    # tr_request_threshold: ダンパー開度がこの値以上を「リクエスト」と数える
    tr_request_threshold: float = 0.95,
    # tr_ignore_requests: 無視するリクエスト本数（これを超えた分だけ Respond を加算）
    tr_ignore_requests: int = 2,
    default_fan_inv: float = 0.6,
    chw_pump_head_pa: float = 80000.0,
    chw_pump_efficiency: float = 0.8,
    chiller_cop: float = DEFAULT_CHILLER_COP,
    coil_pid_kp: float = 0.14,
    coil_pid_ti_s: float = 2000.0,
    coil_valve_min: float = 0.06,
    coil_valve_hold_deadband: float = 0.2,
    coil_valve_filter_alpha: float = 0.25,
    supply_air_setpoint: float = 15.0,
    setpoint: float = 26.0,
    action_callback: Optional[Callable[[datetime, np.ndarray, np.ndarray, np.ndarray], HVACActions]] = None,
    zone_pid_kp: float = 0.6,
    zone_pid_ti: float = 25.0,
    zone_pid_t_reset: int = 30,
    zone_pid_initial: float = 0.35,
    zone_pid_t_step: int = 1,
    zone_pid_max_step: float = 0.10,
    hvac_start_hour: float = 8.0,
    hvac_stop_hour: int = 18,
    verbose_steps: bool = False,
) -> pd.DataFrame:
    """VAVシステムの逐次シミュレーションを実行し、結果をDataFrameで返す。

    日時やゾーン条件、PIDパラメータなどを引数で受け取り、
    1分刻みで空調設備とゾーン熱環境の状態量を積分する。
    コイル・ファン・ポンプ・ダンパーの指令値や負荷も時系列で記録される。
    """
    # === シミュレーション準備: 時刻・効率補正・各種キャッシュを初期化 ===
    steps = int((minutes * 60) / timestep_s)
    current_time = start
    pump_efficiency = float(np.clip(chw_pump_efficiency, 0.05, 0.95))
    bypass_factor = float(np.clip(coil_bypass_factor, 0.0, 0.95))

    # 各ゾーンVAV用PIDを生成し、ダンパー開度の自己調整が行えるようにする
    zone_pid_ti = float(max(zone_pid_ti, 1e-6))
    zone_pid_initial = float(np.clip(zone_pid_initial, 0.0, 1.0))
    zone_pid_t_reset = int(max(zone_pid_t_reset, 1))
    zone_pid_t_step = int(max(zone_pid_t_step, 1))
    zone_pid_max_step = float(max(zone_pid_max_step, 1e-6))
    coil_valve_hold_deadband = float(max(coil_valve_hold_deadband, 0.0))
    coil_valve_filter_alpha = float(np.clip(coil_valve_filter_alpha, 0.0, 1.0))
    # Trim & Respond parameters (ASHRAE G36)
    use_trim_respond = bool(use_trim_respond)
    tr_sample_s = int(max(tr_sample_s, 1))
    tr_stability_wait_s = int(max(tr_stability_wait_s, 0))
    tr_ignore_requests = int(max(tr_ignore_requests, 0))
    tr_request_threshold = float(np.clip(tr_request_threshold, 0.0, 1.0))
    tr_min_sp_val = float(mod_sp_floor if tr_min_sp is None else tr_min_sp)
    tr_min_sp_val = float(np.clip(tr_min_sp_val, 50.0, static_pressure_limit))
    tr_max_sp_val = float(
        static_pressure_limit
        if tr_max_sp is None
        else np.clip(tr_max_sp, tr_min_sp_val, static_pressure_limit)
    )
    tr_initial_sp_val = tr_initial_sp
    if tr_initial_sp_val is None:
        # Start from the upper end so the system can safely trim down
        tr_initial_sp_val = float(min(max(mod_sp_ceiling, tr_min_sp_val), tr_max_sp_val))
    tr_initial_sp_val = float(np.clip(tr_initial_sp_val, tr_min_sp_val, tr_max_sp_val))
    tr_trim_pa = float(tr_trim_pa)
    tr_respond_pa = float(tr_respond_pa)
    tr_max_step_pa = float(max(tr_max_step_pa, 0.0))
    # --- 各ゾーンに対応したPID制御器を生成し、初期ゲインと出力を設定 ---
    pids: List[pv.PID] = []
    for _ in zones:
        pid = pv.PID(
            kp=zone_pid_kp,
            ti=zone_pid_ti,
            a_min=0.0,
            a_max=1.0,
            t_step=zone_pid_t_step,
            kg=-1,
            t_reset=zone_pid_t_reset,
            a=zone_pid_initial,
            max_step=zone_pid_max_step,
        )
        pid.a = zone_pid_initial
        pids.append(pid)

    # --- ゾーン配管モデルを構築し、設計風量から供給系の基準圧力を決定 ---
    branches = [ZoneBranch(zone) for zone in zones]
    supply_trunk_coeff = _resistance_from_geometry(SUPPLY_TRUNK_GEOMETRY)
    return_trunk_coeff = _resistance_from_geometry(RETURN_TRUNK_GEOMETRY)
    supply_trunk_segment_coeffs = _trunk_segment_coefficients(SUPPLY_TRUNK_GEOMETRY)
    supply_trunk_node_count = supply_trunk_segment_coeffs.size + 1
    max_tap_index = max(supply_trunk_segment_coeffs.size, 0)
    zone_tap_indices = np.array(
        [_supply_tap_index_for_zone(zone, max_index=max_tap_index) for zone in zones],
        dtype=int,
    )
    zone_tap_indices = np.clip(zone_tap_indices, 0, supply_trunk_node_count - 1)
    if supply_trunk_segment_coeffs.size > 0 and len(zones) > 0:
        segment_indices = np.arange(supply_trunk_segment_coeffs.size, dtype=int).reshape(-1, 1)
        segment_downstream_mask = (zone_tap_indices.reshape(1, -1) > segment_indices).astype(float)
    else:
        segment_downstream_mask = np.zeros((supply_trunk_segment_coeffs.size, len(zones)), dtype=float)
    # ゾーン設計値から供給ファンの設計風量・静圧を推定し、制御の基準を作る
    design_volumes = []
    design_branch_dps = []
    for branch, zone in zip(branches, zones):
        design_mdot = zone.design_flow_kg_s if zone.design_flow_kg_s is not None else zone.flow_max
        design_vol = max(design_mdot / RHO_AIR * 60.0, 1e-6)
        design_volumes.append(design_vol)
        dp_branch = branch.pressure_drop(design_vol, zone.design_damper_pos)
        design_branch_dps.append(dp_branch)

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
    design_total_head = max(coil_dp_pa + design_supply_static, 150.0)

    coil_design_flow_m3_min = max(design_flow_target, 0.05)
    if coil_dp_pa > 0.0:
        coil_dp_coeff = coil_dp_pa / max(coil_design_flow_m3_min ** 2, 1e-6)
    else:
        coil_dp_coeff = 0.0

    def coil_pressure_drop(total_vol_flow: float) -> float:
        flow = max(float(total_vol_flow), 0.0)
        return coil_dp_coeff * (flow ** 2)

    def trunk_pressure_drop(coeff: float, total_vol_flow: float) -> float:
        if coeff <= 0.0:
            return 0.0
        flow = max(float(total_vol_flow), 0.0)
        return coeff * (flow ** 2)


    # 推定した設計点を反映させた可変速ファンモデルを用い、静圧と風量を一貫管理
    # --- 推定した設計点を反映するファンモデルを用意し、インバータ制御の基準点を確保 ---
    fan = VariableSpeedFan(
        design_flow_m3_min=design_flow_target,
        design_head_pa=design_total_head,
        reference_inv=max(default_fan_inv, 0.1),
    )

    max_fan_inv = 1.5

    # Trim & Respond (G36) state
    tr_sp = tr_initial_sp_val
    tr_elapsed_enable_s = 0.0
    tr_elapsed_sample_s = 0.0
    prev_hvac_on = False


    # --- 冷水ループの弁・ポンプ・枝モデルを生成し、コイル負荷計算に備える ---
    chw_valve = pv.Valve(cv_max=CHW_VALVE_CV_MAX, r=CHW_VALVE_RANGEABILITY)
    chw_pump = pv.Pump(pg=CHW_PUMP_PG, g_d=CHW_PUMP_G_DESIGN, inv=CHW_PUMP_INV_MIN, figure=0)
    chw_branch = pv.BranchW(
        pump=chw_pump,
        valve=chw_valve,
        kr_eq=CHW_BRANCH_KR_EQ,
        kr_pipe=CHW_BRANCH_KR_PIPE,
    )

    # --- 水-空気熱交換器の特性を希望UA値へ合わせ、コイル挙動を調整 ---
    coil_hex = pv.HeatExchangerW2A()
    coil_nominal_area = float(coil_hex.area_surface)
    rated_ua_dry_kw = coil_hex.rated_coef_dry * max(coil_nominal_area, 1e-6)
    target_ua_kw = coil_ua / 1000.0
    if target_ua_kw > 0.0 and rated_ua_dry_kw > 0.0:
        area_scale = target_ua_kw / rated_ua_dry_kw
        coil_base_area = coil_nominal_area * area_scale
        if coil_base_area > 0.0:
            coil_hex.area_surface = max(coil_base_area, 1e-8)
    else:
        coil_base_area = 0.0
    # コイル面積は初期化時に一度だけ決め、以降は弁開度で変調しない

    # --- 単位変換ユーティリティ: 体積流量と質量流量を相互変換 ---
    def vol_to_mdot(flow_m3_min: float) -> float:
        return (float(flow_m3_min) / 60.0) * RHO_AIR

    def mdot_to_vol(flow_kg_s: float) -> float:
        return (float(flow_kg_s) / RHO_AIR) * 60.0

    def solve_air_network(damper_positions: np.ndarray, hvac_active: bool):
        # --- 指定ダンパー開度とファン指令から供給ネットワークを解く主計算ブロック ---
        # 指定したダンパー開度とファン指令から、幹ノード静圧と枝流量の釣り合い点を探索
        n = len(branches)
        zero_array = np.zeros(n, dtype=float)
        if not hvac_active or fan.inv <= 0.0 or n == 0:
            fan.g = 0.0
            fan.dp = 0.0
            return zero_array, zero_array, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        positions = np.clip(damper_positions, 0.0, 1.0)
        segment_count = supply_trunk_segment_coeffs.size
        node_count = supply_trunk_node_count

        def _compute_zone_flows(node_pressures: np.ndarray) -> np.ndarray:
            flows = np.zeros(n, dtype=float)
            for idx in range(n):
                tap = zone_tap_indices[idx]
                tap = int(min(max(tap, 0), node_count - 1))
                tap_pressure = max(float(node_pressures[tap]), 0.0)
                flows[idx] = branches[idx].flow_for_pressure(tap_pressure, positions[idx])
            return flows

        def _propagate_nodes(node0: float, trunk_flows: np.ndarray) -> np.ndarray:
            nodes = np.empty(node_count, dtype=float)
            nodes[0] = max(float(node0), 0.0)
            if segment_count == 0:
                return nodes
            for seg_idx in range(segment_count):
                flow_seg = float(trunk_flows[seg_idx]) if seg_idx < trunk_flows.size else 0.0
                coeff = float(supply_trunk_segment_coeffs[seg_idx])
                drop = coeff * (flow_seg ** 2) if flow_seg > 0.0 and coeff > 0.0 else 0.0
                nodes[seg_idx + 1] = max(nodes[seg_idx] - drop, 0.0)
            return nodes

        def _solve_trunk(node0: float, init_nodes: Optional[np.ndarray] = None) -> tuple:
            node0 = max(float(node0), 0.0)
            if node_count == 0:
                nodes = np.zeros(1, dtype=float)
                flows = np.zeros(n, dtype=float)
                trunk_flows = np.zeros(0, dtype=float)
                return flows, nodes, trunk_flows

            if init_nodes is not None and init_nodes.size == node_count:
                nodes = np.array(init_nodes, dtype=float, copy=True)
                nodes = np.clip(nodes, 0.0, node0)
                nodes[0] = node0
            else:
                nodes = np.full(node_count, node0, dtype=float)

            flows = _compute_zone_flows(nodes)
            relax = 0.65
            for _ in range(36):
                if segment_count:
                    trunk_flows = segment_downstream_mask @ flows
                else:
                    trunk_flows = np.zeros(0, dtype=float)
                nodes_next = _propagate_nodes(node0, trunk_flows)
                diff_nodes = float(np.max(np.abs(nodes_next - nodes))) if node_count else 0.0
                nodes = nodes + relax * (nodes_next - nodes)
                flows_next = _compute_zone_flows(nodes)
                diff_flows = float(np.max(np.abs(flows_next - flows))) if flows.size else 0.0
                flows = flows_next
                if max(diff_nodes, diff_flows) <= 5e-4:
                    break
            if segment_count:
                trunk_flows = segment_downstream_mask @ flows
            else:
                trunk_flows = np.zeros(0, dtype=float)
            nodes = _propagate_nodes(node0, trunk_flows)
            return flows, nodes, trunk_flows

        last_nodes_hint: Optional[np.ndarray] = None

        def _evaluate_node0(node0: float, *, node_hint: Optional[np.ndarray] = None) -> Dict[str, object]:
            nonlocal last_nodes_hint
            flows, nodes, trunk_flows = _solve_trunk(node0, init_nodes=node_hint)
            total_vol = float(np.sum(flows))
            coil_dp_val = float(coil_pressure_drop(total_vol))
            return_dp_val = float(trunk_pressure_drop(return_trunk_coeff, total_vol))
            fan_dp_val = float(fan.f2p(max(total_vol, 0.0)))
            residual = fan_dp_val - coil_dp_val - return_dp_val - node0
            if zone_tap_indices.size:
                zone_dp_vals = np.array([nodes[min(tap, node_count - 1)] for tap in zone_tap_indices], dtype=float)
                far_tap = int(np.max(zone_tap_indices))
                far_tap = min(max(far_tap, 0), node_count - 1)
                supply_trunk_dp = float(nodes[0] - nodes[far_tap])
            else:
                zone_dp_vals = np.zeros(0, dtype=float)
                supply_trunk_dp = 0.0
            last_nodes_hint = nodes
            return {
                "flows": flows,
                "nodes": nodes,
                "total_vol": total_vol,
                "coil_dp": coil_dp_val,
                "return_dp": return_dp_val,
                "fan_dp": fan_dp_val,
                "residual": residual,
                "zone_dp": zone_dp_vals,
                "supply_trunk_dp": supply_trunk_dp,
                "node0": float(node0),
            }

        node0_min = 0.5
        state_low = _evaluate_node0(node0_min)
        attempts = 0
        while state_low["residual"] < 0.0 and node0_min > 1e-3 and attempts < 8:
            node0_min *= 0.5
            state_low = _evaluate_node0(node0_min, node_hint=state_low["nodes"])
            attempts += 1

        node0_max = max(static_pressure_limit, node0_min + 5.0)
        node0_high = min(max(node0_min * 1.5, node0_min + 5.0), node0_max)
        state_high = _evaluate_node0(node0_high, node_hint=state_low["nodes"])
        attempts = 0
        while state_high["residual"] > 0.0 and node0_high < node0_max - 1e-6 and attempts < 24:
            node0_min = node0_high
            state_low = state_high
            node0_high = min(node0_high * 1.6 + 2.0, node0_max)
            state_high = _evaluate_node0(node0_high, node_hint=state_low["nodes"])
            attempts += 1

        chosen_state = state_low
        if state_low["residual"] >= 0.0 and state_high["residual"] <= 0.0:
            lo = node0_min
            hi = node0_high
            state_lo = state_low
            state_hi = state_high
            for _ in range(32):
                r_lo = state_lo["residual"]
                r_hi = state_hi["residual"]
                if abs(r_hi - r_lo) > 1e-9:
                    mid = hi - r_hi * (hi - lo) / (r_hi - r_lo)
                else:
                    mid = 0.5 * (lo + hi)
                if mid <= min(lo, hi) or mid >= max(lo, hi):
                    mid = 0.5 * (lo + hi)
                state_mid = _evaluate_node0(mid, node_hint=state_hi["nodes"])
                chosen_state = state_mid
                if abs(state_mid["residual"]) <= 0.05:
                    break
                if state_mid["residual"] > 0.0:
                    lo = mid
                    state_lo = state_mid
                else:
                    hi = mid
                    state_hi = state_mid
                if abs(hi - lo) <= 0.25:
                    chosen_state = state_mid
                    break
            else:
                chosen_state = state_mid
        elif state_high["residual"] > 0.0:
            chosen_state = state_high

        flows_sol = np.array(chosen_state["flows"], dtype=float)
        zone_dp = np.array(chosen_state["zone_dp"], dtype=float)
        sp_sol = float(chosen_state["node0"])
        total_vol = float(chosen_state["total_vol"])
        fan_dp_final = float(chosen_state["fan_dp"])
        coil_dp_final = float(chosen_state["coil_dp"])
        supply_dp_final = float(chosen_state["supply_trunk_dp"])
        return_dp_final = float(chosen_state["return_dp"])

        return (
            flows_sol,
            zone_dp,
            sp_sol,
            total_vol,
            fan_dp_final,
            coil_dp_final,
            supply_dp_final,
            return_dp_final,
        )

    def network_at_inv(inv: float, positions: np.ndarray) -> tuple:
        # --- ファンインバータ値を与えて空気ネットワークの応答を取得 ---
        fan.inv = float(np.clip(inv, 0.0, max_fan_inv))
        (
            flows,
            zone_dp,
            sp,
            total_vol,
            fan_dp_val,
            coil_dp_val,
            supply_dp_val,
            return_dp_val,
        ) = solve_air_network(positions, True)
        return (
            fan.inv,
            flows,
            zone_dp,
            sp,
            total_vol,
            fan_dp_val,
            coil_dp_val,
            supply_dp_val,
            return_dp_val,
        )

    def balance_fan_to_target(target_mdot: float, positions: np.ndarray) -> tuple:
        # --- 指定質量流量を達成するようファン速度を探索する補助ルーチン ---
        # 目標質量流量に収束するよう、インバータ指令を段階拡張＋二分探索で決定
        if target_mdot <= 1e-9:
            fan.inv = 0.0
            fan.g = 0.0
            fan.dp = 0.0
            return (
                np.zeros(len(zones), dtype=float),
                np.zeros(len(zones), dtype=float),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

        res_low = network_at_inv(0.0, positions)
        if vol_to_mdot(res_low[4]) >= target_mdot:
            return res_low[1:]

        inv_high = max(default_fan_inv, 0.2)
        res_high = network_at_inv(inv_high, positions)

        # まず推定インバータ値を徐々に増やし、目標流量を上回る上限側の解を確保する
        while vol_to_mdot(res_high[4]) < target_mdot and res_high[0] < max_fan_inv - 1e-6:
            next_inv = min(res_high[0] * 1.3 + 0.02, max_fan_inv)
            res_high = network_at_inv(next_inv, positions)
            if abs(res_high[0] - max_fan_inv) <= 1e-6:
                break

        if vol_to_mdot(res_high[4]) < target_mdot:
            return res_high[1:]

        res_a = res_low
        res_b = res_high

        tolerance = max(0.002 * target_mdot, 5e-4)
        # 上限・下限を得たら二分探索で目標流量に近いインバータ解を狭めていく
        for _ in range(40):
            inv_mid = 0.5 * (res_a[0] + res_b[0])
            res_mid = network_at_inv(inv_mid, positions)
            mid_mdot = vol_to_mdot(res_mid[4])

            if abs(mid_mdot - target_mdot) <= tolerance:
                return res_mid[1:]

            if mid_mdot < target_mdot:
                res_a = res_mid
            else:
                res_b = res_mid

        if abs(vol_to_mdot(res_a[4]) - target_mdot) <= abs(vol_to_mdot(res_b[4]) - target_mdot):
            if fan.inv != res_a[0]:
                network_at_inv(res_a[0], positions)
            return res_a[1:]

        if fan.inv != res_b[0]:
            network_at_inv(res_b[0], positions)
        return res_b[1:]

    def balance_fan_to_static(sp_target_pa: float, positions: np.ndarray) -> tuple:
        # --- 静圧基準制御の挙動を模擬し、最も開いているダンパーに合わせて風量を確保 ---
        # 目標静圧を基準にファンインバータを調整し、Most-Open-Damper型の制御を模擬する
        sp_target_pa = float(np.clip(sp_target_pa, 100.0, static_pressure_limit))

        inv_lo, inv_hi = 0.0, max_fan_inv
        fan.inv = inv_lo
        res_lo = solve_air_network(positions, True)
        if res_lo[2] >= sp_target_pa:
            return res_lo

        fan.inv = inv_hi
        res_hi = solve_air_network(positions, True)
        if res_hi[2] < sp_target_pa:
            return res_hi

        best_lo = (inv_lo, *res_lo)
        best_hi = (inv_hi, *res_hi)

        for _ in range(36):
            inv_mid = 0.5 * (inv_lo + inv_hi)
            fan.inv = inv_mid
            (
                flows_mid,
                zone_dp_mid,
                sp_mid,
                total_vol_mid,
                fan_dp_mid,
                coil_dp_mid,
                supply_dp_mid,
                return_dp_mid,
            ) = solve_air_network(positions, True)

            if abs(sp_mid - sp_target_pa) <= 3.0:
                return (
                    flows_mid,
                    zone_dp_mid,
                    sp_mid,
                    total_vol_mid,
                    fan_dp_mid,
                    coil_dp_mid,
                    supply_dp_mid,
                    return_dp_mid,
                )

            if sp_mid < sp_target_pa:
                inv_lo = inv_mid
                best_lo = (
                    inv_mid,
                    flows_mid,
                    zone_dp_mid,
                    sp_mid,
                    total_vol_mid,
                    fan_dp_mid,
                    coil_dp_mid,
                    supply_dp_mid,
                    return_dp_mid,
                )
            else:
                inv_hi = inv_mid
                best_hi = (
                    inv_mid,
                    flows_mid,
                    zone_dp_mid,
                    sp_mid,
                    total_vol_mid,
                    fan_dp_mid,
                    coil_dp_mid,
                    supply_dp_mid,
                    return_dp_mid,
                )

        if abs(best_lo[3] - sp_target_pa) <= abs(best_hi[3] - sp_target_pa):
            fan.inv = best_lo[0]
            return best_lo[1:]

        fan.inv = best_hi[0]
        return best_hi[1:]

    # --- ゾーン初期状態（温度・CO2・湿度）をベクトル化して状態変数に格納 ---
    zone_temps = np.array([zone.initial_temp for zone in zones], dtype=float)
    zone_co2_ppm = np.array([zone.initial_co2_ppm for zone in zones], dtype=float)
    zone_rh = np.array([zone.initial_rh for zone in zones], dtype=float)  # ゾーン相対湿度
    zone_abs_hum = np.array([pv.tdb_rh2w(zone.initial_temp, zone.initial_rh) for zone in zones], dtype=float)  # ゾーン絶対湿度

    def _prepare_occupancy_schedule(values: Sequence[float], zone_name: str, label: str) -> np.ndarray:
        arr = np.array(list(values), dtype=float).flatten()
        if arr.size != 24:
            raise ValueError(
                f"{zone_name} {label} occupancy schedule must provide 24 hourly values, got {arr.size}."
            )
        return arr

    def _prepare_internal_gain_schedule(occupancy: Sequence[float], day_gain: float, night_gain: float, zone_name: str, label: str) -> np.ndarray:
        arr = np.array(list(occupancy), dtype=float).flatten()
        if arr.size != 24:
            raise ValueError(
                f"{zone_name} {label} internal gain schedule must provide 24 hourly values, got {arr.size}."
            )
        max_occ = np.max(arr)
        if max_occ > 0:
            scaled = (arr / max_occ) * day_gain
        else:
            scaled = np.full(24, night_gain)
        # 0の時はnight_gainに設定
        scaled[arr == 0] = night_gain
        return scaled

    # --- 在室スケジュールと内部発熱スケジュールをゾーンごとに正規化 ---
    occupant_profiles_weekday: List[Optional[np.ndarray]] = []
    occupant_profiles_weekend: List[Optional[np.ndarray]] = []
    for zone in zones:
        weekday_profile: Optional[np.ndarray] = None
        weekend_profile: Optional[np.ndarray] = None
        if zone.occupant_schedule is not None:
            weekday_profile = _prepare_occupancy_schedule(zone.occupant_schedule, zone.name, "weekday")
        if zone.occupant_schedule_weekend is not None:
            weekend_profile = _prepare_occupancy_schedule(zone.occupant_schedule_weekend, zone.name, "weekend")
        occupant_profiles_weekday.append(weekday_profile)
        occupant_profiles_weekend.append(weekend_profile)

    # --- コイル制御用の積分項とロギングバッファを初期化 ---
    coil_integral = 0.0  # コイルPIDの積分項（凍結防止のため範囲制限あり）
    coil_valve_state = 0.0
    last_supply_temp = float("nan")
    records = []
    for step in range(steps):
        # 1分ごとの離散ステップで、営業時間のみ空調を稼働させる
        current_hour = current_time.hour + current_time.minute / 60.0
        hour_idx = current_time.hour
        hvac_on = hvac_start_hour <= current_hour < hvac_stop_hour
        outdoor_temp = float(outdoor_temperature(current_time))
        outdoor_rh = float(outdoor_relative_humidity(current_time))

        actions = HVACActions()
        if action_callback is not None:
            user_actions = action_callback(
                current_time,
                zone_temps.copy(),
                zone_co2_ppm.copy(),
                zone_rh.copy(),
            )
            if user_actions is not None:
                actions = user_actions

        # ゾーンごとの希望風量・計算結果・内部熱取得を格納する作業配列を確保
        zone_requested_flows = np.zeros(len(zones))
        zone_vol_flows = np.zeros(len(zones))
        zone_flows = np.zeros(len(zones))
        zone_dp_pa = np.zeros(len(zones))
        damper_positions = np.zeros(len(zones))
        internal_gains = np.zeros(len(zones))

        # --- 在室人数と内部発熱を算出するため、曜日種別を判定 ---
        is_weekend = current_time.weekday() >= 5
        # --- ゾーンごとの時間別在室人数をプロファイルから取得 ---
        occupant_counts = np.zeros(len(zones))
        # --- 各ゾーンで熱・CO2・湿度バランス方程式を解き、次時刻の状態を更新 ---
        for idx, zone in enumerate(zones):
            weekday_profile = occupant_profiles_weekday[idx]
            weekend_profile = occupant_profiles_weekend[idx]
            profile: Optional[np.ndarray]
            if is_weekend:
                profile = weekend_profile if weekend_profile is not None else None
            else:
                profile = weekday_profile

            if profile is not None:
                occ = float(np.clip(profile[hour_idx], 0.0, None))
            else:
                if is_weekend and weekday_profile is not None:
                    occ = 0.0
                else:
                    occ = zone.day_occupants if hvac_on else zone.night_occupants
            if hour_idx >= OCCUPANCY_CUTOFF_HOUR:
                occ = 0.0
            occupant_counts[idx] = occ

        # 新しいループでinternal_gainsを設定
        # --- 内部発熱スケジュールを適用し、ゾーン顕熱負荷を決定 ---
        internal_gains = np.zeros(len(zones))
        for idx, zone in enumerate(zones):
            weekday_internal = zone.internal_gain_schedule
            weekend_internal = zone.internal_gain_schedule_weekend
            profile_internal: Optional[np.ndarray]
            if is_weekend:
                profile_internal = weekend_internal
            else:
                profile_internal = weekday_internal
            if profile_internal is not None:
                internal = float(profile_internal[hour_idx])
            else:
                internal = zone.day_internal_gain if hvac_on else zone.night_internal_gain
            if hour_idx >= OCCUPANCY_CUTOFF_HOUR or occupant_counts[idx] <= 0.0:
                internal = zone.night_internal_gain
            internal_gains[idx] = internal

        for idx, zone in enumerate(zones):
            if hvac_on:
                # 営業時間中はPIDを有効にし、設定温度との差分からダンパー開度を決める
                pids[idx].mode = 1
                if actions.zone_dampers is not None and idx < len(actions.zone_dampers):
                    raw_pos = float(np.clip(actions.zone_dampers[idx], 0.0, 1.0))
                    apply_floor = False  # RL/LLMなど外部指令はそのまま通す
                else:
                    # PID出力は a_min/a_max で0-1にクリップされているが、念のため上限を再確認
                    raw_pos = float(np.clip(pids[idx].control(sp=setpoint, mv=zone_temps[idx]), 0.0, 1.0))
                    apply_floor = True  # ベースラインPIDのときのみ最小開度を適用

                eff_pos = max(raw_pos, zone.damper_min) if apply_floor else raw_pos
                damper_positions[idx] = eff_pos
                zone_requested_flows[idx] = zone.flow_min + eff_pos * (
                    zone.flow_max - zone.flow_min
                )
            else:
                # 夜間はダンパーを閉じ、内部発熱も低位に切り替える
                pids[idx].mode = 0
                pids[idx].a = 0.0
                pids[idx].sig = 0.0
                damper_positions[idx] = 0.0
                zone_requested_flows[idx] = 0.0

        # --- 営業時間内かどうかでファン制御・給外気制御を切り替える ---
        static_pressure_sp = 0.0
        if hvac_on:
            # Trim & Respond per ASHRAE G36: reset duct static pressure setpoint based on damper requests
            if not prev_hvac_on:
                tr_sp = tr_initial_sp_val
                tr_elapsed_enable_s = 0.0
                tr_elapsed_sample_s = 0.0

            tr_elapsed_enable_s += timestep_s
            tr_elapsed_sample_s += timestep_s

            if use_trim_respond and tr_elapsed_enable_s >= tr_stability_wait_s:
                if tr_elapsed_sample_s >= tr_sample_s:
                    tr_elapsed_sample_s = 0.0
                    requests = int(np.sum(damper_positions >= tr_request_threshold))
                    delta = tr_trim_pa
                    if requests > tr_ignore_requests:
                        respond_delta = tr_respond_pa * (requests - tr_ignore_requests)
                        respond_delta = float(
                            np.clip(respond_delta, -tr_max_step_pa, tr_max_step_pa)
                        )
                        delta += respond_delta
                    delta = float(np.clip(delta, -tr_max_step_pa, tr_max_step_pa))
                    tr_sp = float(np.clip(tr_sp + delta, tr_min_sp_val, tr_max_sp_val))
            else:
                # Keep sampling clock bounded even if TR disabled
                tr_elapsed_sample_s = float(min(tr_elapsed_sample_s, tr_sample_s))

            if use_trim_respond:
                static_pressure_sp = tr_sp
            else:
                mod = float(np.max(damper_positions)) if damper_positions.size else 0.0
                sp_floor = float(np.clip(mod_sp_floor, 100.0, static_pressure_limit))
                sp_ceiling = float(np.clip(mod_sp_ceiling, sp_floor, static_pressure_limit))
                if sp_ceiling <= sp_floor + 1.0:
                    sp_ceiling = sp_floor
                mod_norm = float(np.clip((mod - 0.95) / (1.0 - 0.95), 0.0, 1.0))
                static_pressure_sp = sp_floor + (sp_ceiling - sp_floor) * mod_norm

            if actions.fan_speed is not None:
                fan.inv = float(np.clip(actions.fan_speed, 0.0, max_fan_inv))
                (
                    zone_vol_flows,
                    zone_dp_pa,
                    supply_static_pa,
                    total_flow_m3_min,
                    fan_dp_network_pa,
                    coil_dp_active_pa,
                    supply_trunk_dp_pa,
                    return_trunk_dp_pa,
                ) = solve_air_network(damper_positions, True)
            else:
                (
                    zone_vol_flows,
                    zone_dp_pa,
                    supply_static_pa,
                    total_flow_m3_min,
                    fan_dp_network_pa,
                    coil_dp_active_pa,
                    supply_trunk_dp_pa,
                    return_trunk_dp_pa,
                ) = balance_fan_to_static(static_pressure_sp, damper_positions)
        else:
            tr_sp = tr_initial_sp_val
            tr_elapsed_enable_s = 0.0
            tr_elapsed_sample_s = 0.0
            fan.inv = 0.0
            zone_vol_flows = np.zeros(len(zones), dtype=float)
            zone_dp_pa = np.zeros(len(zones), dtype=float)
            supply_static_pa = 0.0
            total_flow_m3_min = 0.0
            fan_dp_network_pa = 0.0
            coil_dp_active_pa = 0.0
            supply_trunk_dp_pa = 0.0
            return_trunk_dp_pa = 0.0
            mixed_co2_ppm = OUTDOOR_CO2_PPM
            supply_co2_ppm = OUTDOOR_CO2_PPM

        zone_flows = (zone_vol_flows / 60.0) * RHO_AIR
        total_flow = float(np.sum(zone_flows))
        fan_flow_m3_min = float(total_flow_m3_min)

        if total_flow > 1e-6:
            return_co2_ppm = float(np.sum(zone_flows * zone_co2_ppm) / total_flow)
            return_abs_hum = float(np.sum(zone_flows * zone_abs_hum) / total_flow)  # 還気絶対湿度
        else:
            return_co2_ppm = float(np.mean(zone_co2_ppm))
            return_abs_hum = float(np.mean(zone_abs_hum))

        if not hvac_on or total_flow <= 1e-6:
            mixed_co2_ppm = OUTDOOR_CO2_PPM if not hvac_on else return_co2_ppm
            supply_co2_ppm = mixed_co2_ppm
            coil_valve_state = 0.0

        # --- 求めた総風量をファンモデルへ反映し、電力や静圧を更新 ---
        fan.g = float(max(fan_flow_m3_min, 0.0))
        fan.cal()
        fan_dp_pa = float(fan_dp_network_pa)
        coil_dp_active = float(coil_dp_active_pa)
        supply_trunk_dp_pa = float(supply_trunk_dp_pa)
        return_trunk_dp_pa = float(return_trunk_dp_pa)

        return_temp = float(np.average(zone_temps, weights=np.maximum(zone_flows, 1e-9)))
        mixed_temp = outdoor_temp
        mixed_abs_hum = pv.tdb_rh2w(outdoor_temp, outdoor_rh)  # 混合空気の絶対湿度（実際の外気相対湿度を使用）
        supply_temp = np.nan
        supply_abs_hum = mixed_abs_hum  # 給気絶対湿度
        coil_load_sensible_w = 0.0
        coil_load_latent_w = 0.0  # 潜熱負荷[W]
        total_coil_load_w = 0.0
        coil_valve = 0.0
        oa_flow = 0.0
        recirc_flow = 0.0
        oa_damper = 0.0
        ra_damper = 0.0
        ea_damper = 0.0
        chw_t_in = chilled_water_temp
        chw_t_out = chilled_water_temp
        chw_mdot = 0.0
        chw_flow_m3_min = 0.0
        chw_pump_power_kw = 0.0
        chw_pump_dp_pa = 0.0
        chw_pump_inv = 0.0
        chiller_power_kw = 0.0

        if hvac_on and total_flow > 1e-6:
            # --- 給外気量と混合空気状態を決定し、コイル負荷計算の準備 ---
            return_temp = float(np.sum(zone_flows * zone_temps) / total_flow)
            if actions.oa_damper is not None:
                oa_damper = float(np.clip(actions.oa_damper, 0.0, 1.0))
                oa_flow = float(np.clip(oa_damper * total_flow, oa_flow_min, total_flow))
            else:
                oa_flow = float(np.clip(max(oa_frac_min * total_flow, oa_flow_min), 0.0, total_flow))
                oa_damper = oa_flow / total_flow
            recirc_flow = max(total_flow - oa_flow, 0.0)
            if total_flow > 0:
                ra_damper = recirc_flow / total_flow
                ea_damper = oa_flow / total_flow
            # 外気と還気の質量流量比から混合空気のエンタルピーと湿度を算出
            outdoor_abs_hum = pv.tdb_rh2w(outdoor_temp, outdoor_rh)  # 実際の外気相対湿度を使用
            outdoor_enthalpy = pv.tdb_w2h(outdoor_temp, outdoor_abs_hum)
            return_enthalpy = pv.tdb_w2h(return_temp, return_abs_hum)
            mixed_abs_hum = (oa_flow * outdoor_abs_hum + recirc_flow * return_abs_hum) / total_flow
            mixed_enthalpy = (oa_flow * outdoor_enthalpy + recirc_flow * return_enthalpy) / total_flow
            denom = pv.CA + pv.CV * mixed_abs_hum
            if denom > 1e-9:
                mixed_temp = (mixed_enthalpy - pv.R0 * mixed_abs_hum) / denom
            else:
                mixed_temp = (oa_flow * outdoor_temp + recirc_flow * return_temp) / total_flow
            if total_flow > 1e-6:
                mixed_co2_ppm = (
                    oa_flow * OUTDOOR_CO2_PPM + recirc_flow * return_co2_ppm
                ) / total_flow
                supply_co2_ppm = mixed_co2_ppm
            else:
                mixed_co2_ppm = return_co2_ppm
                supply_co2_ppm = return_co2_ppm

            cp_mdot = total_flow * CP_AIR
            # --- コイルバルブの開度をPIDまたは外部指令から決定 ---
            coil_valve_cmd = 0.0
            if actions.coil_valve is not None:
                coil_valve_cmd = float(np.clip(actions.coil_valve, 0.0, 1.0))
            else:
                supply_feedback = last_supply_temp
                if not np.isfinite(supply_feedback):
                    supply_feedback = mixed_temp
                error = supply_feedback - supply_air_setpoint
                if coil_pid_ti_s > 1e-6:
                    coil_integral += (timestep_s / coil_pid_ti_s) * error
                coil_integral = float(np.clip(coil_integral, -0.5, 0.5))
                controller_output = coil_pid_kp * error + coil_integral
                # Deadband keeps the valve from chattering around the minimum position
                valve_lower_bound = coil_valve_min if error > -coil_valve_hold_deadband else 0.0
                coil_valve_cmd = float(np.clip(controller_output, valve_lower_bound, 1.0))

            if coil_valve_filter_alpha > 0.0:
                coil_valve_state += coil_valve_filter_alpha * (coil_valve_cmd - coil_valve_state)
                coil_valve = float(np.clip(coil_valve_state, 0.0, 1.0))
            else:
                coil_valve_state = float(np.clip(coil_valve_cmd, 0.0, 1.0))
                coil_valve = coil_valve_state

            chw_valve.vlv = coil_valve

            # --- 冷水ポンプと弁の動作を更新し、必要な冷水流量を算出 ---
            chw_pump_power_kw = 0.0
            chw_pump_dp_pa = 0.0
            chw_flow_m3_min = 0.0
            chw_mdot = 0.0
            chw_pump_inv = CHW_PUMP_INV_MIN

            if coil_valve > 1e-4:
                pump_fraction = float(np.clip(coil_valve, 0.0, 1.0) ** 0.5)
                chw_pump_inv = float(
                    np.clip(
                        CHW_PUMP_INV_MIN
                        + (CHW_PUMP_INV_MAX - CHW_PUMP_INV_MIN) * pump_fraction,
                        CHW_PUMP_INV_MIN,
                        CHW_PUMP_INV_MAX,
                    )
                )
                chw_pump.inv = chw_pump_inv
                chw_flow_m3_min = _solve_branch_flow(
                    chw_branch,
                    g_cap=CHW_BRANCH_FLOW_MAX,
                    tol=CHW_FLOW_TOL_KPA,
                )
                chw_branch.f2p(chw_flow_m3_min)
                chw_pump.cal()
                chw_mdot = (chw_flow_m3_min / 60.0) * RHO_WATER
                chw_pump_dp_kpa = float(chw_pump.dp)
                chw_pump_dp_pa = chw_pump_dp_kpa * 1000.0
                hydraulic_kw = (chw_flow_m3_min / 60.0) * chw_pump_dp_kpa
                if pump_efficiency > 1e-9:
                    chw_pump_power_kw = float(hydraulic_kw / pump_efficiency)
                else:
                    chw_pump_power_kw = 0.0
            else:
                chw_pump.inv = CHW_PUMP_INV_MIN
                chw_branch.f2p(0.0)
                chw_pump.cal()

            chw_pump.pw = chw_pump_power_kw
            chw_pump_inv = float(chw_pump.inv)

            supply_temp = mixed_temp
            coil_load_sensible_w = 0.0
            coil_load_latent_w = 0.0
            total_coil_load_w = 0.0
            chw_t_out = chw_t_in

            if (
                coil_valve > 1e-4
                and chw_mdot > 1e-6
                and coil_base_area > 0.0
                and total_flow > 1e-6
            ):
                # phyvacのコイルモデルで顕熱・潜熱負荷を同時に評価し、送水温度も更新
                coil_air_mdot = max(total_flow, 0.0)
                tout_air, wout_air, rhout_air, tout_water, _, _, _, _ = coil_hex.cal(
                    mixed_temp,
                    mixed_abs_hum,  # 実際の混合空気の絶対湿度を使用
                    chilled_water_temp,
                    coil_air_mdot,
                    chw_mdot,
                )
                supply_candidate = float(tout_air)
                water_out_candidate = float(tout_water)
                min_ref_temp = chilled_water_temp
                if np.isfinite(water_out_candidate):
                    min_ref_temp = min(min_ref_temp, water_out_candidate)
                min_supply = min_ref_temp + coil_approach
                coil_supply_temp = float(np.clip(supply_candidate, min_supply, mixed_temp))
                coil_supply_abs_hum = float(wout_air)  # コイル出口の絶対湿度
                # 数値誤差やモデルの近似で飽和湿度を超えることがあるため、
                # 乾球温度に対応する飽和絶対湿度で上限をかける。
                sat_abs_hum_coil = pv.tdb_rh2w(coil_supply_temp, 100.0)
                if np.isfinite(sat_abs_hum_coil):
                    coil_supply_abs_hum = float(
                        np.clip(coil_supply_abs_hum, 0.0, sat_abs_hum_coil)
                    )

                # バイパスファクターでコイルを通らない空気を混合（混合空気10%既定）
                supply_temp = float(
                    bypass_factor * mixed_temp + (1.0 - bypass_factor) * coil_supply_temp
                )
                supply_abs_hum = float(
                    bypass_factor * mixed_abs_hum
                    + (1.0 - bypass_factor) * coil_supply_abs_hum
                )
                sat_abs_hum = pv.tdb_rh2w(supply_temp, 100.0)
                if np.isfinite(sat_abs_hum):
                    supply_abs_hum = float(np.clip(supply_abs_hum, 0.0, sat_abs_hum))

                enthalpy_drop_kj_per_kg = max(
                    mixed_enthalpy - pv.tdb_w2h(supply_temp, supply_abs_hum), 0.0
                )
                total_kw = max(total_flow * enthalpy_drop_kj_per_kg, 0.0)
                sensible_kw = max(
                    cp_mdot * max(mixed_temp - supply_temp, 0.0) / 1000.0,
                    0.0,
                )
                sensible_kw = min(total_kw, sensible_kw)
                latent_kw = max(total_kw - sensible_kw, 0.0)

                coil_load_sensible_w = sensible_kw * 1000.0
                coil_load_latent_w = latent_kw * 1000.0
                total_coil_load_w = total_kw * 1000.0
                if total_coil_load_w > 0.0:
                    chw_t_out = chw_t_in + total_coil_load_w / (max(chw_mdot, 1e-9) * CP_WATER)
                else:
                    chw_t_out = chw_t_in
        else:
            # 非稼働時はファン停止・弁閉・積分器リセットでエネルギー消費をゼロに近づける
            supply_static_pa = 0.0
            zone_dp_pa = np.zeros(len(zones))
            fan.g = 0.0
            fan.cal()
            fan_dp_pa = float(fan.dp)
            recirc_flow = 0.0
            coil_integral = 0.0
            last_supply_temp = float("nan")
            coil_dp_active = 0.0
            chw_valve.vlv = 0.0
            chw_pump.inv = 0.0
            chw_branch.f2p(0.0)
            chw_pump.cal()
            chw_flow_m3_min = 0.0
            chw_mdot = 0.0
            chw_pump_power_kw = 0.0
            chw_pump_dp_pa = 0.0
            chw_pump_inv = float(chw_pump.inv)
            chw_t_out = chw_t_in

        if not hvac_on or chw_mdot <= 1e-6 or total_coil_load_w <= 0.0:
            chw_t_out = chw_t_in

        # 全負荷（顕熱+潜熱）でチラー電力を計算
        total_coil_load_w = coil_load_sensible_w + coil_load_latent_w
        if total_coil_load_w > 0 and hvac_on:
            # 指定COPで冷凍機電力を逆算（定常COP仮定）
            chiller_power_kw = (total_coil_load_w / max(chiller_cop, 1e-3)) / 1000.0
        else:
            chiller_power_kw = 0.0

        fan_power_kw = float(fan.pw)
        fan_dp_pa = float(fan.dp)

        next_zone_temps = zone_temps.copy()
        next_zone_co2_ppm = zone_co2_ppm.copy()
        next_zone_abs_hum = zone_abs_hum.copy()
        next_zone_rh = zone_rh.copy()
        effective_supply_temp = supply_temp if hvac_on and not np.isnan(supply_temp) else outdoor_temp
        effective_supply_co2 = supply_co2_ppm if hvac_on else OUTDOOR_CO2_PPM
        if hvac_on:
            if np.isnan(supply_temp):
                effective_supply_temp = mixed_temp
                effective_supply_abs_hum = mixed_abs_hum
            else:
                effective_supply_temp = supply_temp
                effective_supply_abs_hum = supply_abs_hum
        else:
            effective_supply_temp = outdoor_temp
            effective_supply_abs_hum = pv.tdb_rh2w(outdoor_temp, outdoor_rh)

        # 系統空気（還気・混合・給気）の相対湿度を算出し、レポート出力に備える
        return_rh = float(np.clip(pv.w_tdb2rh(return_abs_hum, return_temp), 0.0, 100.0))
        mixed_rh = float(np.clip(pv.w_tdb2rh(mixed_abs_hum, mixed_temp), 0.0, 100.0))
        if hvac_on and not np.isnan(supply_temp):
            supply_rh_raw = pv.w_tdb2rh(supply_abs_hum, supply_temp)
        else:
            supply_rh_raw = pv.w_tdb2rh(effective_supply_abs_hum, effective_supply_temp)
        supply_rh = float(np.clip(supply_rh_raw, 0.0, 100.0))

        if hvac_on:
            if np.isnan(supply_temp):
                last_supply_temp = float(mixed_temp)
            else:
                last_supply_temp = float(supply_temp)

        for idx, zone in enumerate(zones):
            mass_flow = zone_flows[idx]
            infil_flow = zone.infil_mdot
            # 外皮・すきま風・供給空気・内部発熱を足し合わせて顕熱バランスを解く
            q_env = zone.ua_env * (outdoor_temp - zone_temps[idx])
            q_inf = infil_flow * CP_AIR * (outdoor_temp - zone_temps[idx])
            q_supply = mass_flow * CP_AIR * (effective_supply_temp - zone_temps[idx])
            q_total = q_env + q_inf + internal_gains[idx] + q_supply
            delta_t = (q_total * timestep_s) / zone.thermal_cap
            next_zone_temps[idx] = zone_temps[idx] + delta_t

            zone_volume = max(zone.volume_m3, 1e-6)
            supply_vol_flow = mass_flow / RHO_AIR
            infil_vol_flow = infil_flow / RHO_AIR
            co2_exchange = 0.0
            if supply_vol_flow > 0.0:
                co2_exchange += supply_vol_flow * (effective_supply_co2 - zone_co2_ppm[idx])
            if infil_vol_flow > 0.0:
                co2_exchange += infil_vol_flow * (OUTDOOR_CO2_PPM - zone_co2_ppm[idx])

            co2_gen_lps = max(zone.co2_gen_lps_per_person, 0.0) * max(occupant_counts[idx], 0.0)
            co2_gen_m3_s = co2_gen_lps / 1000.0
            delta_co2 = (co2_exchange / zone_volume) * timestep_s + (co2_gen_m3_s / zone_volume) * timestep_s * 1e6
            next_zone_co2_ppm[idx] = zone_co2_ppm[idx] + delta_co2
            next_zone_co2_ppm[idx] = float(np.clip(next_zone_co2_ppm[idx], 350.0, 5000.0))

            # 湿度の計算
            # 湿度交換（給気と侵入外気による）
            hum_exchange = 0.0
            if supply_vol_flow > 0.0:
                hum_exchange += supply_vol_flow * RHO_AIR * (effective_supply_abs_hum - zone_abs_hum[idx])
            if infil_vol_flow > 0.0:
                outdoor_abs_hum_infil = pv.tdb_rh2w(outdoor_temp, outdoor_rh)  # 実際の外気相対湿度を使用
                hum_exchange += infil_vol_flow * RHO_AIR * (outdoor_abs_hum_infil - zone_abs_hum[idx])

            # 人体からの潜熱負荷（水蒸気発生量）
            latent_gen_kg_s = max(zone.latent_gain_per_person, 0.0) * max(occupant_counts[idx], 0.0) / (pv.R0)  # kW -> kg/s

            # 絶対湿度の変化
            zone_air_mass = zone_volume * RHO_AIR  # ゾーン内の空気質量[kg]
            delta_abs_hum = (hum_exchange + latent_gen_kg_s) * timestep_s / zone_air_mass
            next_zone_abs_hum[idx] = zone_abs_hum[idx] + delta_abs_hum
            next_zone_abs_hum[idx] = float(np.clip(next_zone_abs_hum[idx], 0.0, 0.030))  # 絶対湿度の上限を設定

            # 相対湿度の更新
            next_zone_rh[idx] = float(pv.w_tdb2rh(next_zone_abs_hum[idx], next_zone_temps[idx]))
            next_zone_rh[idx] = float(np.clip(next_zone_rh[idx], 0.0, 100.0))

        zone_temps = next_zone_temps
        zone_co2_ppm = next_zone_co2_ppm
        zone_abs_hum = next_zone_abs_hum
        zone_rh = next_zone_rh

        # このステップの主要状態量をデータフレームに蓄積
        # --- 現在ステップの主要指標を辞書化し、結果データに追加 ---
        record = {
            "time": current_time,
            "hvac_on": hvac_on,
            "outdoor_temp": outdoor_temp,
            "outdoor_rh": outdoor_rh,
            "return_temp": return_temp,
            "mixed_temp": mixed_temp,
            "supply_temp": supply_temp,
            "return_rh": return_rh,
            "mixed_rh": mixed_rh,
            "supply_rh": supply_rh,
            "outdoor_co2_ppm": OUTDOOR_CO2_PPM,
            "return_co2_ppm": return_co2_ppm,
            "mixed_co2_ppm": mixed_co2_ppm,
            "supply_co2_ppm": supply_co2_ppm,
            "total_flow_kg_s": total_flow,
            "oa_flow_kg_s": oa_flow,
            "recirc_flow_kg_s": recirc_flow,
            "chw_flow_kg_s": chw_mdot,
            "chw_flow_m3_min": chw_flow_m3_min,
            "coil_load_kw": total_coil_load_w / 1000.0,
            "coil_load_latent_kw": coil_load_latent_w / 1000.0,
            "coil_load_sensible_kw": coil_load_sensible_w / 1000.0,
            "chiller_power_kw": chiller_power_kw,
            "coil_valve_pos": coil_valve,
            "chw_t_in": chw_t_in,
            "chw_t_out": chw_t_out,
            "fan_power_kw": fan_power_kw,
            "chw_pump_power_kw": chw_pump_power_kw,
            "chw_pump_dp_pa": chw_pump_dp_pa,
            "chw_pump_inv": chw_pump_inv,
            "fan_dp_pa": fan_dp_pa,
            "supply_static_pa": supply_static_pa,
            "static_pressure_sp_pa": static_pressure_sp,
            "fan_flow_m3_min": fan_flow_m3_min,
            "coil_dp_pa": coil_dp_active,
            "supply_trunk_dp_pa": supply_trunk_dp_pa,
            "return_trunk_dp_pa": return_trunk_dp_pa,
            "fan_inv": float(fan.inv),
        }

        for idx, zone in enumerate(zones):
            record[f"zone{idx + 1}_temp"] = zone_temps[idx]
            record[f"zone{idx + 1}_flow_kg_s"] = zone_flows[idx]
            record[f"zone{idx + 1}_damper"] = damper_positions[idx]
            record[f"zone{idx + 1}_dp_pa"] = zone_dp_pa[idx]
            record[f"zone{idx + 1}_co2_ppm"] = zone_co2_ppm[idx]
            record[f"zone{idx + 1}_rh"] = zone_rh[idx]
            record[f"zone{idx + 1}_occupancy"] = occupant_counts[idx]

        record["oa_damper"] = oa_damper
        record["ra_damper"] = ra_damper
        record["ea_damper"] = ea_damper

        # 1タイムステップ分の状態量を辞書化し、ロギングリストへ追加
        records.append(record)
        current_time += timedelta(seconds=timestep_s)
        prev_hvac_on = hvac_on

    # --- 蓄積した記録をDataFrame化し、時間をインデックスに設定して返却 ---
    df = pd.DataFrame.from_records(records).set_index("time")
    return df


def compute_zone_pid_metrics(df: pd.DataFrame, *, setpoint: float) -> Dict[str, float]:
    """ゾーンPID調整に用いる快適性・アクチュエータ指標を算出する。"""

    # --- 空調稼働区間に絞った評価データを抽出 ---
    hvac_df = df[df["hvac_on"]] if "hvac_on" in df.columns else df
    if hvac_df.empty:
        hvac_df = df

    # 稼働時間帯に限定して、ゾーン温度とダンパー挙動から評価指標を計算

    zone_temp_cols = sorted(
        col for col in hvac_df.columns if col.startswith("zone") and col.endswith("_temp")
    )
    damper_cols = sorted(
        col for col in hvac_df.columns if col.startswith("zone") and col.endswith("_damper")
    )

    # --- 温度偏差とアクチュエータ挙動をまとめた指標容器を初期化 ---
    metrics: Dict[str, float] = {
        "max_overshoot": 0.0,
        "mean_abs_error": 0.0,
        "comfort_violation": 0.0,
        "damper_chatter": 0.0,
        "damper_std": 0.0,
    }

    if zone_temp_cols:
        temps = hvac_df[zone_temp_cols].to_numpy()
        temp_errors = temps - setpoint
        positive_errors = np.clip(temp_errors, 0.0, None)

        if positive_errors.size:
            # 設定温度を超えた分を重視して快適性指標を算出
            metrics["max_overshoot"] = float(np.max(positive_errors))
            metrics["comfort_violation"] = float(
                np.mean(np.clip(positive_errors - 0.2, 0.0, None))
            )

        if temp_errors.size:
            metrics["mean_abs_error"] = float(np.mean(np.abs(temp_errors)))

    if damper_cols:
        damper_values = hvac_df[damper_cols].to_numpy()
        if damper_values.size:
            # ダンパーの変化量・分散を指標化し、チャタリング抑制効果を確認
            # 連続サンプル間の開度差分を求め、チャタリング指標として平均絶対値を算出
            damper_diff = np.diff(damper_values, axis=0)
            if damper_diff.size:
                metrics["damper_chatter"] = float(np.mean(np.abs(damper_diff)))
            metrics["damper_std"] = float(np.mean(np.std(damper_values, axis=0)))

    return metrics


def zone_pid_cost(metrics: Dict[str, float]) -> float:
    """快適性を重視した重み付けスカラーコストを評価する。"""

    # --- 重み付けで単一スカラーに集約し、制御性能を比較しやすくする ---
    # 係数は快適性重視で調整済み。値が小さいほど制御性能が良い。
    return (
        120.0 * metrics["max_overshoot"]
        + 24.0 * metrics["comfort_violation"]
        + 14.0 * metrics["mean_abs_error"]
        + 7.5 * metrics["damper_chatter"]
        + 3.0 * metrics["damper_std"]
    )


def create_plots(df: pd.DataFrame, output_path: Path, damper_path: Path) -> None:
    """主要な温度・流量・電力指標を図化しPNGで保存するユーティリティ。"""
    # 保存先ディレクトリを確保
    output_path.parent.mkdir(parents=True, exist_ok=True)
    damper_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- 温度・風量・電力・ダンパー開度を見やすくチャート化して保存 ---
    fig, axes = plt.subplots(8, 1, figsize=(11, 32), sharex=True)

    zone_temp_cols = [col for col in df.columns if col.endswith("_temp") and col.startswith("zone")]
    df[zone_temp_cols].plot(ax=axes[0])
    axes[0].axhline(26.0, color="k", linestyle="--", linewidth=1, label="Setpoint 26°C")
    axes[0].set_ylabel("Zone Temperature [°C]")
    axes[0].set_title("Zone Air Temperatures")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.8)

    flow_cols = [col for col in df.columns if col.endswith("_flow_kg_s") and col.startswith("zone")]
    chw_flow_col = [col for col in df.columns if col == "chw_flow_kg_s"]

    # ゾーンのフローは左軸、CHWフローは右軸
    (df[flow_cols] * 3600).plot(ax=axes[1], legend=True)
    axes[1].set_ylabel("Supply Flow [kg/h]")
    axes[1].set_title("Zone Supply Mass Flow")
    axes[1].grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.8)

    if chw_flow_col:
        ax2 = axes[1].twinx()
        # 冷水流量は別軸でプロットして空気系との関係を確認
        (df[chw_flow_col] * 3600).plot(ax=ax2, color="k", linestyle="--", legend=True)
        ax2.set_ylabel("CHW Flow [kg/h]")
        ax2.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.8)
        # 凡例を両方表示
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    power_cols = ["chiller_power_kw", "fan_power_kw"]
    if "chw_pump_power_kw" in df.columns:
        power_cols.append("chw_pump_power_kw")

    legend_labels = []
    for col in power_cols:
        if col == "chiller_power_kw":
            legend_labels.append("Chiller Power (COP=4.0 const)")
        elif col == "fan_power_kw":
            legend_labels.append("Fan Power")
        elif col == "chw_pump_power_kw":
            legend_labels.append("CHW Pump Power")
        else:
            legend_labels.append(col)
    df[power_cols].plot(ax=axes[2])
    axes[2].set_ylabel("Power [kW]")
    axes[2].set_title("Chiller, Fan, and Pump Power")
    axes[2].grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.8)
    axes[2].legend(legend_labels, loc="upper right")

    # 空気温度のプロット
    df[["return_temp", "outdoor_temp", "mixed_temp", "supply_temp"]].plot(ax=axes[3])
    axes[3].set_ylabel("Air Temp [°C]")
    axes[3].set_ylim(5.0, 40.0)
    axes[3].set_title("Air Temperatures")
    axes[3].grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.8)
    axes[3].legend(loc="upper left")

    df[["chw_t_in", "chw_t_out"]].plot(ax=axes[4])
    axes[4].set_ylabel("Coil Water Temp [°C]")
    axes[4].set_title("Coil Inlet/Outlet Temperatures")
    axes[4].grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.8)
    axes[4].legend(["Coil Inlet Temp", "Coil Outlet Temp"], loc="upper right")

    co2_cols = [col for col in df.columns if col.endswith("_co2_ppm") and col.startswith("zone")]
    if co2_cols:
        df[co2_cols].plot(ax=axes[5])
        axes[5].axhline(1000.0, color="r", linestyle="--", linewidth=1, label="Target 1000 ppm")
        axes[5].set_ylabel("Zone CO₂ [ppm]")
        axes[5].set_title("Zone CO₂ Concentration")
        axes[5].legend(loc="upper right")
        axes[5].grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.8)

    # 相対湿度のプロット（ゾーン）
    rh_cols = [col for col in df.columns if col.endswith("_rh") and col.startswith("zone")]
    if rh_cols:
        df[rh_cols].plot(ax=axes[6])
        axes[6].axhline(50.0, color="g", linestyle="--", linewidth=1, label="Target 50% RH")
        axes[6].axhline(60.0, color="orange", linestyle="--", linewidth=0.5, label="Upper Comfort 60% RH")
        axes[6].axhline(40.0, color="orange", linestyle="--", linewidth=0.5, label="Lower Comfort 40% RH")
        axes[6].set_ylabel("Zone RH [%]")
        axes[6].set_title("Zone Relative Humidity")
        axes[6].set_ylim(0.0, 100.0)
        axes[6].legend(loc="upper right")
        axes[6].grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.8)
    else:
        axes[6].set_visible(False)

    # 系統空気（RA/OA/MA/SA）の相対湿度を温度と同様に可視化
    system_rh_cols = [col for col in ["return_rh", "outdoor_rh", "mixed_rh", "supply_rh"] if col in df.columns]
    if system_rh_cols:
        df[system_rh_cols].plot(ax=axes[7])
        axes[7].set_ylabel("Air RH [%]")
        axes[7].set_ylim(0.0, 100.0)
        axes[7].set_title("Return / Outdoor / Mixed / Supply Air RH")
        axes[7].legend([
            label.replace("_", " ").title().replace("Rh", "RH")
            for label in system_rh_cols
        ], loc="upper left")
        axes[7].grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.8)
    else:
        axes[7].set_visible(False)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    damper_cols = sorted(
        col for col in df.columns if col.endswith("_damper") and col.startswith("zone")
    )
    control_cols = [
        *damper_cols,
        "oa_damper",
        "coil_valve_pos",
    ]
    control_cols = [col for col in control_cols if col in df.columns]
    fan_cols = [col for col in ["fan_inv"] if col in df.columns]

    damper_fig, damper_axes = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True, height_ratios=[2.0, 1.0]
    )

    if control_cols:
        df[control_cols].plot(ax=damper_axes[0])
        damper_axes[0].set_ylim(0.0, 1.05)
        damper_axes[0].set_ylabel("Damper/Valve Position [-]")
        damper_axes[0].set_title("Damper and Valve Openings (Zones, OA, Coil Valve)")
        damper_axes[0].grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.8)
    else:
        damper_axes[0].set_visible(False)

    if fan_cols:
        df[fan_cols].rename(columns={"fan_inv": "Fan Frequency [p.u.]"}).plot(
            ax=damper_axes[1], color="tab:blue", legend=True
        )
        damper_axes[1].set_ylabel("Fan Frequency Command [p.u.]")
        damper_axes[1].set_title("Supply Fan Frequency")
        damper_axes[1].set_ylim(0.0, 1.5)
        damper_axes[1].set_yticks(np.arange(0.0, 1.4 + 1e-6, 0.2))
        damper_axes[1].grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.8)
        damper_axes[1].set_xlabel("Time")
    else:
        damper_axes[1].set_visible(False)

    if not fan_cols and damper_axes[0].get_visible():
        damper_axes[0].set_xlabel("Time")

    damper_fig.tight_layout()
    damper_fig.savefig(damper_path, dpi=180)
    plt.close(damper_fig)


def main() -> None:
    """サンプル構成で24時間シミュレーションを実行し、CSVとグラフを出力する。"""
    # 2025年7月29日の24時間を1分刻みでシミュレーションするシナリオ定義
    start = datetime(2025, 7, 29, 0, 0)
    # --- シミュレーション対象の4ゾーンについて熱容量や内部負荷を定義 ---
    zones = list(build_default_zones())

    minutes = 24 * 60
    timestep_s = 60
    setpoint = 26.0

    # --- 共通シミュレーション設定を辞書にまとめ、試行条件を一括管理 ---
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
        setpoint=setpoint,
        zone_pid_t_reset=30,
        zone_pid_t_step=1,
        zone_pid_max_step=0.10,
    )
    # 上記辞書に共通パラメータをまとめ、複数パターンを試しやすくする

    # --- 今回検証するゾーンPIDゲインを設定 ---
    selected_kp = 1.1667
    selected_ti = 100.0

    print(
        f"Running simulation with fixed zone PID -> kp={selected_kp:.3f}, ti={selected_ti:.1f}"
    )
    # 実際の制御パラメータでシミュレーションを1回実行

    df = run_simulation(
        zone_pid_kp=selected_kp,
        zone_pid_ti=selected_ti,
        **sim_kwargs,
    )

    # --- シミュレーション結果からゾーンPID指標と運転コストを算出 ---
    metrics = compute_zone_pid_metrics(df, setpoint=setpoint)
    cost = zone_pid_cost(metrics)
    print(
        f"Zone PID metrics -> cost={cost:.3f}, "
        f"max_overshoot={metrics['max_overshoot']:.3f}°C, "
        f"mean_abs_error={metrics['mean_abs_error']:.3f}°C, chatter={metrics['damper_chatter']:.3f}"
    )
    # 結果のメトリクスとCSV/グラフを出力し、分析に活用できる状態にする

    # --- 主要な結果をCSV/テキスト/グラフとして保存し、可視化を支援 ---
    csv_path = SIMULATION_OUTPUT_DIR / "simulation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"結果をCSVに保存: {csv_path}")

    # --- 快適性・省エネ・空気質を指標化し、ベースライン比較に備える ---
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
        temp_errors = temps - setpoint
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
        # 各時間ステップの電力（kW）× 時間（1/60時間）の合計で総電力消費を計算
        total_power_kwh = float(power_data.sum()) * (1.0 / 60.0)
    
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

    # 相対湿度の計算
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

    # テキストファイルでの結果保存
    txt_path = SIMULATION_OUTPUT_DIR / "simulation_results.txt"
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

    plot_path = FIGURES_OUTPUT_DIR / "simulation_results.png"
    damper_plot_path = FIGURES_OUTPUT_DIR / "damper_positions.png"
    # 温調の挙動を視覚的に確認できるようPNGとダンパープロットを保存
    create_plots(df, plot_path, damper_plot_path)
    print(f"プロットを保存: {plot_path}, {damper_plot_path}")


if __name__ == "__main__":
    # --- 単体実行時は上記設定で1日分のシミュレーションを回す ---
    main()
