"""Duct geometry definitions for the humid simulator."""
# ---------------------------------------------------------------------------
# このモジュールでは、空調ダクト網の幾何情報（直管長さ・管径・摩擦係数・局部損失係数）
# を一か所に整理し、シミュレーション本体から参照できるようにする。
#
# 目的:
#   * ゾーン別分岐ダクトの固定抵抗を物理量から計算する（直管摩擦 + 局部損失）
#   * 幹ダクト・リターンダクトの損失を別途定義し、ファン静圧の釣り合いに反映する
#   * 配置変更や口径変更があった際には、このファイルを修正するだけで各種解析が追随
#
# 入力方法:
#   1. `DuctSegment` で 1 区間を記述（長さ[m], 内径[m], Darcy摩擦係数, 局部損失係数群）
#   2. 複数区間を `BranchGeometry` にまとめて 1 本の枝や幹を表現
#   3. ゾーン名をキーに `ZONE_BRANCH_GEOMETRY` に登録。ゾーン名がシナリオで異なる場合は
#      `ZoneConfig(..., duct_key="Zone 1")` のように明示する
#
# 計算方法:
#   * `DuctSegment.resistance_coefficient()` で JP 単位 [Pa/(m3/min)^2] の係数を返す。
#     これは dp = r * g^2 にそのまま代入可能。
#   * モデル内では一定密度 (RHO_AIR=1.2 kg/m3) を仮定し、風量単位を m3/min で統一。
#
# 編集時のTips:
#   * 摩擦係数 f は 0.018〜0.030 程度（スパイラルダクト想定）。粗度違いは実測値に応じて変更。
#   * 局部損失係数 K は継手・分岐・VAV箱・吹出口などのカタログ値を足し込み、Tuple で列挙。
#   * 1ゾーンに対して複数の枝がある場合は、代表となる支線を抽象化して記述し、必要に応じて
#     `BranchGeometry` を複製／調整する。
# ---------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple
from typing import Literal


@dataclass(frozen=True)
class DuctSegment:
    """Represents a straight or lumped duct section with optional local losses."""
    # length_m: 直管長さ [m]。局部損失しか無い区間でも0.1などの仮想長にしておくと便利。
    # diameter_m: 管の内径 [m]。矩形ダクトの場合は等価直径に換算。
    # friction_factor: Darcy-Weisbach摩擦係数 (無次元)。乱流域の代表値を想定。
    # local_ks: エルボ・分岐・ダンパ・吹出口などの局部損失係数 K を列挙する。
    # label: 後で読みやすくするための自由記述ラベル。

    length_m: float
    diameter_m: float
    friction_factor: float = 0.02  # Darcy friction factor
    local_ks: Tuple[float, ...] = ()
    label: str | None = None

    kind: Literal["duct", "terminal"] = "duct"

    def area_m2(self) -> float:
        # 円形ダクトを想定した断面積。矩形の等価直径で置き換えても良い。
        radius = max(self.diameter_m, 1e-6) * 0.5
        return math.pi * radius * radius

    def k_total(self) -> float:
        major_loss = self.friction_factor * (self.length_m / max(self.diameter_m, 1e-6))
        minor_loss = float(sum(self.local_ks))
        return max(major_loss + minor_loss, 0.0)

    def resistance_coefficient(self, rho_air: float = 1.2) -> float:
        """Return the dp coefficient [Pa/(m3/min)^2] for this segment."""
        # dp = 0.5 * rho * V^2 * (f*L/D + ΣK)
        #      = (0.5*rho / A^2) * g^2 * (f*L/D + ΣK)
        # g を m3/min で扱うため、3600^2 で単位変換している。

        area = self.area_m2()
        k_total = self.k_total()
        if area <= 1e-9 or k_total <= 0.0:
            return 0.0
        base = 0.5 * rho_air * k_total
        return base / (area ** 2 * 3600.0)


@dataclass(frozen=True)
class BranchGeometry:
    """Collection of duct segments that form a branch or trunk."""
    # 複数区間を直列に並べた合成抵抗を表す。
    # ここにゾーン枝をまとめて登録し、`resistance_coefficient()` で合算する。

    segments: Tuple[DuctSegment, ...]
    note: str | None = None

    def resistance_coefficient(self, rho_air: float = 1.2) -> float:
        return float(sum(seg.resistance_coefficient(rho_air) for seg in self.segments))

    def resistance_components(self, rho_air: float = 1.2) -> Tuple[float, float]:
        """Return (kr_duct, kr_eq) to distinguish duct and terminal losses."""

        kr_duct = 0.0
        kr_eq = 0.0
        for seg in self.segments:
            coeff = seg.resistance_coefficient(rho_air)
            if coeff <= 0.0:
                continue
            if seg.kind == "terminal":
                kr_eq += coeff
            else:
                kr_duct += coeff
        return float(kr_duct), float(kr_eq)


def _default_branch_geometry() -> BranchGeometry:
    # ゾーン名が登録されていない場合に使用されるフォールバック。
    # 代表的な幹→VAV→吹出口のセットを仮置きしておき、警告を表示して利用者に知らせる。
    return BranchGeometry(
        segments=(
            DuctSegment(6.0, 0.28, 0.022, (0.5, 0.5, 1.0), label="Default upstream"),
            DuctSegment(2.4, 0.22, 0.025, (1.5, 1.2), label="Default VAV/diffuser", kind="terminal"),
        ),
        note="Fallback geometry when a zone-specific definition is missing.",
    )


ZONE_BRANCH_GEOMETRY: Dict[str, BranchGeometry] = {
    # --- ここからゾーン別の分岐設定 ---
    # 例として4ゾーン分を用意。実際のプロジェクトに合わせて長さや径を更新する。
    "Zone 1": BranchGeometry(
        segments=(
            # (1) コイル直後からVAV手前までの太径区間（主に直管＋エルボ）
            DuctSegment(
                7.5,   # 長さ[m] (直管中央部)
                0.32,  # 内径[m]
                0.022, # Darcy摩擦係数f
                (0.6, 0.6),  # 局部損失Kのリスト（エルボ等）
                label="Z1 upstream run",
            ),
            # (2) 枝へ絞るテーパー区間。断面縮小とエルボ損失をまとめて表現
            DuctSegment(3.2,0.26,0.024,(1.1, 0.9),label="Z1 reducer"),
            # (3) VAV箱＋吹出口付近。末端局部損失を大きめに設定
            DuctSegment(0.8,0.20,0.028,(2.2, 1.5),label="Z1 VAV+diffuser", kind="terminal"),
        ),
        note="Branch serving open office Zone 1.",
    ),
    "Zone 2": BranchGeometry(
        segments=(
            DuctSegment(5.8, 0.30, 0.022, (0.5, 0.4), label="Z2 upstream run"),
            DuctSegment(2.8, 0.24, 0.024, (1.0, 1.0), label="Z2 reducer"),
            DuctSegment(0.7, 0.19, 0.028, (2.4, 1.6), label="Z2 VAV+diffuser", kind="terminal"),
        ),
        note="Branch serving conference/Zone 2.",
    ),
    "Zone 3": BranchGeometry(
        segments=(
            DuctSegment(5.2, 0.28, 0.022, (0.4, 0.4), label="Z3 upstream run"),
            DuctSegment(2.4, 0.22, 0.025, (1.1, 0.8), label="Z3 reducer"),
            DuctSegment(0.7, 0.18, 0.030, (2.6, 1.4), label="Z3 VAV+diffuser", kind="terminal"),
        ),
        note="Branch serving smaller office Zone 3.",
    ),
    "Zone 4": BranchGeometry(
        segments=(
            DuctSegment(6.8, 0.31, 0.022, (0.6, 0.5, 0.4), label="Z4 upstream run"),
            DuctSegment(3.0, 0.25, 0.024, (1.0, 1.1), label="Z4 reducer"),
            DuctSegment(0.8, 0.20, 0.028, (2.3, 1.7), label="Z4 VAV+diffuser", kind="terminal"),
        ),
        note="Branch serving Zone 4 perimeter offices.",
    ),
}

DEFAULT_BRANCH_GEOMETRY = _default_branch_geometry()


SUPPLY_TRUNK_GEOMETRY = BranchGeometry(
    segments=(
        DuctSegment(12.0, 0.55, 0.021, (1.2,), label="Supply main 1"),
        DuctSegment(8.0, 0.50, 0.022, (1.0, 0.8), label="Supply main 2"),
        DuctSegment(4.0, 0.45, 0.024, (1.0,), label="Supply header"),
    ),
    note="Main supply trunk from coil plenum to VAV takeoffs.",
)

# ゾーンごとの分岐位置（after_index）を上流→下流の順で定義。
# 値0はファン直後（区間0の手前）、値nは区間n-1通過後を示す。
SUPPLY_TAP_INDEX = {
    "Zone 1": 0,
    "Zone 2": 1,
    "Zone 3": 2,
    "Zone 4": 3,
}
# 上記はコイル出口からVAV分岐 直前までの幹ダクト。多段構成にしておくと圧損の寄与が見えやすい。

RETURN_TRUNK_GEOMETRY = BranchGeometry(
    segments=(
        DuctSegment(10.0, 0.50, 0.021, (0.8,), label="Return main 1"),
        DuctSegment(6.0, 0.45, 0.022, (0.7, 0.6), label="Return main 2"),
        DuctSegment(3.5, 0.40, 0.024, (0.9,), label="Return header"),
    ),
    note="Main return path back to the mixing box.",
)
# リターン側も同じ考え方で、還気プランナム→混合箱までの損失をまとめる。


__all__ = [
    "DuctSegment",
    "BranchGeometry",
    "ZONE_BRANCH_GEOMETRY",
    "SUPPLY_TRUNK_GEOMETRY",
    "RETURN_TRUNK_GEOMETRY",
    "DEFAULT_BRANCH_GEOMETRY",
    "SUPPLY_TAP_INDEX",
]
