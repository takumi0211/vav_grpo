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

# ---------------------------------------------------------------------------
# 標準/外部ライブラリを読み込み、強化学習エージェントで利用する機能を準備
# ---------------------------------------------------------------------------
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# システムシミュレータ側のユーティリティをインポートし、RL環境で再利用
# ---------------------------------------------------------------------------
from simulator_humid.simulation import (
    HVACActions,
    ZoneConfig,
    build_default_zones as simulation_build_default_zones,
    create_plots,
    outdoor_temperature,
    run_simulation,
)
from simulator_humid.utils.paths import RL_OUTPUT_DIR

# ---------------------------------------------------------------------------
# 学習パラメータ全体をまとめる設定コンテナ
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    # === 学習パラメータ ===
    episodes: int = 100_000  # 学習エピソード数
    gamma: float = 0.995  # 割引率
    actor_lr: float = 3e-4  # アクター学習率
    critic_lr: float = 3e-4  # クリティック学習率
    alpha_lr: float = 3e-4  # エントロピー温度学習率
    adam_eps: float = 1e-5  # Adam optimizer epsilon
    tau: float = 0.005  # ターゲットネットワーク更新率
    target_entropy_scale: float = 1.0  # エントロピー目標スケール
    initial_alpha: float = 0.2  # 初期エントロピー温度
    batch_size: int = 512  # バッチサイズ
    replay_buffer_size: int = 200_000  # リプレイバッファサイズ
    warmup_steps: int = 5_000  # ウォームアップステップ数
    gradient_steps_per_env_step: float = 1.0  # 環境ステップあたりの勾配更新数

    # === 報酬設計パラメータ ===
    temp_sigma: float = 0.6  # 温度快適性の標準偏差（0.5℃ズレで≈0.7）
    comfort_center: float = 26.0  # 快適温度の中心
    comfort_low: float = 25.0  # 快適温度下限
    comfort_high: float = 27.0  # 快適温度上限
    comfort_penalty: float = 2.0  # 快適性違反ペナルティ
    power_weight: float = 0.2  # エネルギー消費ペナルティ重み
    off_hvac_weight: float = 0.2  # HVAC停止時ペナルティ重み
    co2_target_ppm: float = 1000.0  # CO₂快適基準
    co2_penalty_weight: float = 0.5  # CO₂ペナルティ重み
    co2_logistic_k: float = 12.0  # ロジスティック関数の傾き係数
    co2_violation_threshold: float = 1100.0  # CO₂違反閾値
    co2_violation_penalty: float = 10.0  # CO₂違反ペナルティ重み

    # === シミュレーション設定 ===
    timestep_s: int = 60  # タイムステップ（秒）
    episode_minutes: int = 24 * 60  # エピソード長（分）
    start_time: datetime = datetime(2025, 7, 29, 0, 0)  # 開始時刻です
    setpoint: float = 26.0  # 温度設定値

    # === ネットワーク構造 ===
    hidden_sizes: Sequence[int] = (256, 256, 128)  # アクター隠れ層サイズ
    critic_hidden_sizes: Sequence[int] | None = None  # クリティック隠れ層サイズ（None=アクターと同じ）
    obs_norm_clip: float = 5.0  # 観測正規化クリップ値
    obs_norm_eps: float = 1e-6  # 観測正規化イプシロン

    # === 制御パラメータ ===
    damper_min: float = 0.05  # ダンパー最小開度
    damper_max: float = 1.0  # ダンパー最大開度
    oa_min: float = 0.05  # 外気ダンパー最小開度
    oa_max: float = 0.6  # 外気ダンパー最大開度
    coil_min: float = 0.06  # コイルバルブ最小開度
    coil_max: float = 1.0  # コイルバルブ最大開度
    fan_min: float = 0.45  # ファン最小速度
    fan_max: float = 1.4  # ファン最大速度
    control_interval_minutes: int = 5  # 制御更新間隔（分）

    # === 運転スケジュール ===
    hvac_start_hour: float = 8.0  # HVAC開始時刻（時）
    hvac_stop_hour: int = 18  # HVAC終了時刻（時）
    eval_start_hour: int = 8  # 評価開始時刻（時）
    eval_stop_hour: int = 18  # 評価終了時刻（時）

    # === 出力・ログ設定 ===
    output_dir: Path = RL_OUTPUT_DIR  # 出力ディレクトリ
    log_interval: int = 1  # ログ出力間隔
    print_every: int = 50  # コンソール出力間隔
    plot_every: int = 50  # プロット出力間隔
    seed: int = 42  # 乱数シード
    zones: Sequence[ZoneConfig] = field(default_factory=list)  # ゾーン設定



def set_global_seed(seed: int) -> None:
    """乱数シードを固定し、実験の再現性を高める。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# 観測値の平均・分散を逐次更新し、オンライン正規化に利用するユーティリティ
# ---------------------------------------------------------------------------
class RunningMeanStd:

    """Tracks running mean and variance using Welford's algorithm."""

    def __init__(self, shape: int, eps: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, batch: np.ndarray) -> None:

        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        batch = batch.astype(np.float64)
        batch_count = batch.shape[0]
        if batch_count == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / max(tot_count, 1.0)

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = tot_count

    def state_dict(self) -> dict[str, np.ndarray | float]:

        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load_state_dict(self, state: dict[str, np.ndarray | float]) -> None:
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = float(state["count"])

class ObservationNormalizer:

    """観測を正規化して学習を安定化させるラッパー。"""

    def __init__(self, obs_dim: int, clip: float = 5.0, eps: float = 1e-6) -> None:
        self.rms = RunningMeanStd(obs_dim)
        self.clip = clip
        self.eps = eps

    def normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:

        obs = np.nan_to_num(obs.astype(np.float32), nan=0.0, posinf=self.clip, neginf=-self.clip)
        if update:
            self.rms.update(obs)
        mean = self.rms.mean.astype(np.float32)
        var = self.rms.var.astype(np.float32)
        normalized = (obs - mean) / np.sqrt(var + self.eps)
        normalized = np.clip(normalized, -self.clip, self.clip)
        return np.nan_to_num(normalized, nan=0.0, posinf=self.clip, neginf=-self.clip)

    def normalize_tensor(self, obs: torch.Tensor) -> torch.Tensor:

        mean = torch.as_tensor(self.rms.mean, dtype=obs.dtype, device=obs.device)
        var = torch.as_tensor(self.rms.var, dtype=obs.dtype, device=obs.device)
        eps = torch.as_tensor(self.eps, dtype=obs.dtype, device=obs.device)
        normalized = (obs - mean) / torch.sqrt(var + eps)
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return torch.clamp(normalized, -self.clip, self.clip)

    def state_dict(self) -> dict[str, np.ndarray | float]:

        state = self.rms.state_dict()
        state.update({"clip": float(self.clip), "eps": float(self.eps)})
        return state

    def load_state_dict(self, state: dict[str, np.ndarray | float]) -> None:
        self.clip = float(state.get("clip", self.clip))
        self.eps = float(state.get("eps", self.eps))
        self.rms.load_state_dict({k: v for k, v in state.items() if k in {"mean", "var", "count"}})

@dataclass
class ActionScaler:

    """tanh出力と実際のアクチュエータ範囲を相互変換する補助クラス。"""

    low: torch.Tensor
    high: torch.Tensor

    def __post_init__(self) -> None:
        self.range = self.high - self.low
        self.scale = 0.5 * self.range
        self.safe_scale = torch.where(
            self.scale.abs() < 1e-6,
            torch.full_like(self.scale, 1e-6),
            self.scale,
        )
        self.bias = self.low + self.safe_scale

    def to(self, device: torch.device) -> "ActionScaler":

        low = self.low.to(device)
        high = self.high.to(device)
        scaler = ActionScaler(low=low, high=high)
        return scaler

    def scale_action(self, squashed: torch.Tensor) -> torch.Tensor:

        return squashed * self.safe_scale + self.bias

    def unscale_action(self, scaled: torch.Tensor) -> torch.Tensor:
        scaled = scaled.to(self.low.device)
        return torch.clamp((scaled - self.bias) / self.safe_scale, -0.999, 0.999)

    def midpoint(self) -> np.ndarray:
        return (self.low.cpu().numpy() + self.high.cpu().numpy()) * 0.5

class ReplayBuffer:

    """経験再生バッファ。環境とのインタラクション履歴を保存する。"""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(

        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.obs[idx] = obs.astype(np.float32, copy=False)
        self.actions[idx] = action.astype(np.float32, copy=False)
        self.rewards[idx] = float(reward)
        self.next_obs[idx] = next_obs.astype(np.float32, copy=False)
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:

        size = len(self)
        if size == 0:
            raise ValueError("Replay buffer is empty")
        indices = np.random.randint(0, size, size=min(batch_size, size))
        batch = {
            "obs": torch.from_numpy(self.obs[indices]),
            "actions": torch.from_numpy(self.actions[indices]),
            "rewards": torch.from_numpy(self.rewards[indices]).unsqueeze(1),
            "next_obs": torch.from_numpy(self.next_obs[indices]),
            "dones": torch.from_numpy(self.dones[indices]).unsqueeze(1),
        }
        return batch

    def __len__(self) -> int:
        return self.capacity if self.full else self.ptr

def build_mlp(input_dim: int, hidden_sizes: Sequence[int]) -> tuple[nn.Sequential, int]:

    """整形済みMLPと最終出力次元を返すヘルパー。"""

    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        linear = nn.Linear(last_dim, hidden_dim)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())
        last_dim = hidden_dim
    return nn.Sequential(*layers), last_dim

# ---------------------------------------------------------------------------
# SACアクター（ポリシーネットワーク）
# ---------------------------------------------------------------------------
class SACActor(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")
        self.net, last_dim = build_mlp(obs_dim, hidden_sizes)
        self.mean_head = nn.Linear(last_dim, action_dim)
        self.log_std_head = nn.Linear(last_dim, action_dim)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), -5.0, 2.0)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        mean, log_std = self.forward(obs)
        std = log_std.exp()
        base_dist = torch.distributions.Normal(mean, std)
        raw_action = base_dist.rsample()
        squashed = torch.tanh(raw_action)
        log_prob = base_dist.log_prob(raw_action) - torch.log(1.0 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return squashed, log_prob, torch.tanh(mean)

# ---------------------------------------------------------------------------
# 状態・行動ペアの価値を推定するQネットワーク
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")
        self.net, last_dim = build_mlp(obs_dim + action_dim, hidden_sizes)
        self.output = nn.Linear(last_dim, 1)
        nn.init.orthogonal_(self.output.weight, gain=1.0)
        nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        features = self.net(x)
        return self.output(features)

# ---------------------------------------------------------------------------
# クリティックを二重化して過大評価を抑制するツインQネットワーク
# ---------------------------------------------------------------------------
class TwinQNetwork(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_sizes)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_sizes)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

# ---------------------------------------------------------------------------
# シミュレーション環境にポリシーを接続し、観測構築とアクション出力を担当
# ---------------------------------------------------------------------------
class PolicyController:

    def __init__(
        self,
        policy: SACActor,
        scaler: ActionScaler,
        config: TrainingConfig,
        device: torch.device,
        normalizer: ObservationNormalizer,
        update_normalizer: bool = True,
    ) -> None:
        self.policy = policy
        self.scaler = scaler
        self.config = config
        self.device = device
        self.normalizer = normalizer
        self.update_normalizer = update_normalizer
        self.zone_count = len(config.zones)
        self.prev_action = self.scaler.midpoint()
        self.prev_tanh_action = np.zeros_like(self.prev_action)
        self.prev_zone_temps: np.ndarray | None = None
        self.prev_zone_co2: np.ndarray | None = None
        self.prev_outdoor_temp: float | None = None
        self.prev_env_timestamp: datetime | None = None
        self.control_period_s = max(1, int(config.control_interval_minutes * 60))
        self.last_command_time: datetime | None = None
        self.obs_history: List[np.ndarray] = []
        self.action_history: List[np.ndarray] = []
        self.tanh_history: List[np.ndarray] = []
        self.timestamps: List[datetime] = []
        self.decision_flags: List[bool] = []

    def build_observation(
        self,
        timestamp: datetime,
        zone_temps: np.ndarray,
        zone_co2: np.ndarray,
    ) -> np.ndarray:

        """現在の環境状態から学習用の観測ベクトルを組み立てる。"""

        zone_temps = zone_temps.astype(np.float32)
        zone_co2 = zone_co2.astype(np.float32)
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

        temp_error = zone_temps - self.config.setpoint
        co2_error = zone_co2 - float(self.config.co2_target_ppm)
        outdoor = float(outdoor_temperature(timestamp))
        temp_slope = 0.0
        if (
            self.prev_env_timestamp is not None
            and self.prev_outdoor_temp is not None
        ):
            dt_hours = max((timestamp - self.prev_env_timestamp).total_seconds() / 3600.0, 1e-6)
            temp_slope = (outdoor - self.prev_outdoor_temp) / dt_hours
        self.prev_outdoor_temp = outdoor
        self.prev_env_timestamp = timestamp
        minute = timestamp.hour * 60 + timestamp.minute
        angle = 2.0 * math.pi * (minute / 1440.0)
        sin_time = math.sin(angle)
        cos_time = math.cos(angle)
        obs = np.concatenate(
            [
                temp_error,
                temp_delta,
                co2_error,
                co2_delta,
                np.array(
                    [
                        outdoor,
                        temp_slope,
                        sin_time,
                        cos_time,
                    ],
                    dtype=np.float32,
                ),
                self.prev_action.astype(np.float32),
            ]
        )
        return np.nan_to_num(obs, nan=0.0)


    def __call__(
        self,
        timestamp: datetime,
        zone_temps: np.ndarray,
        zone_co2: np.ndarray,
        _zone_rh: np.ndarray,
    ) -> HVACActions:

        """シミュレーションから呼び出され、制御入力を生成する。"""

        obs_vec = self.build_observation(timestamp, zone_temps, zone_co2)
        normalized_obs = self.normalizer.normalize(obs_vec, update=self.update_normalizer)
        obs_tensor = torch.from_numpy(normalized_obs).unsqueeze(0).to(self.device)

        current_hour = timestamp.hour + timestamp.minute / 60.0
        hvac_on = self.config.hvac_start_hour <= current_hour < self.config.hvac_stop_hour
        should_sample = hvac_on and (
            self.last_command_time is None
            or (timestamp - self.last_command_time).total_seconds() >= self.control_period_s - 1e-9
        )

        if should_sample:
            with torch.no_grad():
                squashed, _, _ = self.policy.sample(obs_tensor)
            action_tensor = self.scaler.scale_action(squashed).squeeze(0)
            scaled_np = action_tensor.detach().cpu().numpy()
            self.last_command_time = timestamp
        else:
            scaled_np = self.prev_action.copy()

        scaled_tensor = torch.from_numpy(scaled_np.astype(np.float32, copy=False)).to(self.scaler.low.device)
        tanh_action_tensor = self.scaler.unscale_action(scaled_tensor)
        tanh_action = tanh_action_tensor.detach().cpu().numpy()
        scaled_np = scaled_tensor.detach().cpu().numpy()

        self.prev_action = scaled_np.copy()
        self.prev_tanh_action = tanh_action.copy()

        self.obs_history.append(obs_vec.astype(np.float32, copy=False))
        self.action_history.append(scaled_np.astype(np.float32, copy=False))
        self.tanh_history.append(tanh_action.astype(np.float32, copy=False))
        self.timestamps.append(timestamp)
        self.decision_flags.append(bool(should_sample))

        zone_act = np.clip(scaled_np[: self.zone_count], self.config.damper_min, self.config.damper_max)
        oa = float(np.clip(scaled_np[self.zone_count], self.config.oa_min, self.config.oa_max))
        coil = float(np.clip(scaled_np[self.zone_count + 1], self.config.coil_min, self.config.coil_max))
        fan = float(np.clip(scaled_np[self.zone_count + 2], self.config.fan_min, self.config.fan_max))
        return HVACActions(
            zone_dampers=zone_act.tolist(),
            oa_damper=oa,
            coil_valve=coil,
            fan_speed=fan,
        )

    def trajectory(self) -> dict[str, np.ndarray | List[datetime]]:

        """これまでの観測・行動履歴をまとめて返す。"""

        obs = np.asarray(self.obs_history, dtype=np.float32)
        actions = np.asarray(self.action_history, dtype=np.float32)
        tanh_actions = np.asarray(self.tanh_history, dtype=np.float32)
        return {
            "obs": obs,
            "actions": actions,
            "tanh_actions": tanh_actions,
            "timestamps": list(self.timestamps),
            "decision_mask": np.asarray(self.decision_flags, dtype=bool),
        }

def build_default_zones() -> Sequence[ZoneConfig]:
    """
    RLエージェント用のデフォルトゾーン設定を構築する。
    """
    return simulation_build_default_zones()

def build_action_scaler(config: TrainingConfig) -> ActionScaler:

    """ゾーン数に応じたアクションスケーラを組み立てる。"""
    zone_count = len(config.zones)
    low = np.concatenate(
        [
            np.full(zone_count, config.damper_min, dtype=np.float32),
            np.array([config.oa_min, config.coil_min, config.fan_min], dtype=np.float32),
        ]
    )
    high = np.concatenate(
        [
            np.full(zone_count, config.damper_max, dtype=np.float32),
            np.array([config.oa_max, config.coil_max, config.fan_max], dtype=np.float32),
        ]
    )
    return ActionScaler(low=torch.from_numpy(low), high=torch.from_numpy(high))

def build_simulation_kwargs(config: TrainingConfig) -> dict:

    """シミュレーション実行に必要な引数セットをまとめる。"""
    return {
        "start": config.start_time,
        "minutes": config.episode_minutes,
        "timestep_s": config.timestep_s,
        "zones": list(config.zones),
        "coil_approach": 1.0,
        "coil_ua": 3500.0,
        "coil_dp_pa": 260.0,
        "default_fan_inv": 0.95,
        "static_pressure_limit": 900.0,
        "fan_nominal_flow_m3_min": 100.0,
        "chw_pump_efficiency": 0.8,
        "setpoint": config.setpoint,
        "zone_pid_kp": 0.6,
        "zone_pid_ti": 25.0,
        "zone_pid_t_reset": 30,
        "zone_pid_t_step": 2,
        "hvac_start_hour": config.hvac_start_hour,
        "hvac_stop_hour": config.hvac_stop_hour,
    }

def compute_step_rewards(df: pd.DataFrame, config: TrainingConfig) -> tuple[np.ndarray, np.ndarray]:

    """シミュレーション結果から時系列報酬と学習対象マスクを作成する。"""
    # --- 快適性評価: ガウス報酬と許容帯違反を算出 ---
    zone_temp_cols = sorted(col for col in df.columns if col.startswith("zone") and col.endswith("_temp"))
    temps = np.nan_to_num(df[zone_temp_cols].to_numpy(), nan=config.setpoint)
    temp_error = temps - config.comfort_center

    sigma = max(config.temp_sigma, 1e-3)
    gaussian = np.exp(-0.5 * (temp_error / sigma) ** 2)
    comfort_reward = gaussian.mean(axis=1)

    lower_violation = np.clip(config.comfort_low - temps, 0.0, None)
    upper_violation = np.clip(temps - config.comfort_high, 0.0, None)
    band_violation = lower_violation + upper_violation
    band_penalty = (band_violation ** 2).mean(axis=1)

    # --- エネルギー消費: ファン・ポンプ・チラー電力を合算 ---
    power_kw = (
        np.nan_to_num(df["fan_power_kw"].to_numpy(), nan=0.0)
        + np.nan_to_num(df["chw_pump_power_kw"].to_numpy(), nan=0.0)
        + np.nan_to_num(df["chiller_power_kw"].to_numpy(), nan=0.0)
    )

    # --- 室内CO2の罰則: ロジスティック関数と閾値超過を導入 ---
    co2_cols = sorted(col for col in df.columns if col.startswith("zone") and col.endswith("_co2_ppm"))
    if co2_cols:
        zone_co2 = np.nan_to_num(df[co2_cols].to_numpy(), nan=config.co2_target_ppm)
        peak_co2 = np.max(zone_co2, axis=1)
        logistic_arg = np.clip(
            (peak_co2 - config.co2_target_ppm) / max(config.co2_logistic_k, 1e-6),
            -60.0,
            60.0,
        )
        co2_penalty = 1.0 / (1.0 + np.exp(-logistic_arg))
        
        # CO2濃度が1100ppmを超えた場合の大きなペナルティ（一定値）
        co2_violation_mask = peak_co2 > config.co2_violation_threshold
        co2_violation_penalty = np.where(co2_violation_mask, config.co2_violation_penalty, 0.0)
    else:
        peak_co2 = np.zeros(len(df), dtype=np.float32)
        co2_penalty = np.zeros(len(df), dtype=np.float32)
        co2_violation_penalty = np.zeros(len(df), dtype=np.float32)



    # --- HVAC稼働マスク: 実際に空調が動いている時間のみ報酬を付与 ---
    hvac_on_col = df.get("hvac_on")
    if hvac_on_col is not None:
        hvac_active_vals = np.nan_to_num(hvac_on_col.to_numpy(), nan=0.0)
        hvac_active_mask = hvac_active_vals > 0.5
    else:
        hvac_active_mask = np.ones(len(df), dtype=bool)

    reward = (
        comfort_reward
        - config.comfort_penalty * band_penalty
        - config.power_weight * power_kw
        - config.co2_penalty_weight * co2_penalty
        - co2_violation_penalty
    )

    df["peak_co2_ppm"] = peak_co2
    df["co2_penalty"] = co2_penalty
    df["co2_violation_penalty"] = co2_violation_penalty

    # --- 時刻スケジュールと組み合わせて、学習対象となるステップを決定 ---
    if not isinstance(df.index, pd.DatetimeIndex):
        time_index = pd.to_datetime(df.index)
    else:
        time_index = df.index
    current_hours = time_index.hour + time_index.minute / 60.0
    hvac_schedule_mask = (
        (current_hours >= config.hvac_start_hour)
        & (current_hours < config.hvac_stop_hour)
    )
    hvac_schedule_mask = np.asarray(hvac_schedule_mask, dtype=bool)

    training_mask = hvac_schedule_mask & hvac_active_mask

    reward *= training_mask.astype(np.float32)
    reward = np.nan_to_num(reward, nan=0.0, neginf=-1e6, posinf=1e6)
    return reward.astype(np.float32), training_mask

def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:

    """指数平滑でターゲットネットワークを更新する。"""
    tau = float(np.clip(tau, 0.0, 1.0))
    with torch.no_grad():
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.lerp_(src_param.data, tau)

# ---------------------------------------------------------------------------
# SAC本体の学習ループ
# ---------------------------------------------------------------------------
def train(config: TrainingConfig) -> None:

    # === 前処理: デバイス・乱数シード・ゾーン情報を確定 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(config.seed)
    config.zones = config.zones or build_default_zones()

    zone_count = len(config.zones)
    action_dim = zone_count + 3
    obs_dim = zone_count * 4 + 4 + action_dim

    actor_hidden = tuple(config.hidden_sizes)
    critic_hidden = tuple(config.critic_hidden_sizes) if config.critic_hidden_sizes else actor_hidden

    # === モデルとオプティマイザの初期化 ===
    actor = SACActor(obs_dim, action_dim, actor_hidden).to(device)
    critic = TwinQNetwork(obs_dim, action_dim, critic_hidden).to(device)
    target_critic = TwinQNetwork(obs_dim, action_dim, critic_hidden).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=config.actor_lr, eps=config.adam_eps)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.critic_lr, eps=config.adam_eps)

    initial_alpha = max(config.initial_alpha, 1e-6)
    log_alpha = torch.tensor(np.log(initial_alpha), device=device, requires_grad=True)
    alpha_optimizer = optim.Adam([log_alpha], lr=config.alpha_lr, eps=config.adam_eps)
    target_entropy = -float(action_dim) * float(config.target_entropy_scale)

    # === 観測正規化・アクションスケーリング・リプレイバッファの準備 ===
    scaler = build_action_scaler(config).to(device)
    normalizer = ObservationNormalizer(obs_dim, clip=config.obs_norm_clip, eps=config.obs_norm_eps)
    replay_buffer = ReplayBuffer(config.replay_buffer_size, obs_dim, action_dim)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = config.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "training_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "episode,total_return,mean_reward,mean_temp_error,mean_power_kw,total_power_kwh,"
            "mean_co2_ppm,max_co2_ppm,mean_co2_penalty\n"
        )

    sim_kwargs = build_simulation_kwargs(config)
    env_steps = 0
    total_updates = 0
    
    # 自己ベスト追跡用変数
    best_total_return = float('-inf')
    best_episode = 0



    # === エピソードループ: シミュレーション実行 → メモリ蓄積 → 学習 ===
    for episode in range(1, config.episodes + 1):
        controller = PolicyController(
            policy=actor,
            scaler=scaler,
            config=config,
            device=device,
            normalizer=normalizer,
            update_normalizer=True,
        )
        # --- シミュレーションを1エピソード実行し時系列データを取得 ---
        df = run_simulation(action_callback=controller, **sim_kwargs)

        trajectory = controller.trajectory()
        obs = trajectory["obs"]
        tanh_actions = trajectory["tanh_actions"]
        decision_mask = np.asarray(trajectory.get("decision_mask"), dtype=bool)
        if obs.size == 0 or decision_mask.size == 0:
            continue

        # --- 報酬系列と学習対象マスクを算出 ---
        rewards, active_mask = compute_step_rewards(df, config)
        if len(rewards) != len(obs) or len(active_mask) != len(obs):
            raise RuntimeError("Mismatch between rewards and recorded transitions")

        num_steps = len(obs)
        if num_steps == 0:
            continue

        decision_indices = np.flatnonzero(decision_mask)
        if decision_indices.size == 0:
            continue

        active_mask_np = np.asarray(active_mask, dtype=bool)

        # --- 意思決定タイミングごとに報酬を集約 ---
        aggregated_rewards: List[float] = []
        next_obs_indices: List[int] = []
        for i, idx in enumerate(decision_indices):
            next_idx_for_reward = decision_indices[i + 1] if i + 1 < decision_indices.size else len(rewards)
            next_idx_for_obs = decision_indices[i + 1] if i + 1 < decision_indices.size else num_steps - 1
            interval_reward = float(np.sum(rewards[idx:next_idx_for_reward]))
            aggregated_rewards.append(interval_reward)
            next_obs_indices.append(int(next_idx_for_obs))

        decision_obs = obs[decision_indices]
        decision_actions = tanh_actions[decision_indices]
        decision_next_obs = obs[next_obs_indices]
        decision_rewards = np.asarray(aggregated_rewards, dtype=np.float32)
        decision_active_mask = active_mask_np[decision_indices]

        valid_decision_mask = decision_active_mask.astype(bool)
        if not np.any(valid_decision_mask):
            continue

        decision_obs = decision_obs[valid_decision_mask]
        decision_actions = decision_actions[valid_decision_mask]
        decision_next_obs = decision_next_obs[valid_decision_mask]
        decision_rewards = decision_rewards[valid_decision_mask]

        dones = np.zeros((decision_obs.shape[0],), dtype=np.float32)
        if dones.size:
            dones[-1] = 1.0

        # --- 有効な意思決定タイミングのみリプレイバッファに書き込み ---
        for idx in range(decision_obs.shape[0]):
            replay_buffer.add(
                decision_obs[idx],
                decision_actions[idx],
                float(decision_rewards[idx]),
                decision_next_obs[idx],
                bool(dones[idx]),
            )

        env_steps += int(decision_obs.shape[0])



        gradient_target = int(max(0, round(int(decision_obs.shape[0]) * config.gradient_steps_per_env_step)))
        q_losses: List[float] = []
        actor_losses: List[float] = []
        temperature_losses: List[float] = []
        alpha_values: List[float] = []

        # --- バッファが十分に溜まったらSACの勾配更新を実施 ---
        if len(replay_buffer) >= config.batch_size and gradient_target > 0:
            for _ in range(gradient_target):
                batch = replay_buffer.sample(config.batch_size)
                batch = {k: v.to(device=device) for k, v in batch.items()}
                batch["obs"] = normalizer.normalize_tensor(batch["obs"].float())
                batch["next_obs"] = normalizer.normalize_tensor(batch["next_obs"].float())
                batch["actions"] = batch["actions"].float()
                batch["rewards"] = batch["rewards"].float()
                batch["dones"] = batch["dones"].float()

                alpha = log_alpha.exp()

                with torch.no_grad():
                    next_action, next_log_prob, _ = actor.sample(batch["next_obs"])
                    target_q1, target_q2 = target_critic(batch["next_obs"], next_action)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                    target_value = batch["rewards"] + (1.0 - batch["dones"]) * config.gamma * target_q

                current_q1, current_q2 = critic(batch["obs"], batch["actions"])
                critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                q_losses.append(float(critic_loss.detach().cpu().item()))
                total_updates += 1



                if len(replay_buffer) >= config.warmup_steps:
                    pi_action, log_prob, _ = actor.sample(batch["obs"])
                    q1_pi, q2_pi = critic(batch["obs"], pi_action)
                    q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (alpha * log_prob - q_pi).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    actor_losses.append(float(actor_loss.detach().cpu().item()))

                    temperature_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
                    alpha_optimizer.zero_grad()
                    temperature_loss.backward()
                    alpha_optimizer.step()
                    temperature_losses.append(float(temperature_loss.detach().cpu().item()))
                    alpha_values.append(float(alpha.detach().cpu().item()))

                soft_update(critic, target_critic, config.tau)

        # --- ロギング・評価用の統計量を計算 ---
        zone_temp_cols = sorted(
            col for col in df.columns if col.startswith("zone") and col.endswith("_temp")
        )
        zone_temps = df[zone_temp_cols].to_numpy()

        co2_cols = sorted(
            col for col in df.columns if col.startswith("zone") and col.endswith("_co2_ppm")
        )
        zone_co2 = df[co2_cols].to_numpy() if co2_cols else np.empty((len(df), 0))

        mask_np = np.asarray(active_mask, dtype=bool)
        if not mask_np.any():
            mask_np = np.ones(len(df), dtype=bool)

        zone_temps_eval = zone_temps[mask_np]
        abs_temp_error = np.abs(zone_temps_eval - config.setpoint)
        mean_temp_error = float(abs_temp_error.mean()) if abs_temp_error.size else 0.0
        band_ratio = float(
            ((zone_temps_eval >= config.comfort_low) & (zone_temps_eval <= config.comfort_high)).mean()
        ) if zone_temps_eval.size else 0.0

        if co2_cols:
            zone_co2_eval = zone_co2[mask_np]
            if zone_co2_eval.size:
                max_co2_ppm = float(np.max(zone_co2_eval))
                mean_co2_ppm = float(np.mean(zone_co2_eval))
            else:
                max_co2_ppm = 0.0
                mean_co2_ppm = 0.0
            co2_penalty_series = df["co2_penalty"].to_numpy()
            co2_violation_penalty_series = df["co2_violation_penalty"].to_numpy()
            if mask_np.any():
                mean_co2_penalty = float(np.mean(co2_penalty_series[mask_np]))
                mean_co2_violation_penalty = float(np.mean(co2_violation_penalty_series[mask_np]))
            else:
                mean_co2_penalty = float(np.mean(co2_penalty_series)) if co2_penalty_series.size else 0.0
                mean_co2_violation_penalty = float(np.mean(co2_violation_penalty_series)) if co2_violation_penalty_series.size else 0.0
        else:
            max_co2_ppm = 0.0
            mean_co2_ppm = 0.0
            mean_co2_penalty = 0.0
            mean_co2_violation_penalty = 0.0



        power_series = df[["fan_power_kw", "chw_pump_power_kw", "chiller_power_kw"]].sum(axis=1).to_numpy()
        mean_power = float(power_series[mask_np].mean()) if mask_np.any() else float(power_series.mean())
        dt_hours = float(config.timestep_s) / 3600.0
        total_power = (
            float(power_series[mask_np].sum()) * dt_hours if mask_np.any() else float(power_series.sum()) * dt_hours
        )

        filtered_rewards = rewards[mask_np]
        total_return = float(np.sum(filtered_rewards))
        mean_reward = float(np.mean(filtered_rewards)) if filtered_rewards.size else 0.0
        mean_q_loss = float(np.mean(q_losses)) if q_losses else 0.0
        mean_actor_loss = float(np.mean(actor_losses)) if actor_losses else 0.0
        mean_alpha_loss = float(np.mean(temperature_losses)) if temperature_losses else 0.0
        current_alpha = float(log_alpha.exp().detach().cpu().item())
        mean_alpha = float(np.mean(alpha_values)) if alpha_values else current_alpha

        # --- 自己ベスト更新チェック ---
        is_new_best = total_return > best_total_return
        if is_new_best:
            best_total_return = total_return
            best_episode = episode
            # ベスト更新時にテキストへ追記
            best_txt_path = config.output_dir / "best_results.txt"
            try:
                with best_txt_path.open("a") as fp:
                    fp.write(
                        (
                            f"=== ベスト更新 ===\n"
                            f"Episode: {episode}\n"
                            f"total_return: {total_return:.4f}\n"
                            f"平均温度誤差: {mean_temp_error:.3f}°C\n"
                            f"快適域違反率: {max(0.0, 1.0 - band_ratio):.3f}\n"
                            f"平均電力消費: {mean_power:.3f}kW\n"
                            f"総電力量: {total_power:.3f}kWh\n"
                            f"平均CO2濃度: {mean_co2_ppm:.1f}ppm\n"
                            f"最大CO2濃度: {max_co2_ppm:.1f}ppm\n"
                            f"co2_penalty: {mean_co2_penalty:.3f}, co2_violation_penalty: {mean_co2_violation_penalty:.3f}\n\n"
                        )
                    )
            except Exception:
                pass

        # --- CSVログへの書き出し ---
        if episode % config.log_interval == 0:
            with log_path.open("a") as fp:
                fp.write(
                    f"{episode},{total_return:.4f},{mean_reward:.4f},{mean_temp_error:.4f},{mean_power:.4f},{total_power:.4f},"
                    f"{mean_co2_ppm:.1f},{max_co2_ppm:.1f},{mean_co2_penalty:.4f}\n"
                )

        # 通常のprint（50エピソードごと）または自己ベスト更新時
        should_print = (episode % config.print_every == 0) or is_new_best
        if should_print:
            buffer_usage = len(replay_buffer)
            best_marker = " [NEW BEST!]" if is_new_best else ""
            print(
                f"Episode {episode:05d} | return={total_return:.2f} | mean_reward={mean_reward:.4f} | "
                f"mean_temp_error={mean_temp_error:.3f} | within_25_27={band_ratio:.3f} | "
                f"mean_power_kw={mean_power:.3f} | total_power_kwh={total_power:.3f} | q_loss={mean_q_loss:.4f} | pi_loss={mean_actor_loss:.4f} | "
                f"max_co2={max_co2_ppm:.1f}ppm | mean_co2={mean_co2_ppm:.1f}ppm | co2_penalty={mean_co2_penalty:.3f} | co2_violation_penalty={mean_co2_violation_penalty:.3f} | "
                f"alpha={current_alpha:.3f} (mean={mean_alpha:.3f}) | alpha_loss={mean_alpha_loss:.4f} | "
                f"buffer={buffer_usage}/{config.replay_buffer_size} | env_steps={env_steps} | updates={total_updates}{best_marker}",
                flush=True,
            )

        # 通常のplot（50エピソードごと）または自己ベスト更新時
        should_plot = (episode % config.plot_every == 0) or is_new_best
        if should_plot:
            main_plot_path = plot_dir / f"episode_{episode:06d}.png"
            damper_plot_path = plot_dir / f"episode_{episode:06d}_dampers.png"
            create_plots(df, main_plot_path, damper_plot_path)



    # === 学習完了後にモデルと正規化パラメータを保存 ===
    model_path = config.output_dir / "sac_policy_final.pt"
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "target_critic_state_dict": target_critic.state_dict(),
            "log_alpha": float(log_alpha.detach().cpu().item()),
            "config": config,
            "obs_normalizer": normalizer.state_dict(),
        },
        model_path,
    )
    print(f"Saved final policy to {model_path}")

def main() -> None:

    """デフォルト設定で学習ジョブを起動するエントリポイント。"""
    config = TrainingConfig()
    config.zones = build_default_zones()
    train(config)

if __name__ == "__main__":
    # スクリプト直接実行時はそのまま学習を開始
    main()
