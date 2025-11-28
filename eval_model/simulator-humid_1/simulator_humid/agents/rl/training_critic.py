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
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim

# ---------------------------------------------------------------------------
# システムシミュレータ側のユーティリティをインポートし、RL環境で再利用
# ---------------------------------------------------------------------------
from simulator_humid.simulation import (
    HVACActions,
    ZoneConfig,
    build_default_zones as simulation_build_default_zones,
    internal_gain_profile_from_occupancy,
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
    adam_eps: float = 1e-5  # Adam optimizer epsilon
    tau: float = 0.005  # ターゲットネットワーク更新率
    batch_size: int = 512  # バッチサイズ
    replay_buffer_size: int = 200_000  # リプレイバッファサイズ
    warmup_steps: int = 5_000  # ウォームアップステップ数
    gradient_steps_per_env_step: float = 1.0  # 環境ステップあたりの勾配更新数
    policy_noise: float = 0.2  # ターゲットポリシーに加えるガウスノイズ標準偏差
    noise_clip: float = 0.5  # ターゲットポリシーノイズのクリップ範囲
    policy_delay: int = 2  # アクターを更新するタイミングの遅延（ステップ数）
    exploration_noise: float = 0.1  # 環境ステップでのアクションノイズ標準偏差
    cem_iters: int = 3  # CEM反復回数
    cem_samples: int = 64  # CEMサンプル数
    cem_elite_frac: float = 0.2  # CEMエリート比率
    cem_action_smooth_coef: float = 0.0  # CEMでアクション変化を抑制する係数
    cql_alpha: float = 1.0  # CQL正則化の係数
    cql_num_random: int = 10  # CQLでサンプルするランダムアクション数
    cql_num_near: int = 5  # CQLでサンプルする近傍アクション数
    cql_near_sigma: float = 0.1  # 近傍アクションサンプルの標準偏差
    n_step: int = 5  # n-step 収益
    structured_min_prbs: int = 3  # PRBS信号の最短切替間隔（決定ステップ数）
    structured_max_prbs: int = 12  # PRBS信号の最長切替間隔（決定ステップ数）
    structured_rate_limit: float = 0.15  # アクションのステップ間変化制限（スケールド空間）
    structured_ou_sigma: float = 0.3  # OUノイズの標準偏差
    structured_ou_theta: float = 0.05  # OUノイズのθ
    structured_sine_min_period: int = 90  # サイン波の最短周期（分）
    structured_sine_max_period: int = 240  # サイン波の最長周期（分）

    # === 報酬設計パラメータ ===
    temp_sigma: float = 0.6  # 温度快適性の標準偏差（SACと統一）
    comfort_center: float = 26.0  # 快適温度の中心
    comfort_low: float = 25.0  # 快適温度下限
    comfort_high: float = 27.0  # 快適温度上限
    comfort_penalty: float = 2.0  # 快適性違反ペナルティ
    power_weight: float = 0.2  # エネルギー消費ペナルティ重み
    off_hvac_weight: float = 0.2  # HVAC停止時ペナルティ重み
    coil_bonus_weight: float = 0.5  # コイル開度ボーナス係数
    coil_bonus_threshold: float = 0.7  # ボーナスを与え始める開度
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

    # === クリティック強化・教師データ出力オプション ===
    critic_extra_epochs: int = 0  # エピソード終了後に追加するクリティック学習エポック数
    critic_extra_epoch_batches: int | None = None  # 追加エポックあたりのバッチ更新回数（Noneでバッファ全体を1周）
    actor_update_delay_updates: int = 0  # アクター更新を開始するまで待つ勾配更新ステップ数
    critic_weight_decay: float = 0.0  # クリティックOptimizerのL2正則化
    actor_weight_decay: float = 0.0  # アクターOptimizerのL2正則化
    critic_grad_clip: float | None = None  # クリティック勾配のノルムクリップ
    actor_grad_clip: float | None = None  # アクター勾配のノルムクリップ
    export_teacher_every: int = 0  # エピソード毎の教師データ出力間隔（0で無効）
    teacher_dataset_dir: Path = field(default_factory=lambda: RL_OUTPUT_DIR / "teacher_datasets")
    teacher_export_samples: int | None = None  # 教師データに含めるサンプル数（Noneで全件）
    teacher_include_scaled_action: bool = False  # 教師データに実スケールのアクションも含めるか
    teacher_use_target_q: bool = True  # 教師データにターゲットQ値を含めるか



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
        self.discounts = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(

        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        discount: float,
    ) -> None:
        idx = self.ptr
        self.obs[idx] = obs.astype(np.float32, copy=False)
        self.actions[idx] = action.astype(np.float32, copy=False)
        self.rewards[idx] = float(reward)
        self.next_obs[idx] = next_obs.astype(np.float32, copy=False)
        self.dones[idx] = float(done)
        self.discounts[idx] = float(np.clip(discount, 0.0, 1.0))
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
            "discounts": torch.from_numpy(self.discounts[indices]).unsqueeze(1),
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


class StructuredExplorer:

    """構造化探索のための擬似入力信号（PRBS + Sine + OU）。"""

    def __init__(
        self,
        *,
        action_dim: int,
        min_interval: int,
        max_interval: int,
        rate_limit: float | None,
        ou_sigma: float,
        ou_theta: float,
        sine_min_period: int,
        sine_max_period: int,
    ) -> None:
        self.action_dim = int(action_dim)
        self.min_interval = max(1, int(min_interval))
        self.max_interval = max(self.min_interval, int(max_interval))
        self.ou_sigma = float(ou_sigma)
        self.ou_theta = float(ou_theta)
        self.rate_limit = float(rate_limit) if rate_limit is not None else None
        self.sine_min_period = max(1, int(sine_min_period))
        self.sine_max_period = max(self.sine_min_period, int(sine_max_period))
        self._prbs_state = np.ones(self.action_dim, dtype=np.float32)
        self._prbs_countdown = np.random.randint(self.min_interval, self.max_interval + 1, size=self.action_dim)
        self._ou_state = np.zeros(self.action_dim, dtype=np.float32)
        self._prev = np.zeros(self.action_dim, dtype=np.float32)
        self._sine_phase = np.random.uniform(0.0, 2.0 * math.pi, size=self.action_dim).astype(np.float32)
        period_minutes = np.random.uniform(self.sine_min_period, self.sine_max_period, size=self.action_dim)
        # 1分刻みステップでの角速度
        self._sine_omega = (2.0 * math.pi / np.maximum(period_minutes, 1.0)).astype(np.float32)

    def reset(self) -> None:
        self._prbs_state = np.random.choice([-1.0, 1.0], size=self.action_dim).astype(np.float32)
        self._prbs_countdown = np.random.randint(self.min_interval, self.max_interval + 1, size=self.action_dim)
        self._ou_state.fill(0.0)
        self._prev.fill(0.0)
        self._sine_phase = np.random.uniform(0.0, 2.0 * math.pi, size=self.action_dim).astype(np.float32)
        period_minutes = np.random.uniform(self.sine_min_period, self.sine_max_period, size=self.action_dim)
        self._sine_omega = (2.0 * math.pi / np.maximum(period_minutes, 1.0)).astype(np.float32)

    def sample(self) -> np.ndarray:

        for idx in range(self.action_dim):
            self._prbs_countdown[idx] -= 1
            if self._prbs_countdown[idx] <= 0:
                self._prbs_state[idx] = 1.0 if random.random() < 0.5 else -1.0
                self._prbs_countdown[idx] = random.randint(self.min_interval, self.max_interval)

        noise = np.random.randn(self.action_dim).astype(np.float32)
        self._ou_state += self.ou_theta * (-self._ou_state) + self.ou_sigma * noise

        self._sine_phase = (self._sine_phase + self._sine_omega).astype(np.float32)
        sine_component = np.sin(self._sine_phase).astype(np.float32)

        raw = (
            0.55 * self._prbs_state
            + 0.25 * sine_component
            + 0.20 * self._ou_state
        )
        raw = np.clip(raw, -0.999, 0.999)
        if self.rate_limit is not None and self.rate_limit > 0.0:
            raw = np.clip(
                raw,
                self._prev - self.rate_limit,
                self._prev + self.rate_limit,
            )
        self._prev = raw.astype(np.float32, copy=True)
        return self._prev.copy()


def cem_argmax_q(
    *,
    target_critic: TwinQNetwork,
    obs: torch.Tensor,
    action_dim: int,
    config: TrainingConfig,
    prev_action: torch.Tensor | None = None,
) -> torch.Tensor:

    """CEMで Q(s, a) を最大化するアクションを近似的に探索。"""

    batch_size = obs.shape[0]
    device = obs.device
    dtype = obs.dtype
    num_samples = max(1, int(config.cem_samples))
    elite_frac = float(np.clip(config.cem_elite_frac, 1e-3, 1.0))
    elite_k = max(1, int(round(num_samples * elite_frac)))
    iters = max(1, int(config.cem_iters))

    if prev_action is not None:
        mu = prev_action.to(device=device, dtype=dtype).view(batch_size, action_dim)
    else:
        mu = torch.zeros(batch_size, action_dim, device=device, dtype=dtype)
    sigma = torch.full_like(mu, 0.5)

    for _ in range(iters):
        eps = torch.randn(batch_size, num_samples, action_dim, device=device, dtype=dtype)
        samples = torch.clamp(eps * sigma.unsqueeze(1) + mu.unsqueeze(1), -0.999, 0.999)
        obs_rep = obs.unsqueeze(1).expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)
        q1, q2 = target_critic(obs_rep, samples.reshape(batch_size * num_samples, action_dim))
        scores = torch.min(q1, q2).view(batch_size, num_samples)
        if prev_action is not None and config.cem_action_smooth_coef > 0.0:
            prev = prev_action.to(device=device, dtype=dtype).view(batch_size, 1, action_dim)
            penalty = config.cem_action_smooth_coef * torch.sum((samples - prev) ** 2, dim=-1)
            scores = scores - penalty
        elite_idx = torch.topk(scores, k=elite_k, dim=1).indices
        elites = torch.gather(samples, 1, elite_idx.unsqueeze(-1).expand(-1, -1, action_dim))
        mu = elites.mean(dim=1)
        sigma = elites.std(dim=1).clamp_min(1e-3)

    return torch.clamp(mu, -0.999, 0.999)

# ---------------------------------------------------------------------------
# TD3アクター（決定論的ポリシーネットワーク）
# ---------------------------------------------------------------------------
class TD3Actor(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")
        self.net, last_dim = build_mlp(obs_dim, hidden_sizes)
        self.output = nn.Linear(last_dim, action_dim)
        nn.init.orthogonal_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        features = self.net(obs)
        return torch.tanh(self.output(features))

    def sample(
        self,
        obs: torch.Tensor,
        noise_std: float = 0.0,
        noise_clip: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        deterministic = self.forward(obs)
        if noise_std > 0.0:
            noise = torch.randn_like(deterministic) * noise_std
            if noise_clip is not None:
                clip_val = float(abs(noise_clip))
                noise = torch.clamp(noise, -clip_val, clip_val)
            noisy = deterministic + noise
        else:
            noisy = deterministic
        squashed = torch.clamp(noisy, -0.999, 0.999)
        dummy_log_prob = torch.zeros(
            squashed.shape[0],
            1,
            device=squashed.device,
            dtype=squashed.dtype,
        )
        return squashed, dummy_log_prob, deterministic

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

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
        *,
        scaler: ActionScaler,
        config: TrainingConfig,
        device: torch.device,
        normalizer: ObservationNormalizer,
        target_critic: TwinQNetwork | None,
        update_normalizer: bool,
        mode: str = "explore",
        explorer: StructuredExplorer | None = None,
    ) -> None:
        self.scaler = scaler
        self.config = config
        self.device = device
        self.normalizer = normalizer
        self.target_critic = target_critic
        self.update_normalizer = update_normalizer
        self.mode = mode
        zone_count = len(config.zones)
        self.zone_count = zone_count
        self.action_dim = zone_count + 3
        self.rate_limit = float(config.structured_rate_limit) if config.structured_rate_limit is not None else None

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
        self.decision_mask_history: List[bool] = []
        self.timestamps: List[datetime] = []
        self.explorer = explorer or StructuredExplorer(
            action_dim=self.action_dim,
            min_interval=config.structured_min_prbs,
            max_interval=config.structured_max_prbs,
            rate_limit=self.rate_limit,
            ou_sigma=config.structured_ou_sigma,
            ou_theta=config.structured_ou_theta,
            sine_min_period=config.structured_sine_min_period,
            sine_max_period=config.structured_sine_max_period,
        )

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
            if self.mode == "plan" and self.target_critic is not None:
                with torch.no_grad():
                    prev_tanh = torch.from_numpy(self.prev_tanh_action).unsqueeze(0).to(self.device, dtype=obs_tensor.dtype)
                    candidate = cem_argmax_q(
                        target_critic=self.target_critic,
                        obs=obs_tensor,
                        action_dim=self.action_dim,
                        config=self.config,
                        prev_action=prev_tanh,
                    )
                squashed = candidate.squeeze(0)
            else:
                squashed_np = self.explorer.sample()
                squashed = torch.from_numpy(squashed_np).to(self.device, dtype=obs_tensor.dtype)

            squashed = torch.clamp(squashed, -0.999, 0.999)
            action_tensor = self.scaler.scale_action(squashed.unsqueeze(0)).squeeze(0)
            scaled_np = action_tensor.detach().cpu().numpy()
            if self.rate_limit is not None and self.rate_limit > 0.0:
                scaled_np = np.clip(
                    scaled_np,
                    self.prev_action - self.rate_limit,
                    self.prev_action + self.rate_limit,
                )
                action_tensor = torch.from_numpy(scaled_np.astype(np.float32, copy=False)).to(self.device, dtype=squashed.dtype)
                squashed = self.scaler.unscale_action(action_tensor)
            self.last_command_time = timestamp
            tanh_action = np.clip(squashed.detach().cpu().numpy(), -0.999, 0.999)
        else:
            scaled_np = self.prev_action.copy()
            tanh_action = self.prev_tanh_action.copy()

        self.prev_action = scaled_np.copy()
        self.prev_tanh_action = tanh_action.copy()

        self.obs_history.append(obs_vec.astype(np.float32, copy=False))
        self.action_history.append(scaled_np.astype(np.float32, copy=False))
        self.tanh_history.append(tanh_action.astype(np.float32, copy=False))
        self.decision_mask_history.append(should_sample)
        self.timestamps.append(timestamp)

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
        decisions = np.asarray(self.decision_mask_history, dtype=bool)
        return {
            "obs": obs,
            "actions": actions,
            "tanh_actions": tanh_actions,
            "decision_mask": decisions,
            "timestamps": list(self.timestamps),
        }


def aggregate_decision_transitions(
    *,
    trajectory: dict[str, np.ndarray | List[datetime]],
    rewards: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, np.ndarray]:

    obs = np.asarray(trajectory["obs"], dtype=np.float32)
    tanh_actions = np.asarray(trajectory["tanh_actions"], dtype=np.float32)
    decision_mask = np.asarray(trajectory["decision_mask"], dtype=bool)
    if obs.shape[0] != rewards.shape[0]:
        raise ValueError("Observation and reward lengths must match")
    if decision_mask.shape[0] != rewards.shape[0]:
        raise ValueError("Decision mask length mismatch")

    active = np.asarray(active_mask, dtype=bool)
    if active.shape[0] != rewards.shape[0]:
        raise ValueError("Active mask length mismatch")

    decision_indices = np.flatnonzero(decision_mask & active)
    if decision_indices.size == 0:
        return {}

    aggregated_rewards: list[float] = []
    next_indices: list[int] = []
    dones = np.zeros(decision_indices.size, dtype=np.float32)

    for idx_pos, idx in enumerate(decision_indices):
        if idx_pos + 1 < decision_indices.size:
            next_idx = decision_indices[idx_pos + 1]
        else:
            next_idx = rewards.shape[0]
        reward_slice = rewards[idx:next_idx]
        aggregated_rewards.append(float(np.sum(reward_slice)))
        if idx_pos + 1 < decision_indices.size:
            next_indices.append(decision_indices[idx_pos + 1])
        else:
            # terminal: use final available observation
            next_indices.append(min(next_idx - 1, obs.shape[0] - 1))
            dones[idx_pos] = 1.0

    dones[-1] = 1.0
    decision_obs = obs[decision_indices]
    decision_actions = tanh_actions[decision_indices]
    decision_next_obs = obs[np.asarray(next_indices, dtype=np.int64)]
    return {
        "obs": decision_obs.astype(np.float32, copy=False),
        "actions": decision_actions.astype(np.float32, copy=False),
        "next_obs": decision_next_obs.astype(np.float32, copy=False),
        "rewards": np.asarray(aggregated_rewards, dtype=np.float32),
        "dones": dones.astype(np.float32, copy=False),
    }


def make_n_step_targets(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    n_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    count = len(rewards)
    if count == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    n = max(1, int(n_step))
    gamma = float(gamma)
    returns = np.zeros(count, dtype=np.float32)
    discounts = np.zeros(count, dtype=np.float32)
    bootstrap_indices = np.zeros(count, dtype=np.int64)

    for i in range(count):
        acc = 0.0
        discount = 1.0
        steps = 0
        j = i
        while True:
            acc += discount * float(rewards[j])
            steps += 1
            done = bool(dones[j] > 0.5)
            if done or steps >= n or j >= count - 1:
                break
            discount *= gamma
            j += 1
        returns[i] = acc
        done_final = bool(dones[j] > 0.5) or j >= count - 1
        if done_final:
            discounts[i] = 0.0
            bootstrap_indices[i] = min(j, count - 1)
        else:
            discount *= gamma
            discounts[i] = discount
            bootstrap_indices[i] = min(j + 1, count - 1)

    return returns, discounts, bootstrap_indices

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


def export_teacher_dataset(
    *,
    episode: int,
    total_updates: int,
    config: TrainingConfig,
    replay_buffer: ReplayBuffer,
    normalizer: ObservationNormalizer,
    critic: TwinQNetwork,
    target_critic: TwinQNetwork,
    device: torch.device,
    scaler: ActionScaler,
    tag: str | None = None,
) -> Path | None:
    """リプレイバッファからサンプルを抽出し、クリティック教師データセットを保存する。"""
    buffer_size = len(replay_buffer)
    if buffer_size == 0 or config.export_teacher_every == 0:
        return None

    max_samples = config.teacher_export_samples
    sample_size = buffer_size if max_samples is None else min(buffer_size, int(max_samples))
    if sample_size <= 0:
        return None

    indices = np.random.choice(buffer_size, size=sample_size, replace=False)
    chunk_size = max(int(config.batch_size), 1024)

    obs_raw_chunks: list[np.ndarray] = []
    obs_norm_chunks: list[np.ndarray] = []
    next_obs_raw_chunks: list[np.ndarray] = []
    next_obs_norm_chunks: list[np.ndarray] = []
    action_tanh_chunks: list[np.ndarray] = []
    reward_chunks: list[np.ndarray] = []
    done_chunks: list[np.ndarray] = []
    q1_chunks: list[np.ndarray] = []
    q2_chunks: list[np.ndarray] = []
    target_q_chunks: list[np.ndarray] = []
    scaled_action_chunks: list[np.ndarray] = []
    discount_chunks: list[np.ndarray] = []

    critic_was_training = critic.training
    target_critic_was_training = target_critic.training

    critic.eval()
    target_critic.eval()

    action_dim = scaler.low.numel()

    for start in range(0, sample_size, chunk_size):
        end = min(start + chunk_size, sample_size)
        idx_chunk = indices[start:end]

        obs_chunk = torch.from_numpy(replay_buffer.obs[idx_chunk]).to(device=device, dtype=torch.float32)
        next_obs_chunk = torch.from_numpy(replay_buffer.next_obs[idx_chunk]).to(device=device, dtype=torch.float32)
        actions_chunk = torch.from_numpy(replay_buffer.actions[idx_chunk]).to(device=device, dtype=torch.float32)
        rewards_chunk = torch.from_numpy(replay_buffer.rewards[idx_chunk]).view(-1, 1).to(device=device, dtype=torch.float32)
        dones_chunk = torch.from_numpy(replay_buffer.dones[idx_chunk]).view(-1, 1).to(device=device, dtype=torch.float32)
        discounts_chunk = torch.from_numpy(replay_buffer.discounts[idx_chunk]).view(-1, 1).to(device=device, dtype=torch.float32)

        obs_norm = normalizer.normalize_tensor(obs_chunk)
        next_obs_norm = normalizer.normalize_tensor(next_obs_chunk)

        with torch.no_grad():
            q1, q2 = critic(obs_norm, actions_chunk)
            target_q_chunk = None
            if config.teacher_use_target_q:
                target_actions = cem_argmax_q(
                    target_critic=target_critic,
                    obs=next_obs_norm,
                    action_dim=int(action_dim),
                    config=config,
                    prev_action=None,
                )
                target_q1, target_q2 = target_critic(next_obs_norm, target_actions)
                target_min = torch.min(target_q1, target_q2)
                target_q_chunk = rewards_chunk + discounts_chunk * target_min

        obs_raw_chunks.append(obs_chunk.cpu().numpy())
        obs_norm_chunks.append(obs_norm.cpu().numpy())
        next_obs_raw_chunks.append(next_obs_chunk.cpu().numpy())
        next_obs_norm_chunks.append(next_obs_norm.cpu().numpy())
        action_tanh_chunks.append(actions_chunk.cpu().numpy())
        reward_chunks.append(rewards_chunk.cpu().numpy())
        done_chunks.append(dones_chunk.cpu().numpy())
        discount_chunks.append(discounts_chunk.cpu().numpy())
        q1_chunks.append(q1.cpu().numpy())
        q2_chunks.append(q2.cpu().numpy())
        if target_q_chunk is not None:
            target_q_chunks.append(target_q_chunk.cpu().numpy())
        if config.teacher_include_scaled_action:
            scaled = scaler.scale_action(actions_chunk)
            scaled_action_chunks.append(scaled.cpu().numpy())

    dataset_dir = config.teacher_dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if tag:
        safe_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in tag)
        tag_part = f"_{safe_tag}"
    else:
        tag_part = ""
    file_path = dataset_dir / f"critic_dataset_ep{episode:06d}_upd{total_updates:08d}{tag_part}.npz"

    rms_state = normalizer.state_dict()
    data_dict: dict[str, np.ndarray] = {
        "obs_raw": np.concatenate(obs_raw_chunks, axis=0).astype(np.float32, copy=False),
        "obs_normalized": np.concatenate(obs_norm_chunks, axis=0).astype(np.float32, copy=False),
        "next_obs_raw": np.concatenate(next_obs_raw_chunks, axis=0).astype(np.float32, copy=False),
        "next_obs_normalized": np.concatenate(next_obs_norm_chunks, axis=0).astype(np.float32, copy=False),
        "action_tanh": np.concatenate(action_tanh_chunks, axis=0).astype(np.float32, copy=False),
        "rewards": np.concatenate(reward_chunks, axis=0).astype(np.float32, copy=False),
        "dones": np.concatenate(done_chunks, axis=0).astype(np.float32, copy=False),
        "discounts": np.concatenate(discount_chunks, axis=0).astype(np.float32, copy=False),
        "q1": np.concatenate(q1_chunks, axis=0).astype(np.float32, copy=False),
        "q2": np.concatenate(q2_chunks, axis=0).astype(np.float32, copy=False),
        "obs_norm_mean": np.asarray(rms_state["mean"], dtype=np.float32),
        "obs_norm_var": np.asarray(rms_state["var"], dtype=np.float32),
        "obs_norm_count": np.asarray([rms_state["count"]], dtype=np.float64),
        "discount_gamma": np.asarray([config.gamma], dtype=np.float32),
    }

    if target_q_chunks:
        data_dict["target_q"] = np.concatenate(target_q_chunks, axis=0).astype(np.float32, copy=False)
    if scaled_action_chunks:
        data_dict["action_scaled"] = np.concatenate(scaled_action_chunks, axis=0).astype(np.float32, copy=False)

    np.savez_compressed(file_path, **data_dict)

    if critic_was_training:
        critic.train()
    if target_critic_was_training:
        target_critic.train()

    print(f"[Critic] Exported critic teacher dataset: {file_path} ({sample_size} samples)")
    return file_path

# ---------------------------------------------------------------------------
# クリティック（FQI + CQL）学習ループ
# ---------------------------------------------------------------------------
def train(config: TrainingConfig) -> None:

    # === 前処理: デバイス・乱数シード・ゾーン情報を確定 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(config.seed)
    config.zones = config.zones or build_default_zones()

    zone_count = len(config.zones)
    action_dim = zone_count + 3
    obs_dim = zone_count * 4 + 4 + action_dim

    critic_hidden = tuple(config.critic_hidden_sizes) if config.critic_hidden_sizes else tuple(config.hidden_sizes)

    # === モデルとオプティマイザの初期化（クリティックのみ） ===
    critic = TwinQNetwork(obs_dim, action_dim, critic_hidden).to(device)
    target_critic = TwinQNetwork(obs_dim, action_dim, critic_hidden).to(device)
    target_critic.load_state_dict(critic.state_dict())
    critic_optimizer = optim.Adam(
        critic.parameters(),
        lr=config.critic_lr,
        eps=config.adam_eps,
        weight_decay=config.critic_weight_decay,
    )

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
    checkpoint_dir = config.output_dir / "critic_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sim_kwargs = build_simulation_kwargs(config)
    env_steps = 0
    total_updates = 0
    
    # 自己ベスト追跡用変数
    best_total_return = float('-inf')
    best_episode = 0
    last_teacher_export_episode: int | None = None

    def critic_gradient_step() -> tuple[float, float, float, float]:
        """FQI + CQL に基づくクリティックの1ステップ更新を実行する。"""
        nonlocal total_updates
        batch = replay_buffer.sample(config.batch_size)
        obs_batch = batch["obs"].to(device=device, dtype=torch.float32)
        actions_batch = batch["actions"].to(device=device, dtype=torch.float32)
        rewards_batch = batch["rewards"].to(device=device, dtype=torch.float32)
        next_obs_batch = batch["next_obs"].to(device=device, dtype=torch.float32)
        discounts_batch = batch["discounts"].to(device=device, dtype=torch.float32)

        obs_norm = normalizer.normalize_tensor(obs_batch)
        next_obs_norm = normalizer.normalize_tensor(next_obs_batch)

        with torch.no_grad():
            next_actions = cem_argmax_q(
                target_critic=target_critic,
                obs=next_obs_norm,
                action_dim=action_dim,
                config=config,
                prev_action=None,
            )
            target_q1, target_q2 = target_critic(next_obs_norm, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards_batch + discounts_batch * target_q

        current_q1, current_q2 = critic(obs_norm, actions_batch)
        bellman_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        batch_size = obs_norm.shape[0]
        n_rand = max(1, int(config.cql_num_random))
        obs_rand = obs_norm.unsqueeze(1).expand(-1, n_rand, -1).reshape(batch_size * n_rand, -1)
        rand_actions = torch.empty(batch_size, n_rand, action_dim, device=device, dtype=obs_norm.dtype).uniform_(-0.999, 0.999)
        rand_q = torch.min(
            *critic(obs_rand, rand_actions.reshape(batch_size * n_rand, action_dim))
        ).view(batch_size, n_rand)

        cql_logits = rand_q
        n_near = max(0, int(config.cql_num_near))
        if n_near > 0:
            noise = torch.randn(batch_size, n_near, action_dim, device=device, dtype=obs_norm.dtype) * float(config.cql_near_sigma)
            near_actions = torch.clamp(actions_batch.unsqueeze(1) + noise, -0.999, 0.999)
            near_obs = obs_norm.unsqueeze(1).expand(-1, n_near, -1).reshape(batch_size * n_near, -1)
            near_q = torch.min(
                *critic(near_obs, near_actions.reshape(batch_size * n_near, action_dim))
            ).view(batch_size, n_near)
            cql_logits = torch.cat([cql_logits, near_q], dim=1)

        data_q = torch.min(*critic(obs_norm, actions_batch))
        cql_term = torch.logsumexp(cql_logits, dim=1).mean() - data_q.mean()
        cql_loss = cql_term * float(config.cql_alpha)

        total_loss = bellman_loss + cql_loss
        critic_optimizer.zero_grad()
        total_loss.backward()
        if config.critic_grad_clip is not None and config.critic_grad_clip > 0.0:
            nn_utils.clip_grad_norm_(critic.parameters(), config.critic_grad_clip)
        critic_optimizer.step()
        soft_update(critic, target_critic, config.tau)

        total_updates += 1
        if total_updates % 1000 == 0:
            ckpt_path = checkpoint_dir / f"critic_update_{total_updates:08d}.pt"
            torch.save(
                {
                    "critic_state_dict": critic.state_dict(),
                    "target_critic_state_dict": target_critic.state_dict(),
                    "config": config,
                    "obs_normalizer": normalizer.state_dict(),
                    "total_updates": total_updates,
                },
                ckpt_path,
            )
            print(f"[Critic] Saved periodic checkpoint at {ckpt_path}")
        return (
            float(total_loss.detach().cpu().item()),
            float(bellman_loss.detach().cpu().item()),
            float(cql_loss.detach().cpu().item()),
            float(data_q.mean().detach().cpu().item()),
        )



    # === エピソードループ: シミュレーション実行 → メモリ蓄積 → 学習 ===
    for episode in range(1, config.episodes + 1):
        controller = PolicyController(
            scaler=scaler,
            config=config,
            device=device,
            normalizer=normalizer,
            target_critic=target_critic,
            update_normalizer=True,
            mode="explore",
        )
        controller.explorer.reset()
        # --- シミュレーションを1エピソード実行し時系列データを取得 ---
        df = run_simulation(action_callback=controller, **sim_kwargs)

        trajectory = controller.trajectory()
        obs = trajectory["obs"]
        if obs.size == 0:
            continue

        # --- 報酬系列と学習対象マスクを算出 ---
        rewards, active_mask = compute_step_rewards(df, config)
        if len(rewards) != len(obs):
            raise RuntimeError("Mismatch between rewards and recorded transitions")

        aggregates = aggregate_decision_transitions(
            trajectory=trajectory,
            rewards=rewards,
            active_mask=active_mask,
        )
        if not aggregates:
            continue

        decision_obs = aggregates["obs"]
        decision_actions = aggregates["actions"]
        decision_rewards = aggregates["rewards"]
        decision_dones = aggregates["dones"]

        returns_n, discounts_n, bootstrap_indices = make_n_step_targets(
            decision_rewards,
            decision_dones,
            config.gamma,
            config.n_step,
        )
        bootstrap_obs = decision_obs[bootstrap_indices]

        transitions_added = 0
        for idx_dec in range(decision_obs.shape[0]):
            reward_n = float(returns_n[idx_dec])
            discount_n = float(np.clip(discounts_n[idx_dec], 0.0, 1.0))
            done_flag = bool(discount_n <= 1e-6)
            replay_buffer.add(
                decision_obs[idx_dec],
                decision_actions[idx_dec],
                reward_n,
                bootstrap_obs[idx_dec],
                done_flag,
                discount_n,
            )
            transitions_added += 1

        env_steps += transitions_added
        if transitions_added == 0:
            continue



        gradient_target = int(max(0, round(transitions_added * config.gradient_steps_per_env_step)))
        total_losses: List[float] = []
        bellman_losses: List[float] = []
        cql_losses: List[float] = []
        data_q_estimates: List[float] = []

        # --- バッファが十分に溜まったらTD3の勾配更新を実施 ---
        ready_threshold = max(int(config.batch_size), int(config.warmup_steps))
        if len(replay_buffer) >= ready_threshold and gradient_target > 0:
            for _ in range(gradient_target):
                total_loss, bellman_loss, cql_loss_value, data_q = critic_gradient_step()
                total_losses.append(total_loss)
                bellman_losses.append(bellman_loss)
                cql_losses.append(cql_loss_value)
                data_q_estimates.append(data_q)

        # --- 追加のクリティック学習エポック（教師データ用にQ値を磨く） ---
        if config.critic_extra_epochs > 0 and len(replay_buffer) >= ready_threshold:
            batches_per_epoch = config.critic_extra_epoch_batches
            if not batches_per_epoch or batches_per_epoch <= 0:
                batches_per_epoch = max(len(replay_buffer) // config.batch_size, 1)
            for _ in range(config.critic_extra_epochs):
                for _ in range(batches_per_epoch):
                    total_loss, bellman_loss, cql_loss_value, data_q = critic_gradient_step()
                    total_losses.append(total_loss)
                    bellman_losses.append(bellman_loss)
                    cql_losses.append(cql_loss_value)
                    data_q_estimates.append(data_q)

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
        mean_total_loss = float(np.mean(total_losses)) if total_losses else 0.0
        mean_bellman_loss = float(np.mean(bellman_losses)) if bellman_losses else 0.0
        mean_cql_loss = float(np.mean(cql_losses)) if cql_losses else 0.0
        mean_data_q = float(np.mean(data_q_estimates)) if data_q_estimates else 0.0

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
                f"mean_power_kw={mean_power:.3f} | total_power_kwh={total_power:.3f} | total_loss={mean_total_loss:.4f} | bellman={mean_bellman_loss:.4f} | cql={mean_cql_loss:.4f} | q_data={mean_data_q:.3f} | "
                f"max_co2={max_co2_ppm:.1f}ppm | mean_co2={mean_co2_ppm:.1f}ppm | co2_penalty={mean_co2_penalty:.3f} | co2_violation_penalty={mean_co2_violation_penalty:.3f} | "
                f"buffer={buffer_usage}/{config.replay_buffer_size} | env_steps={env_steps} | updates={total_updates}{best_marker}",
                flush=True,
            )

        # 通常のplot（50エピソードごと）または自己ベスト更新時
        should_plot = (episode % config.plot_every == 0) or is_new_best
        if should_plot:
            main_plot_path = plot_dir / f"episode_{episode:06d}.png"
            damper_plot_path = plot_dir / f"episode_{episode:06d}_dampers.png"
            create_plots(df, main_plot_path, damper_plot_path)

        if config.export_teacher_every > 0 and (episode % config.export_teacher_every == 0):
            export_teacher_dataset(
                episode=episode,
                total_updates=total_updates,
                config=config,
                replay_buffer=replay_buffer,
                normalizer=normalizer,
                critic=critic,
                target_critic=target_critic,
                device=device,
                scaler=scaler,
            )
            last_teacher_export_episode = episode

    if config.export_teacher_every != 0 and last_teacher_export_episode != config.episodes:
        export_teacher_dataset(
            episode=config.episodes,
            total_updates=total_updates,
            config=config,
            replay_buffer=replay_buffer,
            normalizer=normalizer,
            critic=critic,
            target_critic=target_critic,
            device=device,
            scaler=scaler,
            tag="final",
        )
        last_teacher_export_episode = config.episodes

    # === 学習完了後にモデルと正規化パラメータを保存 ===
    model_path = config.output_dir / "critic_model_final.pt"
    torch.save(
        {
            "critic_state_dict": critic.state_dict(),
            "target_critic_state_dict": target_critic.state_dict(),
            "config": config,
            "obs_normalizer": normalizer.state_dict(),
        },
        model_path,
    )
    print(f"Saved critic checkpoint to {model_path}")

def main() -> None:

    """デフォルト設定で学習ジョブを起動するエントリポイント。"""
    config = TrainingConfig()
    config.zones = build_default_zones()
    train(config)

if __name__ == "__main__":
    # スクリプト直接実行時はそのまま学習を開始
    main()
