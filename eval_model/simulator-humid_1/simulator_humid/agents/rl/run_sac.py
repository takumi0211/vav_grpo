#!/usr/bin/env python3
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

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

import simulator_humid.simulation as sim_module
from simulator_humid.agents.llm_agent import (
    build_default_zones as build_llm_zones,
    compute_llm_metrics,
)
from simulator_humid.agents.rl.training_sac import (
    ObservationNormalizer,
    PolicyController,
    SACActor,
    TrainingConfig,
    _prepare_weather_profile,
    build_action_scaler,
    build_simulation_kwargs,
    compute_step_rewards as compute_sac_rewards,
)
from simulator_humid.simulation import create_plots, run_simulation
from simulator_humid.utils.paths import RL_OUTPUT_DIR, WEATHER_DATA_DIR

# ---------------------------------------------------------------------------
# 実行パラメータ（=の右側を書き換えて使用）
# ・天気データ: weather_data/outdoor_temp_20250729.csv（LLM runと共通）
# ・人数データ: simulator_humid/simulation.py に定義された
#   DEFAULT_ZONE_OCCUPANCY_WEEKDAY / WEEKEND（LLM runと共通）
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = RL_OUTPUT_DIR / "sac_policy_final_no_randomize_occupancy.pt"  # 学習済みSACチェックポイントのパス
OUTPUT_DIR = RL_OUTPUT_DIR / "sac_eval_no_randomize_occupancy"  # 評価結果を保存するディレクトリ
RUN_TAG = None  # サブフォルダ名（Noneの場合は現在時刻）
DEVICE_NAME = None  # PyTorchデバイス指定（例: "cuda", "cuda:0"。Noneで自動判定）
WEATHER_FILE = WEATHER_DATA_DIR / "outdoor_temp_20250729.csv"  # 気象条件CSV（LLMと同一）
VERBOSE_STEPS = False  # Trueにするとシミュレーション各ステップを出力
DISABLE_PLOTS = False  # Trueにするとプロット生成をスキップ


def resolve_device(requested: Optional[str]) -> torch.device:
    """希望されたデバイス名を優先しつつ、利用可能なPyTorchデバイスを決定する。"""
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enforce_llm_conditions(config: TrainingConfig, weather_csv: Path) -> None:
    """LLM実験と同じ気象条件・在室条件になるようにRL設定を書き換える。"""
    if not weather_csv.exists():
        raise FileNotFoundError(f"Weather file not found: {weather_csv}")

    config.zones = tuple(build_llm_zones())
    config.start_time = datetime(2025, 7, 29, 0, 0)
    config.episode_minutes = 24 * 60
    config.hvac_start_hour = 8.0
    config.hvac_stop_hour = 18

    df = pd.read_csv(weather_csv)
    minutes, temps, rhs = _prepare_weather_profile(df, weather_csv)
    sim_module.OUTDOOR_DATA_PATH = weather_csv
    sim_module.OUTDOOR_MINUTES = minutes
    sim_module.OUTDOOR_TEMPS = temps
    sim_module.OUTDOOR_RHS = rhs


def load_sac_policy(
    checkpoint_path: Path,
    device: torch.device,
    weather_csv: Path,
) -> tuple[SACActor, ObservationNormalizer, TrainingConfig]:
    """SACチェックポイントを読み込み、アクターと正規化器、コンフィグを復元する。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,  # TrainingConfigや正規化器の状態も復元するため信頼できる環境で実行
    )
    if "config" not in checkpoint or "actor_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing required keys ('config', 'actor_state_dict').")

    raw_config: TrainingConfig = checkpoint["config"]
    config = replace(raw_config)
    enforce_llm_conditions(config, weather_csv)

    zone_count = len(config.zones)
    action_dim = zone_count + 3
    obs_dim = zone_count * 4 + 4 + action_dim

    actor_hidden: Sequence[int] = tuple(config.hidden_sizes)
    actor = SACActor(obs_dim, action_dim, actor_hidden).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    normalizer = ObservationNormalizer(obs_dim, clip=config.obs_norm_clip, eps=config.obs_norm_eps)
    if "obs_normalizer" in checkpoint:
        normalizer.load_state_dict(checkpoint["obs_normalizer"])
    else:
        raise ValueError("Checkpoint missing 'obs_normalizer' statistics.")

    return actor, normalizer, config


def run_sac_episode(
    *,
    actor: SACActor,
    normalizer: ObservationNormalizer,
    config: TrainingConfig,
    device: torch.device,
    verbose_steps: bool,
) -> tuple[np.ndarray, np.ndarray, "pd.DataFrame"]:
    """指定したSACポリシーでシミュレーションを1本実行し、報酬や結果を返す。"""
    scaler = build_action_scaler(config).to(device)
    controller = PolicyController(
        policy=actor,
        scaler=scaler,
        config=config,
        device=device,
        normalizer=normalizer,
        update_normalizer=False,
        exploration_noise=0.0,
    )
    sim_kwargs = build_simulation_kwargs(config)
    with torch.no_grad():
        df = run_simulation(action_callback=controller, verbose_steps=verbose_steps, **sim_kwargs)

    rewards, mask = compute_sac_rewards(df, config)
    df["sac_reward"] = rewards
    return rewards, mask, df


def save_artifacts(
    *,
    df,
    output_dir: Path,
    disable_plots: bool,
) -> tuple[Path, Optional[Path]]:
    """シミュレーション結果CSVとプロットを所定ディレクトリに保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "sac_simulation_results.csv"
    df.to_csv(results_path, index=True)

    plots_dir: Optional[Path] = None
    if not disable_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        create_plots(df, plots_dir / "sac_run.png", plots_dir / "sac_run_dampers.png")
    return results_path, plots_dir


def summarize_run(
    *,
    rewards: np.ndarray,
    mask: np.ndarray,
    df,
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
    setpoint: float,
) -> Path:
    """報酬とLLM指標を要約し、JSONに書き出してコンソールにも整形表示する。"""
    active_steps = int(mask.sum())
    total_reward = float(np.sum(rewards))
    mean_reward = float(np.mean(rewards[mask])) if active_steps > 0 else 0.0
    metrics = compute_llm_metrics(df, setpoint=setpoint)

    summary = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "run_id": run_dir.name,
        "total_reward": total_reward,
        "active_steps": active_steps,
        "mean_reward_active": mean_reward,
        "llm_metrics": metrics,
        "results_csv": str((run_dir / "sac_simulation_results.csv").resolve()),
        "plots_dir": str((run_dir / "plots").resolve()) if (run_dir / "plots").exists() else None,
    }

    summary_path = run_dir / "sac_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"[SAC] total_reward={total_reward:.3f}, active_steps={active_steps}, "
        f"mean_reward_active={mean_reward:.4f}"
    )
    print("[SAC] LLM-aligned metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")

    return summary_path


def main() -> None:
    """パラメータセクションの値を使ってSAC評価をまとめて実行する。"""
    device = resolve_device(DEVICE_NAME)
    checkpoint_path = Path(CHECKPOINT_PATH).expanduser().resolve()
    weather_csv = Path(WEATHER_FILE).expanduser().resolve()

    actor, normalizer, config = load_sac_policy(
        checkpoint_path=checkpoint_path,
        device=device,
        weather_csv=weather_csv,
    )

    output_root = Path(OUTPUT_DIR).expanduser()
    run_dir = output_root / RUN_TAG if RUN_TAG else output_root

    rewards, mask, df = run_sac_episode(
        actor=actor,
        normalizer=normalizer,
        config=config,
        device=device,
        verbose_steps=VERBOSE_STEPS,
    )

    save_artifacts(df=df, output_dir=run_dir, disable_plots=DISABLE_PLOTS)
    summary_path = summarize_run(
        rewards=rewards,
        mask=mask,
        df=df,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        device=device,
        setpoint=config.setpoint,
    )
    print(f"[SAC] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
