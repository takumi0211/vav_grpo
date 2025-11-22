#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve()
    _pkg_root = _here.parent
    while not (_pkg_root / "simulator_humid").exists() and _pkg_root.parent != _pkg_root:
        _pkg_root = _pkg_root.parent
    if (_pkg_root / "simulator_humid").exists() and str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    del _pkg_root, _here

import argparse
import json
import os
import pathlib
import pickle
import subprocess
import sys
import types
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

import simulator_humid.simulation as sim_module
from critic_value.common import critic_input_columns
from simulator_humid.agents.llm_agent import (
    build_default_zones as build_llm_zones,
    compute_llm_metrics,
)
from simulator_humid.agents.rl.training_td3 import (
    ObservationNormalizer,
    PolicyController,
    TD3Actor,
    TrainingConfig,
    _prepare_weather_profile,
    build_action_scaler,
    build_simulation_kwargs,
    compute_step_rewards as compute_td3_rewards,
)
from simulator_humid.simulation import create_plots, run_simulation
from simulator_humid.utils.paths import RL_OUTPUT_DIR, WEATHER_DATA_DIR

# ---------------------------------------------------------------------------
# クロスプラットフォーム互換性: Windowsで保存したcheckpointに含まれるWindowsPathを
# Mac/LinuxでロードするときにPosixPathとして復元できるようにする。
# pathlib.WindowsPathの__new__は非Windows環境でUnsupportedOperationを投げるため、
# 非Windows環境ではWindowsPathをPosixPathに差し替える。
# ---------------------------------------------------------------------------
if os.name != "nt":
    pathlib.WindowsPath = pathlib.PosixPath

# ---------------------------------------------------------------------------
# 実行パラメータ（=の右側を書き換えて使用）
# ・天気データ: weather_data/outdoor_temp_20250729.csv（LLM runと共通）
# ・人数データ: simulator_humid/simulation.py に定義された
#   DEFAULT_ZONE_OCCUPANCY_WEEKDAY / WEEKEND（LLM runと共通）
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = RL_OUTPUT_DIR / "td3_policy_final.pt"  # 学習済みTD3チェックポイントのパス
OUTPUT_DIR = RL_OUTPUT_DIR / "td3_eval"  # 評価結果を保存するディレクトリ
RUN_TAG = None  # サブフォルダ名（Noneの場合は現在時刻）
DEVICE_NAME = None  # PyTorchデバイス指定（例: "cuda", "cuda:0"。Noneで自動判定）
WEATHER_FILE = WEATHER_DATA_DIR / "outdoor_temp_20250726.csv"  # 気象条件CSV（LLMと同一）
VERBOSE_STEPS = False  # Trueにするとシミュレーション各ステップを出力
DISABLE_PLOTS = False  # Trueにするとプロット生成をスキップ
DECISIONS_DIR = Path("critic_value/data")
DATASET_OUTPUT = Path("critic_value/data/dataset.csv")
PER_DAY_DEFAULT = 120


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


def _date_from_weather(path: Path) -> str | None:
    """Extract YYYYMMDD token from weather filename like outdoor_temp_20250726.csv."""

    stem = path.stem
    for token in stem.split("_"):
        if token.isdigit() and len(token) == 8:
            return token
    return None


def load_td3_policy(
    checkpoint_path: Path,
    device: torch.device,
    weather_csv: Path,
) -> tuple[TD3Actor, ObservationNormalizer, TrainingConfig]:
    """TD3チェックポイントを読み込み、アクターと正規化器、コンフィグを復元する。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Windows環境で保存したチェックポイントには WindowsPath が含まれるため、
    # 非Windows環境でそのままtorch.loadするとUnsupportedOperationで落ちる。
    # カスタムUnpicklerで WindowsPath を PosixPath に差し替えてロードする。
    class _PathCompatibleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):  # noqa: N802
            if module.startswith("pathlib") and "WindowsPath" in name:
                return pathlib.PosixPath
            return super().find_class(module, name)

    pickle_module = types.ModuleType("pickle")
    pickle_module.Unpickler = _PathCompatibleUnpickler

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,  # TrainingConfigや正規化器の状態も復元するため信頼できる環境で実行
        pickle_module=pickle_module,
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
    actor = TD3Actor(obs_dim, action_dim, actor_hidden).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    normalizer = ObservationNormalizer(obs_dim, clip=config.obs_norm_clip, eps=config.obs_norm_eps)
    if "obs_normalizer" in checkpoint:
        normalizer.load_state_dict(checkpoint["obs_normalizer"])
    else:
        raise ValueError("Checkpoint missing 'obs_normalizer' statistics.")

    return actor, normalizer, config


def run_td3_episode(
    *,
    actor: TD3Actor,
    normalizer: ObservationNormalizer,
    config: TrainingConfig,
    device: torch.device,
    verbose_steps: bool,
) -> tuple[np.ndarray, np.ndarray, "pd.DataFrame", PolicyController]:
    """指定したTD3ポリシーでシミュレーションを1本実行し、報酬や結果を返す。"""
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

    rewards, mask = compute_td3_rewards(df, config)
    df["td3_reward"] = rewards
    return rewards, mask, df, controller


def save_artifacts(
    *,
    df,
    output_dir: Path,
    disable_plots: bool,
) -> tuple[Path, Optional[Path]]:
    """シミュレーション結果CSVとプロットを所定ディレクトリに保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "td3_simulation_results.csv"
    df.to_csv(results_path, index=True)

    plots_dir: Optional[Path] = None
    if not disable_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        create_plots(df, plots_dir / "td3_run.png", plots_dir / "td3_run_dampers.png")
    return results_path, plots_dir


def build_decision_dataframe(
    *,
    controller: PolicyController,
    config: TrainingConfig,
    episode_id: int,
    weather_token: str | None,
) -> pd.DataFrame:
    """PolicyControllerの軌跡からdecision_stepのみ抽出してDataFrame化する。"""

    traj = controller.trajectory()
    obs = np.asarray(traj.get("obs"), dtype=np.float32)
    acts = np.asarray(traj.get("tanh_actions"), dtype=np.float32)
    decision_mask = np.asarray(traj.get("decision_mask"), dtype=bool)
    timestamps: List[datetime] = traj.get("timestamps", [])  # type: ignore[assignment]

    if obs.shape[0] != decision_mask.shape[0] or acts.shape[0] != decision_mask.shape[0]:
        raise RuntimeError("trajectory shapes mismatch between obs/actions/mask")

    obs_dec = obs[decision_mask]
    acts_dec = acts[decision_mask]
    ts_dec = [timestamps[i] for i, flag in enumerate(decision_mask) if flag] if timestamps else []

    zone_count = len(config.zones)
    obs_cols, act_cols = critic_input_columns(config)

    obs_data: Dict[str, np.ndarray] = {}
    block = {
        "temp_error": 0,
        "temp_delta": zone_count,
        "co2_error": 2 * zone_count,
        "co2_delta": 3 * zone_count,
    }
    for idx in range(zone_count):
        obs_data[f"zone_{idx+1}_temp_error"] = obs_dec[:, block["temp_error"] + idx]
        obs_data[f"zone_{idx+1}_temp_delta"] = obs_dec[:, block["temp_delta"] + idx]
        obs_data[f"zone_{idx+1}_co2_error"] = obs_dec[:, block["co2_error"] + idx]
        obs_data[f"zone_{idx+1}_co2_delta"] = obs_dec[:, block["co2_delta"] + idx]

    global_offset = 4 * zone_count
    obs_data["outdoor_temp"] = obs_dec[:, global_offset]
    obs_data["outdoor_temp_slope"] = obs_dec[:, global_offset + 1]
    obs_data["sin_time"] = obs_dec[:, global_offset + 2]
    obs_data["cos_time"] = obs_dec[:, global_offset + 3]

    prev_offset = global_offset + 4
    for idx in range(zone_count):
        obs_data[f"prev_zone_{idx+1}_damper"] = obs_dec[:, prev_offset + idx]
    prev_offset += zone_count
    obs_data["prev_oa_damper"] = obs_dec[:, prev_offset]
    obs_data["prev_coil_valve"] = obs_dec[:, prev_offset + 1]
    obs_data["prev_fan_speed"] = obs_dec[:, prev_offset + 2]

    df_obs = pd.DataFrame(obs_data, columns=[c for c in obs_cols if c in obs_data])
    df_act = pd.DataFrame(acts_dec, columns=act_cols)
    df = pd.concat([df_obs, df_act], axis=1)
    df.insert(0, "step", np.arange(len(df), dtype=np.int32))
    df.insert(1, "decision_step", 1)
    if ts_dec:
        df.insert(2, "timestamp", pd.to_datetime(ts_dec))
    df.insert(0, "episode_id", episode_id)
    return df


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
        "results_csv": str((run_dir / "td3_simulation_results.csv").resolve()),
        "plots_dir": str((run_dir / "plots").resolve()) if (run_dir / "plots").exists() else None,
    }

    summary_path = run_dir / "td3_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"[TD3] total_reward={total_reward:.3f}, active_steps={active_steps}, "
        f"mean_reward_active={mean_reward:.4f}"
    )
    print("[TD3] LLM-aligned metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")

    return summary_path


def build_prompt_dataset(
    *,
    sources: List[Path],
    output_path: Path,
    per_day: int,
) -> None:
    """Invoke build_state_prompt_dataset.py to turn decision CSVs into prompts."""

    if not sources:
        raise ValueError("No decision CSVs provided for dataset build")
    script_path = Path(__file__).resolve().with_name("build_state_prompt_dataset.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--per-day",
        str(per_day),
        "--output",
        str(output_path),
        "--sources",
    ]
    cmd.extend(str(path) for path in sources)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TD3 policy, log decisions, and rebuild dataset.csv")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="TD3 checkpoint path")
    parser.add_argument(
        "--weather",
        type=Path,
        nargs="+",
        default=[WEATHER_FILE],
        help="Weather CSV file(s) to simulate",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to run per weather file")
    parser.add_argument("--device", type=str, default=DEVICE_NAME, help="Torch device override (e.g. cuda:0)")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DIR, help="Directory for TD3 eval artifacts")
    parser.add_argument("--run-tag", type=str, default=RUN_TAG, help="Optional subfolder under output root")
    parser.add_argument("--verbose-steps", action="store_true", default=VERBOSE_STEPS)
    parser.add_argument("--disable-plots", action="store_true", default=DISABLE_PLOTS)
    parser.add_argument(
        "--decisions-dir",
        type=Path,
        default=DECISIONS_DIR,
        help="Directory to dump decision_step CSVs",
    )
    parser.add_argument(
        "--skip-decisions",
        action="store_true",
        help="Skip writing decision CSVs",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=DATASET_OUTPUT,
        help="Final dataset.csv path (state_json + prompt)",
    )
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset rebuild")
    parser.add_argument(
        "--per-day",
        type=int,
        default=PER_DAY_DEFAULT,
        help="Samples per source passed to build_state_prompt_dataset",
    )
    return parser.parse_args()


def main() -> None:
    """TD3エージェントを指定した天気で走らせ、decision/datasetを更新する。"""

    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    decisions_root = Path(args.decisions_dir).expanduser().resolve()
    dataset_output = Path(args.dataset_output).expanduser().resolve()
    single_weather = len(args.weather) == 1
    multi_weather = len(args.weather) > 1

    decision_frames: Dict[str, List[pd.DataFrame]] = {}
    all_frames: List[pd.DataFrame] = []
    episode_counter = 0

    for weather_path in args.weather:
        weather_path = weather_path.expanduser().resolve()
        weather_token = _date_from_weather(weather_path) or weather_path.stem

        actor, normalizer, config = load_td3_policy(
            checkpoint_path=checkpoint_path,
            device=device,
            weather_csv=weather_path,
        )

        base_dir = output_root / args.run_tag if args.run_tag else output_root
        if multi_weather or args.episodes > 1 or (not args.run_tag and not single_weather):
            base_dir = base_dir / weather_token

        for ep_idx in range(args.episodes):
            run_dir = base_dir / f"ep{ep_idx:02d}" if args.episodes > 1 else base_dir
            run_dir.mkdir(parents=True, exist_ok=True)

            rewards, mask, df, controller = run_td3_episode(
                actor=actor,
                normalizer=normalizer,
                config=config,
                device=device,
                verbose_steps=args.verbose_steps,
            )

            save_artifacts(df=df, output_dir=run_dir, disable_plots=args.disable_plots)
            summary_path = summarize_run(
                rewards=rewards,
                mask=mask,
                df=df,
                checkpoint_path=checkpoint_path,
                run_dir=run_dir,
                device=device,
                setpoint=config.setpoint,
            )
            print(f"[TD3] Summary saved to {summary_path}")

            decision_df = build_decision_dataframe(
                controller=controller,
                config=config,
                episode_id=episode_counter,
                weather_token=weather_token,
            )
            decision_frames.setdefault(weather_token, []).append(decision_df)
            all_frames.append(decision_df)
            episode_counter += 1

    decision_sources: List[Path] = []
    if not args.skip_decisions and decision_frames:
        decisions_root.mkdir(parents=True, exist_ok=True)
        for token in sorted(decision_frames.keys()):
            df = pd.concat(decision_frames[token], ignore_index=True)
            dest = decisions_root / f"decisions_{token}.csv"
            df.to_csv(dest, index=False)
            decision_sources.append(dest)
            print(f"[TD3] Wrote {len(df)} decision rows to {dest}")

        if len(all_frames) > 1:
            all_path = decisions_root / "decisions_all.csv"
            pd.concat(all_frames, ignore_index=True).to_csv(all_path, index=False)
            print(f"[TD3] Wrote aggregated decisions to {all_path}")

    if not args.skip_dataset and decision_sources:
        build_prompt_dataset(sources=decision_sources, output_path=dataset_output, per_day=args.per_day)



if __name__ == "__main__":
    main()
