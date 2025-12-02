# 必要なライブラリをインポートします。
# os: ファイルシステム操作用
# random: ランダムサンプリング用
# re: 正規表現マッチング用
# glob: ファイルパターンマッチング用
# datasets: Hugging Faceのデータセットライブラリ用
import os
import random
import re
import csv
import atexit
import math
import json
import numpy as np
from collections.abc import Iterable as IterableSequence
from glob import glob
from datasets import load_dataset, Dataset
from typing import Any, List, Optional, Sequence, Union

TRUNCATION_TOKEN_THRESHOLD = 4000

# アクションを抽出するための正規表現パターン
# 形式: [0], [1], [2], [3] を文中から検出
ACTION_RE = re.compile(r"\[(\d)\]")

# 無効なアクションや判定不能な completion を学習から除外するための値
PENALTY = math.nan


def _is_main_process() -> bool:
    rank = os.environ.get("RANK")
    if rank not in (None, "", "0"):
        return False
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank not in (None, "", "0"):
        return False
    return True


CSV_LOG_ENABLED = os.getenv("GRPO_LOG_COMPLETIONS", "1") != "0" and _is_main_process()
CSV_PATH = os.getenv(
    "GRPO_COMPLETION_LOG_PATH",
    os.path.join("output", "micro_step_completions.csv"),
)
_CSV_FILE = None
_CSV_WRITER = None
_MICRO_STEP_COUNTER = 0
CSV_COLUMNS = [
    "step",
    "micro_step",
    "sample_id",
    "prompt",
    "completion",
    "action",
    "reward",
    "target_action",
    "q1",
    "q_mean",
    "error",
    "selected_action",
    "q1_ideal",
]
_TD3_MODEL_CACHE = None


def _steps_per_generation() -> int:
    raw = os.getenv("GRPO_STEPS_PER_GENERATION") or os.getenv("GRADIENT_ACCUMULATION_STEPS")
    try:
        return max(1, int(raw)) if raw is not None else 1
    except ValueError:
        return 1


def _init_csv_logger() -> None:
    global _CSV_FILE, _CSV_WRITER
    if not CSV_LOG_ENABLED or _CSV_WRITER is not None:
        return
    os.makedirs(os.path.dirname(CSV_PATH) or ".", exist_ok=True)
    _CSV_FILE = open(CSV_PATH, "w", newline="", encoding="utf-8")
    _CSV_WRITER = csv.writer(_CSV_FILE)
    _CSV_WRITER.writerow(CSV_COLUMNS)


def _close_csv_logger() -> None:
    global _CSV_FILE, _CSV_WRITER
    if _CSV_FILE is not None:
        try:
            _CSV_FILE.flush()
            _CSV_FILE.close()
        except ValueError:
            # Already closed; safe to ignore
            pass
    _CSV_FILE = None
    _CSV_WRITER = None


atexit.register(_close_csv_logger)

# プロンプトデータセットを読み込む関数
def load_prompt_dataset(data_dir: str = "data", harmony_only: bool = True) -> Dataset:
    """Load all CSV prompts (defaults to Harmony-formatted files)."""

    preferred = os.path.join(data_dir, "grpo_dataset.csv")
    if os.path.isfile(preferred):
        return load_dataset("csv", data_files=[preferred], split="train")

    def _list(pattern):
        return sorted(glob(os.path.join(data_dir, pattern)))

    files = _list("*_harmony.csv") if harmony_only else _list("*.csv")
    if not files and harmony_only:
        # Harmony変換前の環境でも動くようにフォールバック
        files = _list("*.csv")
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return load_dataset("csv", data_files=files, split="train")


# データセットからランダムにk個のサンプルを抽出する関数
# dataset: 入力データセット
# k: 抽出するサンプル数
# 戻り値: サンプリングされたDatasetオブジェクト
def sample_batch(dataset: Dataset, k: int) -> Dataset:
    # データセットのインデックスからランダムにk個を選択
    idx = random.sample(range(len(dataset)), k)
    # 選択されたインデックスに基づいてサブセットを作成
    return dataset.select(idx)


# 値を指定サイズまで拡張するヘルパー関数 (内部使用)
# values: 拡張元の値リスト
# size: 目標サイズ
# 戻り値: 拡張されたリスト (繰り返しや追加でサイズを合わせる)
def _expand(values, size):
    seq = list(values)
    if not seq or size == 0:
        return [0.0] * size
    if len(seq) == size:
        return seq
    repeat = max(1, size // len(seq))
    expanded = []
    for val in seq:
        expanded.extend([val] * repeat)
    while len(expanded) < size:
        expanded.extend(seq)
    return expanded[:size]


def _argmax_action(values) -> str:
    best_idx = None
    best_val = float("-inf")
    for idx, val in enumerate(values):
        try:
            val_float = float(val)
        except (TypeError, ValueError):
            continue
        if math.isnan(val_float):
            continue
        if best_idx is None or val_float > best_val:
            best_idx = idx
            best_val = val_float
    return str(best_idx) if best_idx is not None else "NaN"


# 報酬計算のメイン関数
# completions: 完了したアクションのリスト (例: "[0]", "[1]" など)
# reward_action_0 ~ reward_action_3: 各アクション(0-3)に対する報酬値 (スカラまたはリスト)
# **kwargs: 追加のキーワード引数 (未使用)
# 戻り値: 各completionに対する報酬のリスト
def _to_list(value: Optional[Union[Sequence, "torch.Tensor"]]) -> list:
    """Convert tensors/sequences to a plain list without importing torch globally."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())
        except TypeError:
            return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _broadcast_any(value: Any, size: int) -> list:
    """Broadcast scalars/short sequences to match the completions length."""

    if size <= 0:
        return []
    if value is None:
        return [None] * size
    if isinstance(value, (str, bytes)):
        return [value] * size
    try:
        seq = list(value)
    except TypeError:
        return [value] * size
    if not seq:
        return [None] * size
    if len(seq) == size:
        return seq
    if len(seq) == 1:
        return seq * size
    # Fallback: repeat to cover the requested size
    repeated = (seq * math.ceil(size / len(seq)))[:size]
    return repeated


def _to_scalar(val: Any):
    """Best-effort conversion to a float for CSV logging."""

    try:
        if hasattr(val, "item"):
            return val.item()
    except Exception:
        pass
    if isinstance(val, (list, tuple)) and len(val) == 1:
        return _to_scalar(val[0])
    try:
        return float(val)
    except Exception:
        return val


def _legacy_reward_fn(
    completions,
    reward_action_0,
    reward_action_1,
    reward_action_2,
    reward_action_3,
    **kwargs,
):
    global _MICRO_STEP_COUNTER

    size = len(completions)
    trainer_state = kwargs.get("trainer_state")
    rewards_0 = _expand(reward_action_0, size)
    rewards_1 = _expand(reward_action_1, size)
    rewards_2 = _expand(reward_action_2, size)
    rewards_3 = _expand(reward_action_3, size)

    completion_tokens_raw = kwargs.get("completion_ids")
    completion_masks_raw = kwargs.get("completion_mask")
    token_sequences = []
    if isinstance(completion_tokens_raw, IterableSequence) and not isinstance(
        completion_tokens_raw, (str, bytes)
    ):
        token_sequences = [_to_list(tokens) for tokens in completion_tokens_raw]
    mask_sequences = []
    if isinstance(completion_masks_raw, IterableSequence) and not isinstance(
        completion_masks_raw, (str, bytes)
    ):
        mask_sequences = [_to_list(mask) for mask in completion_masks_raw]
    if mask_sequences and not token_sequences:
        mask_sequences = []
    if token_sequences and not mask_sequences:
        mask_sequences = [[] for _ in token_sequences]
    rewards = []
    action_tokens = []
    target_actions = []
    steps_per_generation = _steps_per_generation()
    completions_per_micro = max(1, math.ceil(size / steps_per_generation))
    for idx, (completion, r0, r1, r2, r3) in enumerate(
        zip(completions, rewards_0, rewards_1, rewards_2, rewards_3)
    ):
        reward_map = (r0, r1, r2, r3)
        target_action = _argmax_action(reward_map)
        is_truncated = False
        if token_sequences and idx < len(token_sequences):
            tokens_seq = token_sequences[idx]
            mask_seq = mask_sequences[idx] if idx < len(mask_sequences) else []
            seq_len = len(tokens_seq)
            mask_vals = [int(v) for v in mask_seq] if mask_seq else []
            mask_all_zero = bool(mask_vals) and all(v == 0 for v in mask_vals)
            threshold_hit = (
                TRUNCATION_TOKEN_THRESHOLD > 0
                and seq_len >= TRUNCATION_TOKEN_THRESHOLD
            )
            if mask_all_zero or threshold_hit:
                is_truncated = True
        if is_truncated:
            rewards.append(math.nan)
            action_tokens.append("NaN")
            target_actions.append(target_action)
            continue

        matches = list(ACTION_RE.finditer(completion))
        if not matches:
            rewards.append(PENALTY)
            action_tokens.append("NaN")
            target_actions.append(target_action)
            continue
        action = matches[-1].group(1)
        idx_int = int(action)
        reward_val = float(reward_map[idx_int]) if 0 <= idx_int < 4 else PENALTY
        rewards.append(reward_val)
        action_tokens.append(action if 0 <= idx_int < 4 else "NaN")
        target_actions.append(target_action)

    if CSV_LOG_ENABLED and size:
        _init_csv_logger()
        if _CSV_WRITER is not None:
            prompts = kwargs.get("prompt") or kwargs.get("prompts") or []
            if isinstance(prompts, str):
                prompts = [prompts]
            sample_ids = _broadcast_any(kwargs.get("sample_id"), size)
            selected_actions = _broadcast_any(kwargs.get("selected_action"), size)
            q1_ideal_values = _broadcast_any(kwargs.get("q1") or kwargs.get("q1_ideal"), size)
            global_step = getattr(trainer_state, "global_step", -1)
            step_value = global_step if (global_step is not None and global_step >= 0) else _MICRO_STEP_COUNTER
            for idx, (completion, reward_val, action_token, target_action) in enumerate(
                zip(completions, rewards, action_tokens, target_actions)
            ):
                prompt_text = ""
                if prompts:
                    prompt_text = prompts[idx % len(prompts)]
                micro_step_value = min(
                    (idx // completions_per_micro) + 1,
                    steps_per_generation,
                )
                reward_csv = reward_val
                if isinstance(reward_val, float) and math.isnan(reward_val):
                    reward_csv = "NaN"
                q1_ideal_val = q1_ideal_values[idx] if idx < len(q1_ideal_values) else None
                q1_ideal_csv = _to_scalar(q1_ideal_val)
                if q1_ideal_csv is None:
                    q1_ideal_csv = ""
                sample_id_val = sample_ids[idx] if idx < len(sample_ids) else None
                sample_id_csv = _to_scalar(sample_id_val)
                if sample_id_csv is None:
                    sample_id_csv = ""
                selected_action_val = selected_actions[idx] if idx < len(selected_actions) else None
                selected_action_csv = selected_action_val if selected_action_val is not None else ""
                _CSV_WRITER.writerow(
                    [
                        step_value,
                        micro_step_value,
                        sample_id_csv,
                        prompt_text,
                        completion,
                        action_token,
                        reward_csv,
                        target_action,
                        "",  # q1 placeholder for legacy mode
                        "",  # q_mean placeholder for legacy mode
                        "",  # error placeholder for legacy mode
                        selected_action_csv,
                        q1_ideal_csv,
                    ]
                )
            if _CSV_FILE is not None:
                _CSV_FILE.flush()
        _MICRO_STEP_COUNTER += 1

    return rewards


def td3_reward_fn(
    completions,
    state_json=None,
    state_raw_json=None,
    state_json_normalize=None,
    **kwargs,
):
    """Reward function that scores actions with the TD3 critic on-the-fly."""

    global _MICRO_STEP_COUNTER, _TD3_MODEL_CACHE

    size = len(completions)
    if size == 0:
        return []

    # Lazy import to avoid pulling simulator dependencies unless needed.
    from train_support.td3_reward import ParsedSample, softmax_from_scores, td3_model_from_env

    trainer_state = kwargs.get("trainer_state")

    # Completion truncation info (same handling as legacy path)
    completion_tokens_raw = kwargs.get("completion_ids")
    completion_masks_raw = kwargs.get("completion_mask")
    token_sequences = []
    if isinstance(completion_tokens_raw, IterableSequence) and not isinstance(
        completion_tokens_raw, (str, bytes)
    ):
        token_sequences = [_to_list(tokens) for tokens in completion_tokens_raw]
    mask_sequences = []
    if isinstance(completion_masks_raw, IterableSequence) and not isinstance(
        completion_masks_raw, (str, bytes)
    ):
        mask_sequences = [_to_list(mask) for mask in completion_masks_raw]
    if mask_sequences and not token_sequences:
        mask_sequences = []
    if token_sequences and not mask_sequences:
        mask_sequences = [[] for _ in token_sequences]

    # Prefer full-precision state if present; otherwise fall back to the normalized snapshot.
    states_primary = _broadcast_any(state_json, size)
    states_fallback = _broadcast_any(state_json_normalize, size)
    states_raw = _broadcast_any(state_raw_json, size)
    prompts = kwargs.get("prompt") or kwargs.get("prompts") or []
    if isinstance(prompts, str):
        prompts = [prompts]

    if _TD3_MODEL_CACHE is None:
        _TD3_MODEL_CACHE = td3_model_from_env()
    model = _TD3_MODEL_CACHE
    scaler = model.scaler

    temperature = float(os.getenv("TD3_SOFTMAX_TEMP", os.getenv("TD3_SOFTMAX_TAU", "0.5")))

    q1_values: List[float] = [math.nan] * size
    q_mean_values: List[float] = [math.nan] * size
    rewards: List[float] = [math.nan] * size
    action_summaries: List[str] = [""] * size
    errors: List[str] = [""] * size

    valid_samples: List[ParsedSample] = []
    valid_indices: List[int] = []

    steps_per_generation = _steps_per_generation()
    completions_per_micro = max(1, math.ceil(size / steps_per_generation))

    for idx, completion in enumerate(completions):
        is_truncated = False
        if token_sequences and idx < len(token_sequences):
            tokens_seq = token_sequences[idx]
            mask_seq = mask_sequences[idx] if idx < len(mask_sequences) else []
            seq_len = len(tokens_seq)
            mask_vals = [int(v) for v in mask_seq] if mask_seq else []
            mask_all_zero = bool(mask_vals) and all(v == 0 for v in mask_vals)
            threshold_hit = (
                TRUNCATION_TOKEN_THRESHOLD > 0
                and seq_len >= TRUNCATION_TOKEN_THRESHOLD
            )
            if mask_all_zero or threshold_hit:
                is_truncated = True
        if is_truncated:
            errors[idx] = "truncated"
            continue

        try:
            primary = states_primary[idx] if idx < len(states_primary) else None
            fallback = states_fallback[idx] if idx < len(states_fallback) else None
            obs_vec = model._resolve_obs(primary if primary is not None else fallback, states_raw[idx])
        except Exception as exc:  # pragma: no cover - defensive
            errors[idx] = str(exc)
            continue

        try:
            action_vec, payload = model.parse_action_from_text(completion)
            action_summaries[idx] = json.dumps(payload, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - defensive
            errors[idx] = str(exc)
            continue

        valid_samples.append(ParsedSample(obs_vec=obs_vec, action_vec=action_vec, action_payload=payload))
        valid_indices.append(idx)

    if valid_samples:
        scored = model.score_batch(valid_samples, assume_obs_normalized=False)
        for scored_sample, idx in zip(scored, valid_indices):
            if scored_sample.q_min is not None:
                q1_values[idx] = float(scored_sample.q_min)  # q_min stores q1 in TD3RewardModel
            if scored_sample.q_mean is not None:
                q_mean_values[idx] = float(scored_sample.q_mean)

    # Compute softmax per micro-step so each micro batch sums to ~1.
    for micro_idx in range(steps_per_generation):
        start = micro_idx * completions_per_micro
        end = min(start + completions_per_micro, size)
        if start >= end:
            continue
        segment = q1_values[start:end]
        softmax = softmax_from_scores(segment, temperature)
        for offset, val in enumerate(softmax):
            idx = start + offset
            rewards[idx] = float(val) if isinstance(val, (float, np.floating)) and math.isfinite(val) else math.nan

    if CSV_LOG_ENABLED and size:
        _init_csv_logger()
        if _CSV_WRITER is not None:
            global_step = getattr(trainer_state, "global_step", -1)
            step_value = global_step if (global_step is not None and global_step >= 0) else _MICRO_STEP_COUNTER
            sample_ids = _broadcast_any(kwargs.get("sample_id"), size)
            selected_actions = _broadcast_any(kwargs.get("selected_action"), size)
            q1_ideal_values = _broadcast_any(kwargs.get("q1") or kwargs.get("q1_ideal"), size)
            for idx, completion in enumerate(completions):
                prompt_text = prompts[idx % len(prompts)] if prompts else ""
                micro_step_value = min(
                    (idx // completions_per_micro) + 1,
                    steps_per_generation,
                )
                reward_csv = rewards[idx]
                if isinstance(reward_csv, float) and math.isnan(reward_csv):
                    reward_csv = "NaN"
                q1_val = q1_values[idx]
                q1_csv = _to_scalar(q1_val)
                if isinstance(q1_val, float) and math.isnan(q1_val):
                    q1_csv = "NaN"
                q1_ideal_val = q1_ideal_values[idx] if idx < len(q1_ideal_values) else None
                q1_ideal_csv = _to_scalar(q1_ideal_val)
                if q1_ideal_csv is None:
                    q1_ideal_csv = ""
                sample_id_val = sample_ids[idx] if idx < len(sample_ids) else None
                sample_id_csv = _to_scalar(sample_id_val)
                if sample_id_csv is None:
                    sample_id_csv = ""
                selected_action_val = selected_actions[idx] if idx < len(selected_actions) else None
                selected_action_csv = (
                    json.dumps(selected_action_val, ensure_ascii=False)
                    if isinstance(selected_action_val, (dict, list))
                    else (selected_action_val if selected_action_val is not None else "")
                )
                action_payload = action_summaries[idx] if action_summaries[idx] else ""
                _CSV_WRITER.writerow(
                    [
                        step_value,
                        micro_step_value,
                        sample_id_csv,
                        prompt_text,
                        completion,
                        json.dumps(action_payload, ensure_ascii=False) if action_payload else "",
                        reward_csv,
                        "",  # target_action not used in TD3 mode
                        q1_csv,
                        q_mean_values[idx],
                        errors[idx],
                        selected_action_csv,
                        q1_ideal_csv,
                    ]
                )
            if _CSV_FILE is not None:
                _CSV_FILE.flush()
        _MICRO_STEP_COUNTER += 1

    return rewards


def reward_fn(
    completions,
    reward_action_0=None,
    reward_action_1=None,
    reward_action_2=None,
    reward_action_3=None,
    state_json=None,
    state_raw_json=None,
    state_json_normalize=None,
    **kwargs,
):
    """Dispatch to TD3-based reward when state vectors are present."""

    has_state = any(v is not None for v in (state_json, state_raw_json, state_json_normalize))
    td3_env_flag = os.getenv("GRPO_USE_TD3_REWARD", "1") != "0"
    if has_state and td3_env_flag:
        return td3_reward_fn(
            completions,
            state_json=state_json,
            state_raw_json=state_raw_json,
            state_json_normalize=state_json_normalize,
            **kwargs,
        )
    return _legacy_reward_fn(
        completions,
        reward_action_0,
        reward_action_1,
        reward_action_2,
        reward_action_3,
        **kwargs,
    )
