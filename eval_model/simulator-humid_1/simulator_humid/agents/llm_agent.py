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
# 標準/外部ライブラリを読み込み、LLMベースのHVAC制御エージェントを構築
# ---------------------------------------------------------------------------
import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# システムシミュレータ側のユーティリティをインポート
# ---------------------------------------------------------------------------
from simulator_humid.simulation import (
    HVACActions,
    ZoneConfig,
    internal_gain_profile_from_occupancy,
    create_plots,
    outdoor_temperature,
    run_simulation,
)
from simulator_humid.utils.paths import LLM_OUTPUT_DIR

# ---------------------------------------------------------------------------
# LLM制御設定コンテナ
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    # === 報酬設計パラメータ ===
    temp_sigma: float = 0.25  # 温度快適性の標準偏差
    comfort_center: float = 26.0  # 快適温度の中心
    comfort_low: float = 25.0  # 快適温度下限
    comfort_high: float = 27.0  # 快適温度上限
    comfort_penalty: float = 2.0  # 快適性違反ペナルティ
    power_weight: float = 0.2  # エネルギー消費ペナルティ重み
    co2_target_ppm: float = 1000.0  # CO₂快適基準
    co2_penalty_weight: float = 0.5  # CO₂ペナルティ重み
    co2_logistic_k: float = 12.0  # ロジスティック関数の傾き係数
    co2_violation_threshold: float = 1100.0  # CO₂違反閾値
    co2_violation_penalty: float = 10.0  # CO₂違反ペナルティ重み

    # === シミュレーション設定 ===
    timestep_s: int = 60  # タイムステップ（秒）
    episode_minutes: int = 24 * 60  # エピソード長（分）
    start_time: datetime = datetime(2025, 7, 29, 0, 0)  # 開始時刻
    setpoint: float = 26.0  # 温度設定値

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

    # === 出力・ログ設定 ===
    output_dir: Path = LLM_OUTPUT_DIR  # 出力ディレクトリ
    zones: Sequence[ZoneConfig] = ()  # ゾーン設定



@dataclass
class ActionScaler:
    """アクチュエータの範囲を定義する補助クラス。"""
    
    low: torch.Tensor
    high: torch.Tensor

    def midpoint(self) -> np.ndarray:
        return (self.low.cpu().numpy() + self.high.cpu().numpy()) * 0.5

# ---------------------------------------------------------------------------
# LLMベースの意思決定エージェント
# ---------------------------------------------------------------------------

@dataclass
class LLMDecisionResult:
    action_vector: np.ndarray
    thought_process: str
    action_payload: dict[str, Any]
    raw_response_text: str
    error: str | None = None


class LLMDecisionController:

    def __init__(
        self,
        *,
        client: OpenAI,
        config: LLMConfig,
        scaler: ActionScaler,
        log_path: Path,
        model: str,
    ) -> None:
        self.client = client
        self.config = config
        self.scaler = scaler
        self.model = model
        self.zone_count = len(config.zones)
        if self.zone_count == 0:
            raise ValueError("LLMDecisionController requires at least one zone configuration.")
        self.zone_names = [zone.name for zone in config.zones]
        self.control_period_s = max(1, int(config.control_interval_minutes * 60))
        self.low_bounds = self.scaler.low.detach().cpu().numpy().astype(np.float32)
        self.high_bounds = self.scaler.high.detach().cpu().numpy().astype(np.float32)
        self.prev_action = np.asarray(self.scaler.midpoint(), dtype=np.float32)
        self.idle_action = self.low_bounds.copy()
        self.prev_zone_temps: np.ndarray | None = None
        self.prev_zone_co2: np.ndarray | None = None
        self.prev_outdoor_temp: float | None = None
        self.prev_env_timestamp: datetime | None = None
        self.last_decision_time: datetime | None = None
        self._has_responses_api = bool(
            getattr(self.client, "responses", None)
            and hasattr(self.client.responses, "create")
        )
        self._has_chat_completions_api = bool(
            getattr(self.client, "chat", None)
            and getattr(self.client.chat, "completions", None)
            and hasattr(self.client.chat.completions, "create")
        )
        self.action_bounds = {
            "zone_dampers": [float(config.damper_min), float(config.damper_max)],
            "oa_damper": [float(config.oa_min), float(config.oa_max)],
            "coil_valve": [float(config.coil_min), float(config.coil_max)],
            "fan_speed": [float(config.fan_min), float(config.fan_max)],
        }
        self.reward_design = {
            "comfort_center_c": float(config.comfort_center),
            "comfort_band_c": [float(config.comfort_low), float(config.comfort_high)],
            "comfort_sigma": float(config.temp_sigma),
            "comfort_penalty": float(config.comfort_penalty),
            "power_weight": float(config.power_weight),
            "co2_target_ppm": float(config.co2_target_ppm),
            "co2_penalty_weight": float(config.co2_penalty_weight),
            "co2_violation_threshold": float(config.co2_violation_threshold),
            "co2_violation_penalty": float(config.co2_violation_penalty),
        }
        zone_list = ", ".join(self.zone_names)
        self.system_prompt = (
            "You are GPT-5 acting as a supervisory decision-making agent for a multi-zone VAV HVAC system. "
            f"The controlled zones are: {zone_list}. "
            "You must select actuator setpoints every 5 minutes while the HVAC plant is running. "
            "Your objective is to maximise the reward used by the existing RL agent: reward = comfort_reward "
            "- comfort_penalty * band_penalty - power_weight * power_kw - co2_penalty_weight * co2_penalty "
            "- co2_violation_penalty (applied when peak CO2 crosses the violation threshold). "
            "Maintain thermal comfort around 26 degC (comfort band 25-27 degC), keep CO2 below 1000 ppm, "
            "and minimise fan/chiller/pump energy. Always respect the actuator limits provided and plan for the "
            "next 5-minute interval."
        )
        self.developer_prompt = (
            "Formatting rules: respond with a single JSON object. "
            "Use keys 'thought_process' (string) and 'action' (object). "
            "The 'action' object must include 'zone_dampers' (list of length {length}), 'oa_damper', "
            "'coil_valve', and 'fan_speed'. "
            "Provide the thought_process as explicit step-by-step reasoning. "
            "Return numeric values as floats; clip them to the provided bounds before finalising."
        ).format(length=self.zone_count)
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(
                    [
                        "timestamp_iso",
                        "hvac_on",
                        "prompt",
                        "thought_process",
                        "applied_action_json",
                        "raw_response_text",
                        "error",
                    ]
                )

    def __call__(
        self,
        timestamp: datetime,
        zone_temps: np.ndarray,
        zone_co2: np.ndarray,
        zone_rh: np.ndarray,
    ) -> HVACActions:
        hvac_on = self._is_hvac_on(timestamp)
        state_summary = self._build_state_summary(timestamp, zone_temps, zone_co2, zone_rh, hvac_on)
        if not hvac_on:
            self.prev_action = self.idle_action.copy()
            self.last_decision_time = None
            return self._vector_to_actions(self.prev_action)
        if not self._should_query(timestamp):
            return self._vector_to_actions(self.prev_action)
        prompt = self._build_prompt(state_summary)
        
        # LLM呼び出し直前に状態をプリント
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
        zone_state = "; ".join(
            f"{self.zone_names[idx]}: {zone_temps[idx]:.2f}C/{zone_co2[idx]:.0f}ppm"
            for idx in range(self.zone_count)
        )
        # 前回のダンパー開度を取得
        damper_state = "; ".join(
            f"{self.zone_names[idx]}: {self.prev_action[idx] * 100.0:.1f}%"
            for idx in range(self.zone_count)
        )
        print(f"[{timestamp_str}] zone state -> {zone_state}")
        print(f"    dampers -> {damper_state}")
        
        result = self._query_llm_with_retries(prompt)
        self.last_decision_time = timestamp
        if result.error is not None:
            print(f"[LLMDecisionController] {result.error}")
        self.prev_action = result.action_vector.copy()
        
        # LLMが実行したアクションを表示
        new_damper_state = "; ".join(
            f"{self.zone_names[idx]}: {self.prev_action[idx] * 100.0:.1f}%"
            for idx in range(self.zone_count)
        )
        oa = self.prev_action[self.zone_count]
        coil = self.prev_action[self.zone_count + 1]
        fan = self.prev_action[self.zone_count + 2]
        print(f"    LLM action -> dampers: [{new_damper_state}]; OA: {oa*100:.1f}%; Coil: {coil*100:.1f}%; Fan: {fan:.2f}")
        
        self._log_action(timestamp, hvac_on, prompt, result)
        return self._vector_to_actions(self.prev_action)

    def _is_hvac_on(self, timestamp: datetime) -> bool:
        current_hour = timestamp.hour + timestamp.minute / 60.0
        return self.config.hvac_start_hour <= current_hour < self.config.hvac_stop_hour

    def _should_query(self, timestamp: datetime) -> bool:
        if self.last_decision_time is None:
            return True
        delta = (timestamp - self.last_decision_time).total_seconds()
        return delta >= self.control_period_s - 1e-9

    def _build_state_summary(
        self,
        timestamp: datetime,
        zone_temps: np.ndarray,
        zone_co2: np.ndarray,
        zone_rh: np.ndarray,
        hvac_on: bool,
    ) -> dict[str, Any]:
        temps = np.asarray(zone_temps, dtype=np.float32)
        co2 = np.asarray(zone_co2, dtype=np.float32)
        if (
            temps.shape[0] != self.zone_count
            or co2.shape[0] != self.zone_count
        ):
            raise ValueError("Observation dimensions do not match zone count.")
        dt_minutes = self.config.timestep_s / 60.0
        if self.prev_env_timestamp is not None:
            dt_minutes = max((timestamp - self.prev_env_timestamp).total_seconds() / 60.0, 1e-6)
        temp_delta = np.zeros_like(temps)
        co2_delta = np.zeros_like(co2)
        if self.prev_zone_temps is not None:
            temp_delta = (temps - self.prev_zone_temps) / dt_minutes
        if self.prev_zone_co2 is not None:
            co2_delta = (co2 - self.prev_zone_co2) / dt_minutes
        outdoor_temp = float(outdoor_temperature(timestamp))
        temp_trend = 0.0
        if self.prev_outdoor_temp is not None and self.prev_env_timestamp is not None:
            dt_hours = max((timestamp - self.prev_env_timestamp).total_seconds() / 3600.0, 1e-6)
            temp_trend = (outdoor_temp - self.prev_outdoor_temp) / dt_hours
        zone_summary: List[dict[str, Any]] = []
        for idx, name in enumerate(self.zone_names):
            zone_summary.append(
                {
                    "name": name,
                    "temperature_c": round(float(temps[idx]), 1),
                    "temp_error_c": round(float(temps[idx] - self.config.setpoint), 1),
                    "temp_delta_c_per_min": round(float(temp_delta[idx]), 2),
                    "co2_ppm": round(float(co2[idx]), 0),
                    "co2_delta_ppm_per_min": round(float(co2_delta[idx]), 1),
                }
            )
        self.prev_zone_temps = temps.copy()
        self.prev_zone_co2 = co2.copy()
        self.prev_outdoor_temp = outdoor_temp
        self.prev_env_timestamp = timestamp
        return {
            "timestamp": timestamp.isoformat(),
            "setpoint_c": float(self.config.setpoint),
            "control_interval_minutes": float(self.config.control_interval_minutes),
            "time_features": {
                "hour": timestamp.hour,
                "minute": timestamp.minute,
                "second": timestamp.second,
            },
            "zones": zone_summary,
            "outdoor": {
                "temperature_c": round(outdoor_temp, 1),
                "temperature_trend_c_per_hour": round(temp_trend, 2),
            },
            "previous_action": self._vector_to_dict(self.prev_action),
            "action_bounds": self.action_bounds,
            "reward_design": self.reward_design,
        }

    def _build_prompt(self, state: dict[str, Any]) -> str:
        observation_json = json.dumps(state, ensure_ascii=False, indent=2)
        return (
            "Observation for the next 5-minute HVAC control interval:\n"
            f"{observation_json}\n"
            "Select actuator commands that maximise the stated reward over the upcoming interval."
        )

    def _build_chat_messages(self, prompt: str) -> list[dict[str, str]]:
        developer_instruction = f"Developer instructions: {self.developer_prompt}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": developer_instruction},
            {"role": "user", "content": prompt},
        ]

    def _build_instructions(self) -> str:
        return f"{self.system_prompt}\n\n{self.developer_prompt}"

    def _query_llm_with_retries(self, prompt: str) -> LLMDecisionResult:
        attempt = 0
        while True:
            attempt += 1
            result = self._query_llm(prompt)
            if not self._is_output_format_error(result):
                if attempt > 1 and result.error is None:
                    print(
                        f"[LLMDecisionController] Output format resolved after {attempt} attempts."
                    )
                return result
            print(
                f"[LLMDecisionController] Output format error (attempt {attempt}): {result.error}."
            )

    def _is_output_format_error(self, result: LLMDecisionResult) -> bool:
        if result.error is None:
            return False
        message = result.error.lower()
        if "llm request failed" in message:
            return False
        return True

    def _query_llm(self, prompt: str) -> LLMDecisionResult:
        try:
            if self._has_responses_api:
                instructions = self._build_instructions()
                response = self.client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=prompt,
                )
            elif self._has_chat_completions_api:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._build_chat_messages(prompt),
                )
            else:
                raise RuntimeError(
                    "OpenAI client does not expose responses or chat.completions APIs"
                )
        except Exception as exc:
            message = f"LLM request failed: {exc}"
            return LLMDecisionResult(
                action_vector=self.prev_action.copy(),
                thought_process=message,
                action_payload=self._vector_to_dict(self.prev_action),
                raw_response_text="",
                error=message,
            )
        raw_text = self._extract_output_text(response)
        if not raw_text:
            message = "Model returned no textual content."
            return LLMDecisionResult(
                action_vector=self.prev_action.copy(),
                thought_process=message,
                action_payload=self._vector_to_dict(self.prev_action),
                raw_response_text="",
                error=message,
            )
        try:
            parsed = self._parse_json_payload(raw_text)
        except ValueError as exc:
            message = str(exc)
            return LLMDecisionResult(
                action_vector=self.prev_action.copy(),
                thought_process=message,
                action_payload=self._vector_to_dict(self.prev_action),
                raw_response_text=raw_text,
                error=message,
            )
        action_obj = parsed.get("action") if isinstance(parsed, dict) else None
        if not isinstance(action_obj, dict):
            message = "Response missing 'action' object."
            return LLMDecisionResult(
                action_vector=self.prev_action.copy(),
                thought_process=f"{message}\nRaw: {raw_text}",
                action_payload=self._vector_to_dict(self.prev_action),
                raw_response_text=raw_text,
                error=message,
            )
        try:
            action_vector = self._action_dict_to_vector(action_obj)
        except ValueError as exc:
            message = str(exc)
            return LLMDecisionResult(
                action_vector=self.prev_action.copy(),
                thought_process=f"{message}\nRaw: {raw_text}",
                action_payload=action_obj,
                raw_response_text=raw_text,
                error=message,
            )
        thought = ""
        if isinstance(parsed, dict):
            thought = str(parsed.get("thought_process", "")).strip()
        if not thought:
            thought = "Model did not include thought_process; ensure the prompt enforces it."
        return LLMDecisionResult(
            action_vector=action_vector,
            thought_process=thought,
            action_payload=action_obj,
            raw_response_text=raw_text,
        )

    def _extract_output_text(self, response: Any) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        fragments: List[str] = []
        for item in getattr(response, "output", []):
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) in {"text", "output_text"}:
                        fragments.append(getattr(content, "text", ""))
            elif item_type == "reasoning":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "text":
                        fragments.append(getattr(content, "text", ""))
        if fragments:
            combined = "\n".join(fragments).strip()
            if combined:
                return combined
        choices = getattr(response, "choices", None)
        if choices:
            for choice in choices:
                message = getattr(choice, "message", None)
                if isinstance(message, dict):
                    content = message.get("content")
                else:
                    content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            block_text = block.get("text")
                            if isinstance(block_text, str) and block_text.strip():
                                parts.append(block_text.strip())
                    if parts:
                        return "\n".join(parts).strip()
                text_attr = getattr(choice, "text", None)
                if isinstance(text_attr, str) and text_attr.strip():
                    return text_attr.strip()
        return "\n".join(fragments).strip()

    def _parse_json_payload(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = text[start : end + 1]
                return json.loads(snippet)
            raise ValueError(f"Could not parse JSON from model output: {text}")

    def _action_dict_to_vector(self, action: dict[str, Any]) -> np.ndarray:
        zone_values = action.get("zone_dampers")
        if not isinstance(zone_values, list) or len(zone_values) != self.zone_count:
            raise ValueError("zone_dampers must be a list matching the number of zones.")
        try:
            zone_array = np.array([float(v) for v in zone_values], dtype=np.float32)
            oa = float(action.get("oa_damper"))
            coil = float(action.get("coil_valve"))
            fan = float(action.get("fan_speed"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid actuator value: {exc}") from exc
        vector = np.concatenate([zone_array, np.array([oa, coil, fan], dtype=np.float32)], axis=0)
        clipped = np.clip(vector, self.low_bounds, self.high_bounds)
        return clipped

    def _vector_to_dict(self, vector: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(vector, dtype=np.float32)
        zone_vals = [round(float(v), 4) for v in arr[: self.zone_count]]
        return {
            "zone_dampers": zone_vals,
            "oa_damper": round(float(arr[self.zone_count]), 4),
            "coil_valve": round(float(arr[self.zone_count + 1]), 4),
            "fan_speed": round(float(arr[self.zone_count + 2]), 4),
        }

    def _vector_to_actions(self, vector: np.ndarray) -> HVACActions:
        arr = np.asarray(vector, dtype=np.float32)
        actions = HVACActions()
        actions.zone_dampers = arr[: self.zone_count].tolist()
        actions.oa_damper = float(arr[self.zone_count])
        actions.coil_valve = float(arr[self.zone_count + 1])
        actions.fan_speed = float(arr[self.zone_count + 2])
        return actions

    def _log_action(
        self,
        timestamp: datetime,
        hvac_on: bool,
        prompt: str,
        result: LLMDecisionResult,
    ) -> None:
        applied_json = json.dumps(self._vector_to_dict(result.action_vector), ensure_ascii=False)
        raw_text = result.raw_response_text or json.dumps({"action": result.action_payload}, ensure_ascii=False)
        with self.log_path.open("a", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    timestamp.isoformat(),
                    int(hvac_on),
                    prompt,
                    result.thought_process,
                    applied_json,
                    raw_text,
                    "" if result.error is None else result.error,
                ]
            )


def build_default_zones() -> Sequence[ZoneConfig]:
    """LLMデモ用のゾーン設定."""
    from simulator_humid.simulation import build_default_zones as simulation_build_default_zones

    return simulation_build_default_zones()

def build_action_scaler(config: LLMConfig) -> ActionScaler:

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

def build_simulation_kwargs(config: LLMConfig) -> dict:

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

def compute_step_rewards(df: pd.DataFrame, config: LLMConfig) -> tuple[np.ndarray, np.ndarray]:

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


def compute_llm_metrics(df: pd.DataFrame, *, setpoint: float) -> dict[str, float]:

    """LLM実行の性能指標を計算（best_results.txt 風の要約出力用）。

    - 温度: 平均絶対誤差、最大絶対誤差、快適域(25-27℃)違反率
    - 電力: 平均電力[kW]（ファン+ポンプ+チラー）、総電力量[kWh]（1分刻み想定）
    - CO2: 平均/最大CO2[ppm]
    - 追加: co2_penalty, co2_violation_penalty の平均
    """

    hvac_df = df[df["hvac_on"]] if "hvac_on" in df.columns else df
    if hvac_df.empty:
        hvac_df = df

    metrics: dict[str, float] = {
        "mean_temp_error": 0.0,
        "max_temp_error": 0.0,
        "comfort_violation_ratio": 0.0,
        "mean_power_kw": 0.0,
        "total_power_kwh": 0.0,
        "mean_co2_ppm": 0.0,
        "max_co2_ppm": 0.0,
        "mean_co2_penalty": 0.0,
        "mean_co2_violation_penalty": 0.0,
    }

    # 温度関連
    zone_temp_cols = sorted(col for col in hvac_df.columns if col.startswith("zone") and col.endswith("_temp"))
    if zone_temp_cols:
        temps = hvac_df[zone_temp_cols].to_numpy()
        temp_errors = temps - float(setpoint)
        if temp_errors.size:
            metrics["mean_temp_error"] = float(np.mean(np.abs(temp_errors)))
            metrics["max_temp_error"] = float(np.max(np.abs(temp_errors)))
            comfort_violations = (temps < 25.0) | (temps > 27.0)
            metrics["comfort_violation_ratio"] = float(np.mean(comfort_violations))

    # 電力関連
    power_cols = ["fan_power_kw", "chw_pump_power_kw", "chiller_power_kw"]
    available_power_cols = [c for c in power_cols if c in hvac_df.columns]
    if available_power_cols:
        power_sum_kw = hvac_df[available_power_cols].sum(axis=1)
        if not power_sum_kw.empty:
            metrics["mean_power_kw"] = float(power_sum_kw.mean())
            metrics["total_power_kwh"] = float(power_sum_kw.sum()) / 60.0

    # CO2関連
    co2_cols = sorted(col for col in hvac_df.columns if col.startswith("zone") and col.endswith("_co2_ppm"))
    if co2_cols:
        co2_vals = hvac_df[co2_cols].to_numpy()
        if co2_vals.size:
            metrics["mean_co2_ppm"] = float(np.mean(co2_vals))
            metrics["max_co2_ppm"] = float(np.max(co2_vals))

    # ペナルティ（あれば）
    if "co2_penalty" in hvac_df.columns:
        metrics["mean_co2_penalty"] = float(np.mean(hvac_df["co2_penalty"].to_numpy()))
    if "co2_violation_penalty" in hvac_df.columns:
        metrics["mean_co2_violation_penalty"] = float(np.mean(hvac_df["co2_violation_penalty"].to_numpy()))

    return metrics

def run_llm_agent(
    config: LLMConfig,
    *,
    model: str = "gpt-5",
) -> pd.DataFrame:

    """LLMベースの制御でシミュレーションを1回実行し、結果を保存する。"""

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Set it in your environment or .env file before running.")

    client = OpenAI()
    if not config.zones:
        config.zones = build_default_zones()

    scaler = build_action_scaler(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "llm_actions_log.csv"

    controller = LLMDecisionController(
        client=client,
        config=config,
        scaler=scaler,
        log_path=log_path,
        model=model,
    )

    sim_kwargs = build_simulation_kwargs(config)
    df = run_simulation(action_callback=controller, verbose_steps=True, **sim_kwargs)
    df_llm = df.copy()

    rewards, mask = compute_step_rewards(df_llm, config)
    df_llm["llm_reward"] = rewards

    results_path = config.output_dir / "llm_simulation_results.csv"
    df_llm.to_csv(results_path, index=True)

    plot_dir = config.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    create_plots(df_llm, plot_dir / "llm_run.png", plot_dir / "llm_run_dampers.png")

    active_steps = int(mask.sum())
    total_reward = float(np.sum(rewards))
    mean_reward = float(np.mean(rewards[mask])) if active_steps > 0 else 0.0

    summary = {
        "model": model,
        "total_reward": total_reward,
        "mean_reward_active": mean_reward,
        "active_steps": active_steps,
        "results_csv": results_path.name,
        "action_log_csv": log_path.name,
    }

    summary_path = config.output_dir / "llm_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(
        f"LLM control completed with total_reward={total_reward:.3f}, "
        f"active_steps={active_steps}, mean_active_reward={mean_reward:.4f}"
    )

    # === best_results.txt 風のテキスト要約を保存 ===
    try:
        metrics = compute_llm_metrics(df_llm, setpoint=config.setpoint)
        best_txt_path = config.output_dir / "best_results.txt"
        with best_txt_path.open("a", encoding="utf-8") as fp:
            fp.write(
                (
                    f"=== ベスト更新 ===\n"
                    f"Episode: 1\n"
                    f"total_return: {total_reward:.4f}\n"
                    f"平均温度誤差: {metrics['mean_temp_error']:.3f}°C\n"
                    f"快適域違反率: {metrics['comfort_violation_ratio']:.3f}\n"
                    f"平均電力消費: {metrics['mean_power_kw']:.3f}kW\n"
                    f"総電力量: {metrics['total_power_kwh']:.3f}kWh\n"
                    f"平均CO2濃度: {metrics['mean_co2_ppm']:.1f}ppm\n"
                    f"最大CO2濃度: {metrics['max_co2_ppm']:.1f}ppm\n"
                    f"co2_penalty: {metrics['mean_co2_penalty']:.3f}, co2_violation_penalty: {metrics['mean_co2_violation_penalty']:.3f}\n\n"
                )
            )
        print(f"best_results.txt を保存: {best_txt_path}")
    except Exception as exc:
        print(f"best_results.txt の保存に失敗: {exc}")

    return df_llm

def parse_args() -> argparse.Namespace:
    """CLI引数を解析し、LLM制御のパラメータを取得する。"""
    
    parser = argparse.ArgumentParser(
        description="Run HVAC control using LLM decision agent."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs (CSV, plots, logs). Defaults to outputs/llm.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5",
        help="Model name for the LLM-based controller.",
    )
    return parser.parse_args()


def main() -> None:
    """CLIエントリポイント: LLM制御を実行する。"""
    
    args = parse_args()
    config = LLMConfig()
    config.zones = build_default_zones()
    
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    run_llm_agent(
        config=config,
        model=args.llm_model,
    )

if __name__ == "__main__":
    # CLI経由でLLM制御を実行
    main()
    
