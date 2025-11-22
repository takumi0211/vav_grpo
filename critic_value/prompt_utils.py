"""Prompt-building helpers shared across critic_value scripts."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import importlib.util
import sys
import types

import pandas as pd

if importlib.util.find_spec("ollama") is None:  # pragma: no cover - falls back in CI/help envs
    stub = types.ModuleType("ollama")

    class _StubClient:
        def chat(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Ollama stub should not be called during prompt building")

    class _StubResponseError(Exception):
        pass

    stub.Client = _StubClient
    stub.ResponseError = _StubResponseError
    sys.modules["ollama"] = stub

from simulator_humid.agents.ollama_agent import (
    LLMConfig,
    LLMDecisionController,
    build_action_scaler,
    build_default_zones,
)

_PROMPT_LOG_PATH = Path("critic_value/logs/prompt_builder_log.csv")


class _DummyClient:
    """Minimal stub so LLMDecisionController can be reused offline."""

    def chat(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - defensive
        raise RuntimeError("Dummy client should never be invoked during dataset prep")


@lru_cache(maxsize=1)
def _prompt_controller() -> LLMDecisionController:
    config = LLMConfig(zones=build_default_zones())
    scaler = build_action_scaler(config)
    return LLMDecisionController(
        client=_DummyClient(),
        config=config,
        scaler=scaler,
        log_path=_PROMPT_LOG_PATH,
        model="stub",
    )


def get_prompt_controller() -> LLMDecisionController:
    """Return a singleton LLMDecisionController wired to the Ollama agent helpers."""

    return _prompt_controller()


def compose_prompt_from_agent(state_obj: Any) -> str:
    """Compose a full prompt (system + user) identical to the live agent."""

    controller = get_prompt_controller()
    user_prompt = controller._build_prompt(state_obj)
    return controller._compose_full_prompt(user_prompt)


def build_state_dict(row: Mapping[str, Any], config: LLMConfig | None = None) -> dict[str, Any]:
    """Reconstruct a prompt-friendly state summary from a tabular row."""

    controller = get_prompt_controller()
    cfg = config or controller.config

    ts_obj = pd.to_datetime(row.get("timestamp"), errors="coerce")
    has_ts = ts_obj is not None and not pd.isna(ts_obj)
    timestamp_iso = ts_obj.to_pydatetime().isoformat() if has_ts else ""

    zone_entries: list[dict[str, Any]] = []
    for idx, zone in enumerate(cfg.zones):
        temp_err = float(row.get(f"zone_{idx+1}_temp_error", 0.0))
        temp_delta = float(row.get(f"zone_{idx+1}_temp_delta", 0.0))
        co2_err = float(row.get(f"zone_{idx+1}_co2_error", 0.0))
        co2_delta = float(row.get(f"zone_{idx+1}_co2_delta", 0.0))
        zone_entries.append(
            {
                "name": zone.name,
                "temperature_c": round(cfg.setpoint + temp_err, 1),
                "temp_error_c": round(temp_err, 1),
                "temp_delta_c_per_min": round(temp_delta, 2),
                "co2_ppm": round(cfg.co2_target_ppm + co2_err, 0),
                "co2_delta_ppm_per_min": round(co2_delta, 1),
            }
        )

    outdoor_temp = float(row.get("outdoor_temp", 0.0))
    outdoor_slope = float(row.get("outdoor_temp_slope", 0.0))

    def _round_action(value: Any) -> float:
        return round(float(value), 2)

    prev_zone = [_round_action(row.get(f"prev_zone_{i+1}_damper", 0.0)) for i in range(len(cfg.zones))]
    prev_oa = _round_action(row.get("prev_oa_damper", 0.0))
    prev_coil = _round_action(row.get("prev_coil_valve", 0.0))
    prev_fan = _round_action(row.get("prev_fan_speed", 0.0))

    time_features = {
        "hour": int(ts_obj.hour) if has_ts else 0,
        "minute": int(ts_obj.minute) if has_ts else 0,
        "second": int(ts_obj.second) if has_ts else 0,
    }

    return {
        "timestamp": timestamp_iso,
        "setpoint_c": float(cfg.setpoint),
        "control_interval_minutes": float(cfg.control_interval_minutes),
        "time_features": time_features,
        "zones": zone_entries,
        "outdoor": {
            "temperature_c": round(outdoor_temp, 1),
            "temperature_trend_c_per_hour": round(outdoor_slope, 2),
        },
        "previous_action": {
            "zone_dampers": prev_zone,
            "oa_damper": prev_oa,
            "coil_valve": prev_coil,
            "fan_speed": prev_fan,
        },
        "action_bounds": controller.action_bounds,
    }


def sanitize_observation(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove humidity-only keys so prompt JSON stays aligned with agent schema."""

    clean = copy.deepcopy(payload)
    zones = clean.get("zones")
    if isinstance(zones, list):
        for zone in zones:
            if isinstance(zone, dict):
                zone.pop("relative_humidity_pct", None)
                zone.pop("rh_delta_pct_per_min", None)
    outdoor = clean.get("outdoor")
    if isinstance(outdoor, dict):
        outdoor.pop("absolute_humidity_kg_per_kg", None)
        outdoor.pop("abs_humidity_trend_per_hour", None)
    return clean
