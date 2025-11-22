#!/usr/bin/env python3
"""Sample Ollama actions for a dataset prompt and score them with the TD3 critic.

Steps performed:
1) Pick one row from critic_value/data/dataset.csv (prompt + state vector).
2) Send the prompt to the Ollama agent multiple times to sample action candidates.
3) Convert the sampled actions to tanh space, feed (state, action) to the critic,
   and obtain q1/q2/q_min for each candidate.
4) Report the winner = action with the larger q1.

Only files inside critic_value are touched; simulator_humid code is imported but not modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch

# Provide a lightweight stub when the Ollama Python package is missing.
try:  # pragma: no cover - import guard
    from ollama import Client, ResponseError
except ImportError:  # pragma: no cover - fallback for help/CI environments
    class _StubClient:
        def __init__(self, *_, **__):
            raise RuntimeError("ollama package is not installed. Install it to run sampling.")

    class _StubResponseError(Exception):
        pass

    Client = _StubClient  # type: ignore
    ResponseError = _StubResponseError  # type: ignore

# Ensure repository root is on sys.path when executed directly
if __package__ in (None, ""):
    import pathlib
    import sys

    _project_root = pathlib.Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from critic_value.common import default_device, derive_dims, load_td3_checkpoint  # noqa: E402
from critic_value.llm_tools import (  # noqa: E402
    LLMDecisionResult,
    action_dict_to_vector,
    build_action_bounds,
    evaluate_candidate,
    parse_json_payload,
    pick_row,
    resolve_observation_vector,
    softmax_from_scores,
)
from simulator_humid.agents.rl.training_td3 import (  # noqa: E402
    TwinQNetwork,
    build_action_scaler,
)


# --------------------------------------------------------------------------- #
# Lightweight copies of Ollama agent helpers (kept inside critic_value only) #
# --------------------------------------------------------------------------- #


def _extract_output_text(response: Any) -> tuple[str, str]:
    """Collect message text & chain-of-thought style fields from an Ollama response."""

    def _get(obj: Any, name: str) -> Any:
        return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

    def _collect_message(value: Any, bucket: List[str]) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                bucket.append(text)
        elif isinstance(value, list):
            for element in value:
                if isinstance(element, dict):
                    _collect_message(element.get("text"), bucket)
        elif isinstance(value, dict):
            _collect_message(value.get("text"), bucket)

    def _collect_thinking(value: Any, bucket: List[str]) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                bucket.append(text)

    def _dedupe(parts: List[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for part in parts:
            if part not in seen:
                ordered.append(part)
                seen.add(part)
        return ordered

    message_parts: List[str] = []
    thinking_parts: List[str] = []

    message_obj = getattr(response, "message", None)
    if message_obj is None and isinstance(response, dict):
        message_obj = response.get("message")
    if message_obj is not None:
        _collect_message(_get(message_obj, "content"), message_parts)
        _collect_thinking(_get(message_obj, "thinking"), thinking_parts)

    output_items = getattr(response, "output", None)
    if output_items is None and isinstance(response, dict):
        output_items = response.get("output")
    if output_items:
        for item in output_items:
            item_type = _get(item, "type")
            if item_type == "message":
                for content in _get(item, "content") or []:
                    if _get(content, "type") in {"text", "output_text"}:
                        _collect_message(_get(content, "text"), message_parts)
            elif item_type == "reasoning":
                for content in _get(item, "content") or []:
                    if _get(content, "type") == "text":
                        _collect_thinking(_get(content, "text"), thinking_parts)

    _collect_message(_get(response, "output_text"), message_parts)
    _collect_thinking(_get(response, "thinking"), thinking_parts)

    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if choices:
        for choice in choices:
            message = _get(choice, "message")
            if message is not None:
                _collect_message(_get(message, "content"), message_parts)
                _collect_thinking(_get(message, "thinking"), thinking_parts)
            _collect_message(_get(choice, "text"), message_parts)
            _collect_thinking(_get(choice, "reasoning"), thinking_parts)

    message_parts = _dedupe(message_parts)
    thinking_parts = _dedupe(thinking_parts)
    message_text = "\n".join(message_parts).strip()
    thinking_text = "\n".join(thinking_parts).strip()
    return message_text, thinking_text


class OllamaSampler:
    """Thin wrapper that reuses the Ollama agent's parsing/clipping logic."""

    def __init__(
        self,
        *,
        client: Client,
        model: str,
        zone_count: int,
        low_bounds: np.ndarray,
        high_bounds: np.ndarray,
    ) -> None:
        self.client = client
        self.model = model
        self.zone_count = zone_count
        self.low_bounds = low_bounds.astype(np.float32)
        self.high_bounds = high_bounds.astype(np.float32)
        self.prev_action = (self.low_bounds + self.high_bounds) * 0.5

    def sample(self, prompt: str) -> LLMDecisionResult:
        """Send the prompt to Ollama and return a parsed action vector."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except ResponseError as exc:
            message = f"LLM request failed: {getattr(exc, 'error', None) or exc}"
            return LLMDecisionResult(
                action_vector=self.prev_action.copy(),
                thought_process=message,
                action_payload={},
                raw_response_text="",
                error=message,
            )
        except Exception as exc:  # pragma: no cover - defensive
            message = f"LLM request failed: {exc}"
            return LLMDecisionResult(
                action_vector=self.prev_action.copy(),
                thought_process=message,
                action_payload={},
                raw_response_text="",
                error=message,
            )

        raw_text, thinking_text = _extract_output_text(response)
        if not raw_text:
            message = "Model returned no textual content."
            return LLMDecisionResult(
                action_vector=self.prev_action.copy(),
                thought_process=message,
                action_payload={},
                raw_response_text="",
                model_thinking=thinking_text,
                error=message,
            )
        try:
            parsed = parse_json_payload(raw_text)
        except ValueError as exc:
            message = str(exc)
            return LLMDecisionResult(
                action_vector=self.prev_action.copy(),
                thought_process=message,
                action_payload={},
                raw_response_text=raw_text,
                model_thinking=thinking_text,
                error=message,
            )

        action_obj = parsed.get("action") if isinstance(parsed, dict) else None
        if not isinstance(action_obj, dict):
            message = "Response missing 'action' object."
            return LLMDecisionResult(
                action_vector=self.prev_action.copy(),
                thought_process=f"{message}\nRaw: {raw_text}",
                action_payload=action_obj if isinstance(action_obj, dict) else {},
                raw_response_text=raw_text,
                model_thinking=thinking_text,
                error=message,
            )

        try:
            action_vector = action_dict_to_vector(
                action_obj,
                zone_count=self.zone_count,
                low_bounds=self.low_bounds,
                high_bounds=self.high_bounds,
            )
        except ValueError as exc:
            message = str(exc)
            return LLMDecisionResult(
                action_vector=self.prev_action.copy(),
                thought_process=f"{message}\nRaw: {raw_text}",
                action_payload=action_obj,
                raw_response_text=raw_text,
                model_thinking=thinking_text,
                error=message,
            )

        thought = ""
        if isinstance(parsed, dict):
            thought = str(parsed.get("thought_process", "")).strip()
        if not thought:
            thought = "Model did not include thought_process; ensure the prompt enforces it."

        self.prev_action = action_vector.copy()
        return LLMDecisionResult(
            action_vector=action_vector,
            thought_process=thought,
            action_payload=action_obj,
            raw_response_text=raw_text,
            model_thinking=thinking_text,
        )


# --------------------------------------------------------------------------- #
# Core evaluation routine                                                    #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Ollama actions for one prompt and score with the TD3 critic.")
    parser.add_argument("--dataset", type=Path, default=Path("critic_value/data/dataset.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("critic_value/td3_policy_final.pt"))
    parser.add_argument("--model", type=str, default=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
    parser.add_argument("--prompt-sample-id", type=int, default=0, help="sample_id to load from the dataset")
    parser.add_argument("--row-index", type=int, default=None, help="fallback row index if sample_id is absent")
    parser.add_argument("--samples", type=int, default=4, help="number of LLM samples to draw (default: 4)")
    parser.add_argument("--device", type=str, default=None, help="torch device (default: auto)")
    parser.add_argument("--output", type=Path, default=Path("critic_value/data/llm_vs_critic_results.csv"))
    parser.add_argument("--softmax-temp", type=float, default=0.5, help="Temperature used for q1 softmax weighting (default: 0.5)")
    args = parser.parse_args()

    device = default_device(args.device)

    # --- load dataset and select row ---
    df = pd.read_csv(args.dataset)
    try:
        row = pick_row(df, sample_id=args.prompt_sample_id, row_index=args.row_index)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    prompt_text = str(row.get("prompt", "")).strip()
    if not prompt_text:
        raise SystemExit("Selected row does not contain a prompt.")

    # --- critic + normalizer ---
    config, normalizer, checkpoint = load_td3_checkpoint(args.checkpoint, device=device)
    obs_dim, action_dim = derive_dims(config)
    try:
        obs_vec_norm = resolve_observation_vector(row, normalizer=normalizer, obs_dim=obs_dim)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    obs_tensor_norm = torch.from_numpy(obs_vec_norm).unsqueeze(0).to(device)

    critic_hidden = tuple(config.critic_hidden_sizes) if config.critic_hidden_sizes else tuple(config.hidden_sizes)
    critic = TwinQNetwork(obs_dim, action_dim, critic_hidden).to(device)
    if "critic_state_dict" not in checkpoint:
        raise SystemExit("Checkpoint missing 'critic_state_dict'.")
    critic.load_state_dict(checkpoint["critic_state_dict"])
    critic.eval()

    scaler = build_action_scaler(config).to(device)
    low_bounds, high_bounds = build_action_bounds(config)

    # --- ollama sampler ---
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client = Client(host=ollama_host)
    sampler = OllamaSampler(
        client=client,
        model=args.model,
        zone_count=len(config.zones),
        low_bounds=low_bounds,
        high_bounds=high_bounds,
    )

    results: list[dict[str, Any]] = []
    for i in range(max(1, args.samples)):
        sample = sampler.sample(prompt_text)
        eval_scores = evaluate_candidate(
            critic=critic,
            scaler=scaler,
            device=device,
            obs_tensor_norm=obs_tensor_norm,
            action_vec_scaled=sample.action_vector,
        )
        results.append(
            {
                "candidate": i,
                "error": sample.error or "",
                "thought_process": sample.thought_process,
                "model_thinking": sample.model_thinking,
                "raw_response": sample.raw_response_text,
                "action_zone_dampers": sample.action_payload.get("zone_dampers") if sample.action_payload else "",
                "action_oa_damper": sample.action_payload.get("oa_damper") if sample.action_payload else "",
                "action_coil_valve": sample.action_payload.get("coil_valve") if sample.action_payload else "",
                "action_fan_speed": sample.action_payload.get("fan_speed") if sample.action_payload else "",
                **eval_scores,
            }
        )

    results_df = pd.DataFrame(results)
    q1_values = results_df["q1"].to_numpy(dtype=np.float64)
    scores = softmax_from_scores(q1_values, temperature=args.softmax_temp)
    results_df["softmax_q1"] = scores
    results_df["winner"] = results_df["q1"] == results_df["q1"].max()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    winner_row = results_df.loc[results_df["q1"].idxmax()]
    print("\n=== Prompt sample ===")
    print(f"sample_id={row.get('sample_id', 'n/a')}  timestamp={row.get('timestamp', 'n/a')}")
    print(f"Using model: {args.model} @ {ollama_host}")
    print(f"Observation dims: {obs_dim}  Action dims: {action_dim}")

    print("\n=== Candidate scores (higher q1 wins) ===")
    for _, rec in results_df.iterrows():
        action_summary = [
            f"zone_dampers={rec['action_zone_dampers']}",
            f"oa={rec['action_oa_damper']}",
            f"coil={rec['action_coil_valve']}",
            f"fan={rec['action_fan_speed']}",
        ]
        print(
            f"[{int(rec['candidate'])}] q1={rec['q1']:.4f}  q2={rec['q2']:.4f}  "
            f"q_min={rec['q_min']:.4f}  q_mean={rec['q_mean']:.4f}  "
            f"softmax_q1={rec['softmax_q1']:.3f}  "
            f"{'WINNER' if rec['winner'] else ''}"
        )
        if rec["error"]:
            print(f"  error: {rec['error']}")
        else:
            print("  action:", "; ".join(action_summary))

    print(
        f"\nWinner candidate: {int(winner_row['candidate'])} "
        f"(q1={winner_row['q1']:.4f}, softmax={winner_row['softmax_q1']:.3f})"
    )
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
