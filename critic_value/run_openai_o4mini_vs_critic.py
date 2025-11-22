#!/usr/bin/env python3
"""Sample OpenAI o4-mini actions for a dataset prompt and score them with the TD3 critic.

This mirrors run_llm_vs_critic.py but uses OpenAI Chat Completions instead of Ollama.
Winner = highest q1. Adds a softmax over q1 for quick ranking.
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
from dotenv import load_dotenv
from openai import OpenAI

# ensure repo root on sys.path
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

def _extract_output_text(response: Any) -> tuple[str, str]:
    """Extract message content and optional reasoning-like fields from OpenAI response."""
    message_text = ""
    thinking_text = ""
    choices = getattr(response, "choices", None)
    if choices:
        msg = choices[0].message
        message_text = (msg.content or "").strip()
        thinking_text = (getattr(msg, "reasoning", "") or "").strip()
    return message_text, thinking_text


class OpenAISampler:
    """Minimal sampler that sends the dataset prompt to OpenAI o4-mini and parses JSON."""

    def __init__(
        self,
        *,
        client: OpenAI,
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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
            )
        except Exception as exc:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample OpenAI o4-mini actions for one prompt and score with the TD3 critic.")
    parser.add_argument("--dataset", type=Path, default=Path("critic_value/data/dataset.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("critic_value/td3_policy_final.pt"))
    parser.add_argument("--model", type=str, default="o4-mini", help="OpenAI model name (default: o4-mini)")
    parser.add_argument("--prompt-sample-id", type=int, default=0, help="sample_id to load from the dataset")
    parser.add_argument("--row-index", type=int, default=None, help="fallback row index if sample_id is absent")
    parser.add_argument("--samples", type=int, default=4, help="number of LLM samples to draw (default: 4)")
    parser.add_argument("--device", type=str, default=None, help="torch device (default: auto)")
    parser.add_argument("--output", type=Path, default=Path("critic_value/data/openai_vs_critic_results.csv"))
    parser.add_argument("--softmax-temp", type=float, default=0.5, help="Temperature used for q1 softmax weighting (default: 0.5)")
    args = parser.parse_args()

    # Load API keys from .env if present (align with llm_agent.py behavior)
    load_dotenv()

    device = default_device(args.device)

    df = pd.read_csv(args.dataset)
    try:
        row = pick_row(df, sample_id=args.prompt_sample_id, row_index=args.row_index)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    prompt_text = str(row.get("prompt", "")).strip()
    if not prompt_text:
        raise SystemExit("Selected row does not contain a prompt.")

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

    client = OpenAI()
    sampler = OpenAISampler(
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
    print(f"Using model: {args.model} (OpenAI Chat Completions)")
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
