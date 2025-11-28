from __future__ import annotations

"""Run the humid-climate VAV simulator with the trained GRPO model.

This script mirrors `simulator_humid.agents.ollama_agent.run_llm_agent` but
swaps the Ollama backend for a local Hugging Face model and ensures every
prompt is rendered with the Harmony chat template before generation.

Outputs (CSV, PNG, logs) are written directly under `eval_model/`.
"""

import json
import sys
from pathlib import Path
from typing import Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Path setup so we can import the simulator package without modifying it
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
SIM_ROOT = ROOT_DIR / "simulator-humid_1"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

# Import after sys.path adjustment
from simulator_humid_1.agents.ollama_agent import (
    LLMConfig,
    LLMDecisionController,
    build_action_scaler,
    build_default_zones,
    build_simulation_kwargs,
    compute_llm_metrics,
    compute_step_rewards,
)
from simulator_humid_1.simulation import create_plots, run_simulation


# ---------------------------------------------------------------------------
# Generation settings (match run_deploy/run_exported_model.py defaults)
# ---------------------------------------------------------------------------
MODEL_ID = "takumi0211/vav_grpo"
MAX_NEW_TOKENS = 400  # smaller than 4000 to keep the control loop responsive
TEMPERATURE = 0.8
TOP_P = 0.95
DO_SAMPLE = True


class HarmonyHFClient:
    """Minimal client shim that provides `.chat(...)` for LLMDecisionController.

    The incoming messages list is rendered with the model's Harmony chat
    template (`tokenizer.apply_chat_template`) before generation.
    """

    def __init__(
        self,
        model_id: str,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        do_sample: bool = DO_SAMPLE,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required to run the evaluation model.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # Training used left-padding; keep the same.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        # Assume a single-GPU placement (H100 as noted in run_exported_model.py).
        self.device = torch.device("cuda")

        self.generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def chat(self, *, model: str, messages: List[dict[str, str]], **_: Any) -> dict[str, Any]:
        """Generate a single reply in Harmony format."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **self.generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[:, prompt_len:]
        text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

        # LLMDecisionController expects a mapping with a "message" payload.
        return {"message": {"content": text}}


def run_evaluation() -> None:
    """Execute one simulation episode with the GRPO model and save artefacts."""

    output_dir = ROOT_DIR

    config = LLMConfig()
    config.zones = build_default_zones()
    config.output_dir = output_dir  # ensure artefacts stay under eval_model/

    scaler = build_action_scaler(config)
    log_path = output_dir / "llm_actions_log.csv"

    client = HarmonyHFClient(MODEL_ID)
    controller = LLMDecisionController(
        client=client,
        config=config,
        scaler=scaler,
        log_path=log_path,
        model=MODEL_ID,
    )

    sim_kwargs = build_simulation_kwargs(config)
    df = run_simulation(action_callback=controller, verbose_steps=True, **sim_kwargs)

    rewards, mask = compute_step_rewards(df, config)
    df["llm_reward"] = rewards

    results_path = output_dir / "llm_simulation_results.csv"
    df.to_csv(results_path, index=True)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    create_plots(df, plot_dir / "llm_run.png", plot_dir / "llm_run_dampers.png")

    active_steps = int(mask.sum())
    total_reward = float(rewards.sum())
    mean_reward = float(rewards[mask].mean()) if active_steps else 0.0

    summary = {
        "model": MODEL_ID,
        "total_reward": total_reward,
        "mean_reward_active": mean_reward,
        "active_steps": active_steps,
        "results_csv": results_path.name,
        "action_log_csv": log_path.name,
    }

    summary_path = output_dir / "llm_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # Optional text summary (mirrors simulator_humid best_results.txt style)
    metrics = compute_llm_metrics(df, setpoint=config.setpoint)
    best_txt_path = output_dir / "best_results.txt"
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

    print(
        f"LLM control completed with total_reward={total_reward:.3f}, "
        f"active_steps={active_steps}, mean_active_reward={mean_reward:.4f}"
    )
    print(f"Outputs saved under {output_dir}")


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
