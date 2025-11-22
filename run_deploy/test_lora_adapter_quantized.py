#!/usr/bin/env python
"""
Run generation with the LoRA adapter while keeping the base model in MXFP4.

Edit the constants below to suit your environment, then execute:
    python test_lora_adapter_quantized.py
"""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import PeftModel

# ------- 実行パラメータ（必要に応じて調整） -------
# ADAPTER_PATH = Path("output/grpo_gptoss20b_lora4_tes")  # output/ に切り替える場合
ADAPTER_PATH = Path("runs_10step分/grpo_gptoss20b_lora4_tes")
PROMPT_PATH = Path("data/test.md")
BASE_MODEL_ID = "openai/gpt-oss-20b"
MAX_NEW_TOKENS = 4000
TEMPERATURE = 0.8
TOP_P = 0.95
DO_SAMPLE = False  # True: サンプリング, False: 貪欲法

# パスの準備
adapter_path = ADAPTER_PATH.expanduser().resolve()
prompt_path = PROMPT_PATH.expanduser().resolve()

# プロンプトの読み込み
if not prompt_path.exists():
    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
prompt_text = prompt_path.read_text(encoding="utf-8").strip()

# 量子化設定
try:
    quant_config = Mxfp4Config(dequantize=False, compute_dtype=torch.bfloat16)
except AttributeError:
    raise RuntimeError("This transformers version does not support Mxfp4Config.")

# ベースモデル名の取得
if BASE_MODEL_ID:
    base_model_name = BASE_MODEL_ID
else:
    adapter_cfg_path = adapter_path / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError("adapter_config.json not found; set BASE_MODEL_ID manually.")
    with adapter_cfg_path.open("r", encoding="utf-8") as fh:
        adapter_cfg = json.load(fh)
    base_model_name = adapter_cfg.get("base_model_name_or_path", "openai/gpt-oss-20b")

# トークナイザーの準備
tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else base_model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# GPUチェック
if not torch.cuda.is_available():
    raise RuntimeError("A CUDA-capable GPU (H100) is required for MXFP4 inference.")

# モデルの読み込み
model_kwargs = dict(
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
if torch.cuda.is_available():
    model_kwargs["attn_implementation"] = "kernels-community/vllm-flash-attn3"

model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
model = PeftModel.from_pretrained(model, adapter_path, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

# 生成
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

generation_kwargs = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    do_sample=DO_SAMPLE,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

with torch.inference_mode():
    outputs = model.generate(**inputs, **generation_kwargs)

prompt_len = inputs["input_ids"].shape[1]
generated = outputs[:, prompt_len:]
completion = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]

print(completion.strip())
