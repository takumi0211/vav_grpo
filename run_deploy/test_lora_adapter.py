#!/usr/bin/env python
"""
Run a quick generation with the 10-step LoRA adapter.

Edit the constants near the top of the file if you need to change paths or decoding args,
then run:
    python test_lora_adapter.py
"""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import PeftModel

# ------- テスト設定（必要に応じて書き換えてください） -------
# ADAPTER_PATH = Path("output/grpo_gptoss20b_lora4_tes")  # output/ に移行する場合はこちらに差し替え
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

# 量子化設定（オプション）
try:
    quant_config = Mxfp4Config(dequantize=True)
except AttributeError:
    quant_config = None

# ベースモデル名の取得
adapter_config_path = adapter_path / "adapter_config.json"
if BASE_MODEL_ID:
    base_model_name = BASE_MODEL_ID
elif adapter_config_path.exists():
    with adapter_config_path.open("r", encoding="utf-8") as fh:
        adapter_cfg = json.load(fh)
    base_model_name = adapter_cfg.get("base_model_name_or_path", "openai/gpt-oss-20b")
else:
    base_model_name = "openai/gpt-oss-20b"

# トークナイザーの準備
tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else base_model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# デバイス設定
has_cuda = torch.cuda.is_available()
dtype = torch.bfloat16 if has_cuda else torch.float32
device_map = "auto" if has_cuda else None
target_device = torch.device("cuda:0") if has_cuda else torch.device("cpu")

# モデルの読み込み
model_kwargs = dict(
    torch_dtype=dtype,
    quantization_config=quant_config,
    device_map=device_map,
    low_cpu_mem_usage=True,
)
if has_cuda:
    model_kwargs["attn_implementation"] = "kernels-community/vllm-flash-attn3"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
model = PeftModel.from_pretrained(base_model, adapter_path, device_map=device_map, torch_dtype=dtype)
model.eval()

# 生成
inputs = tokenizer(prompt_text, return_tensors="pt").to(target_device)

generation_kwargs = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    do_sample=DO_SAMPLE,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

with torch.no_grad():
    output_ids = model.generate(**inputs, **generation_kwargs)

prompt_length = inputs["input_ids"].shape[1]
generated_ids = output_ids[:, prompt_length:]
completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

print(completion.strip())
