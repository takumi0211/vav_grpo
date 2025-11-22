#!/usr/bin/env python
# export recap:
#   1. Merge LoRA into the base model via `python export_quantized_model.py`
#      (creates the merged folder under `OUTPUT_DIR`).
#   2. Upload the merged folder to Hugging Face if needed.
#   3. Set the `MODEL_ID` below to the uploaded repo.
#   4. Run `python run_exported_model.py` to generate text from `PROMPT_PATH`.

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# パラメータ設定
MODEL_ID = "takumi0211/tes_grpo"
PROMPT_PATH = Path("data/test.md")
MAX_NEW_TOKENS = 4000
TEMPERATURE = 0.8
TOP_P = 0.95
DO_SAMPLE = True # True: サンプリング, False: 貪欲法

# プロンプトの読み込み
prompt_path = PROMPT_PATH.expanduser().resolve()
if not prompt_path.exists():
    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
prompt_text = prompt_path.read_text(encoding="utf-8").strip()

# GPUチェック
if not torch.cuda.is_available():
    raise RuntimeError("A CUDA-capable GPU (H100) is required for MXFP4 inference.")

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# モデルの読み込み
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
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
