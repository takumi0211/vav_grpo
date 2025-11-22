#!/usr/bin/env python
"""
Merge the trained LoRA adapter into the GPT-OSS base model and store a
quantization hint (MXFP4) alongside the merged weights.

Export procedure (update constants below if paths differ):
  1. export HF_TOKEN=<your token>
  2. export HF_REPO_ID=<your repo id> ex) takumi0211/tes_grpo_trl
  3. Run `python export_quantized_model.py`
"""
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

# ----- Export parameters -----
BASE_MODEL_ID = "openai/gpt-oss-20b"
ADAPTER_PATH = Path("output/grpo_gptoss20b_lora4_tes")  # output/ を使う場合はこの行を有効化
# ADAPTER_PATH = Path("runs_10step分/grpo_gptoss20b_lora4_tes")
OUTPUT_DIR = Path("exports/grpo_gptoss20b_lora4_tes_merged")
SAFE_SERIALIZATION = True  # set False to save as PyTorch binaries (.bin)
PUSH_TO_HUB = True
CREATE_REPO = True
REPO_ID = os.getenv("HF_REPO_ID")
COMMIT_MESSAGE = "Add merged LoRA weights with MXFP4 config"
TOKEN_ENV = "HF_TOKEN"

# 出力ディレクトリの作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# トークナイザーの保存
tok_config = ADAPTER_PATH / "tokenizer_config.json"
tokenizer_source = ADAPTER_PATH if tok_config.exists() else BASE_MODEL_ID
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
tokenizer.save_pretrained(OUTPUT_DIR)

# モデルの読み込み設定
has_cuda = torch.cuda.is_available()
load_kwargs = dict(
    torch_dtype=torch.bfloat16,
    device_map="auto" if has_cuda else None,
    low_cpu_mem_usage=True,
)
if has_cuda:
    load_kwargs["attn_implementation"] = "kernels-community/vllm-flash-attn3"

# LoRAアダプターの読み込みとマージ
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, torch_dtype=torch.bfloat16)
merged_model = peft_model.merge_and_unload()

# 量子化設定の追加
quant_cfg = Mxfp4Config(dequantize=False, compute_dtype=torch.bfloat16)
if hasattr(quant_cfg, "to_dict"):
    quant_cfg_dict = quant_cfg.to_dict()
else:
    quant_cfg_dict = {
        "quant_method": getattr(quant_cfg, "quant_method", "mxfp4"),
        "dequantize": getattr(quant_cfg, "dequantize", False),
    }
merged_model.config.quantization_config = quant_cfg_dict

# マージされたモデルの保存
merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=SAFE_SERIALIZATION)
(OUTPUT_DIR / "quantization_config.json").write_text(json.dumps(quant_cfg_dict, indent=2), encoding="utf-8")

# アダプター設定のコピー（存在する場合）
adapter_config_src = ADAPTER_PATH / "adapter_config.json"
if adapter_config_src.exists():
    adapter_cfg = json.loads(adapter_config_src.read_text(encoding="utf-8"))
    adapter_cfg_path = OUTPUT_DIR / "adapter_config.merged.json"
    adapter_cfg_path.write_text(json.dumps(adapter_cfg, indent=2), encoding="utf-8")

# Hugging Face Hubへのプッシュ（オプション）
if PUSH_TO_HUB:
    token = os.getenv(TOKEN_ENV)
    if not token:
        raise RuntimeError(f"{TOKEN_ENV} environment variable must be set to push to the Hub.")
    if not REPO_ID:
        raise RuntimeError("Set HF_REPO_ID to the target private repo (e.g. 'username/my-private-model').")
    api = HfApi()
    if CREATE_REPO:
        api.create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=True, token=token)
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(OUTPUT_DIR),
        repo_type="model",
        commit_message=COMMIT_MESSAGE,
        token=token,
    )
