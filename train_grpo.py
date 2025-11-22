# train_grpo_single_gpu.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config, logging as hf_logging
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
import torch
from train_support.data_reward import load_prompt_dataset, reward_fn
from train_support.step_stream import StepStream
import os, random
# import os, random, logging

MODEL_ID = "openai/gpt-oss-20b"
OUT = "output/grpo_gptoss20b_lora4_tes"

TOTAL_STEPS = 200
SAVE_STEPS = 20
NUM_GENERATIONS = 16           # プロンプトごとにサンプルされる完了数
GRADIENT_ACCUMULATION_STEPS = 4
PROMPTS_PER_STEP = 1          # マイクロステップごとにサンプルされる異なるプロンプト数
TRAIN_BATCH_SIZE = NUM_GENERATIONS  # マイクロバッチ = 1プロンプト分の完了数
MAX_PROMPT_LEN = 1000
MAX_COMPLETION_LEN = 4000
SEED = 42

# Reward logger uses this to reconstruct micro-step indices per optimizer step
os.environ.setdefault("GRPO_STEPS_PER_GENERATION", str(GRADIENT_ACCUMULATION_STEPS))

# trainer.pyが全体の設定を表示してくれる
hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()

# --- トークナイザー ---
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# --- MXFP4 -> BF16 にデクオンして学習（公式ルート） ---
# 参考: OpenAI/Transformersのcookbook・ブログで Mxfp4Config(dequantize=True) を明記。:contentReference[oaicite:5]{index=5}
quant_cfg = Mxfp4Config(dequantize=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    quantization_config=quant_cfg,
    attn_implementation="kernels-community/vllm-flash-attn3",
    use_cache=False,               # 勾配チェックポイントと相性良
    device_map="auto",
)

# --- LoRA r=4 ---
# LoRA: Self-AttentionのQ, K, Vを対象に適用
lora = LoraConfig(
    r=4, lora_alpha=8,
    target_modules="all-linear",
    task_type="CAUSAL_LM",            
)

model = get_peft_model(model, lora)

# ----------------- Dataset (データローダ) ----------------
base = load_prompt_dataset()
random.seed(SEED)
stream = StepStream(base, k=PROMPTS_PER_STEP, num_generations=NUM_GENERATIONS)

# ----------------- TRL/GRPO + vLLM (colocate) -----------------
# colocate: 学習プロセス内でvLLMを起動（省メモリのため sleep を有効化）。
# ※ vLLM 0.10.2 を使用（TRLのサポートバージョン）
args = GRPOConfig(
    output_dir=OUT,
    max_steps=TOTAL_STEPS,
    learning_rate=5e-5,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    seed=SEED,
    accelerator_config={"split_batches": True},
    logging_steps=1,
    save_steps=SAVE_STEPS,
    use_liger_loss=True,
    loss_type="dr_grpo",
    mask_truncated_completions=True,
    scale_rewards=False,                  # Dr.GRPOの推奨

    # 生成エンジン（vLLM）
    use_vllm=False,
    # vllm_mode="colocate",
    # vllm_gpu_memory_utilization=0.35,  # 学習と競合しないよう枠を抑える
    # # vllm_kv_cache_dtype="fp8",
    # vllm_enable_sleep_mode=True,       # 生成←→学習の切替でVRAMを返す（初回のみ起床遅延あり）

    # 各マイクロステップで 1 プロンプト × 4 completion を生成
    num_generations=NUM_GENERATIONS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    steps_per_generation=GRADIENT_ACCUMULATION_STEPS,
    
    # 長さまわり
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_COMPLETION_LEN,
    # 生成エンジンの設定
    generation_kwargs={
        "use_cache": True,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "eos_token_id": tok.eos_token_id,
    },
)

# 実行
trainer = GRPOTrainer(
    model=model,
    processing_class=tok,   # 現行API名（左パディング必須）
    args=args,
    reward_funcs=reward_fn,
    train_dataset=stream,   # 各マイクロステップで 4 completion（1 prompt × 4）を供給
)

trainer.train()

# 保存（LoRAアダプタ形式）
trainer.save_model(OUT)
tok.save_pretrained(OUT)
# logger.debug("Training artifacts saved to %s", OUT)
print("✅ finished")
