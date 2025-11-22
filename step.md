# train_grpo.py セットアップ手順（クラウドGPU・ターミナル専用）

更新日: 2025-10-27（この日時点での互換構成）

本手順は、このリポジトリの `train_grpo.py` を単一GPUマシン上で実行するための最小セットアップです。GUIは不要で、ターミナルのみを前提にしています。

---

## 対応バージョンまとめ（検証済み）

- Python: 3.11 または 3.12
- NVIDIA Driver: 570+ を推奨（CUDA 12.8 の公式要件。DC GPUは互換パッケージで R470/R525/R535/R545 も可だが推奨は570+）
- PyTorch: 2.8.0（CUDA 12.8ビルド）
- Transformers: 4.57.1 以上（`Mxfp4Config` 利用のため）
- TRL (GRPO): 0.23.1（vLLMコロケート対応の安定版）
- vLLM: 0.10.2（colocate利用; GPT‑OSS用の 0.10.1+gptoss でも可）
- PEFT: 0.17.1 以上（`target_modules="all-linear"`, `target_parameters` を使用）
- 追加: `triton>=3.4`（MXFP4用 Triton; PyTorch 2.8 では同梱・互換、2.7 系は明示導入推奨）, `flash-attn==2.8.3`（FlashAttention 2。Ampere/Ada/Hopper GPU を対象に、`packaging` と `ninja` を先に入れて `uv pip install --no-build-isolation` で導入）, `datasets`, `pandas`, `accelerate`, `huggingface_hub>=0.25`

注:
- 本スクリプトは `Mxfp4Config(dequantize=True)` を用いて MXFP4 から BF16 にデクオンした上で LoRA 学習します。MXFP4 での後方伝播カーネルは現状不要です。
- `openai/gpt-oss-20b` の推論は vLLM をコロケート起動で使用します（TRLが内部で起動・停止）。
 - MXFP4 利用時のハード前提: Compute Capability 7.5 以上（T4/以降, Ampere/Ada/Hopper/Blackwell 等）

---

## 1. Python 環境の用意（uv 推奨, 3分）

uv は高速なパッケージ管理ツールです。未導入ならインストールします。

```
# uv インストール（ユーザ領域）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 仮想環境（Python 3.12推奨）
uv venv --python 3.12
source .venv/bin/activate

# pip/uv 自体を更新
uv pip install -U pip wheel setuptools
```

（標準の venv/pip を使いたい場合は `python3 -m venv .venv && source .venv/bin/activate` でもOK）

---

## 2. 依存関係のインストール（10分）

vLLM 0.10.2 の公式ホイールは、対応する PyTorch/cuDNN を自動で解決できます。まず vLLM を入れてから、NLP関連を追加で入れると依存競合が起きにくいです。

```
# vLLM 0.10.2（CUDA 12.8向け公式ホイールレジストリ）
uv pip install "vllm==0.10.2" \
  --extra-index-url https://wheels.vllm.ai/0.10.2/ \
  --config-settings vllm:torch-backend=auto

# NLP/学習系ライブラリ（FlashAttention 2 を含む）
uv pip install --no-build-isolation \
  "transformers>=4.57.1" \
  "trl==0.23.1" \
  "peft>=0.17.1" \
  "accelerate>=1.10.0" \
  datasets pandas \
  "huggingface_hub>=0.25" \
  packaging ninja \
  "kernels>=0.10" \
  "triton>=3.4" \
  "liger-kernel"

**Liger対応:** 上記コマンドで `liger-kernel` を導入しています。ビルドが失敗する環境では代わりに  
`uv pip install "trl[liger] @ git+https://github.com/huggingface/trl.git"` を実行し、CUDA / PyTorch に適合した Liger カーネルを用意してください。

# FlashAttention 2 のビルド時間が長い場合は `MAX_JOBS=4 uv pip install --no-build-isolation ...` のように CPU スレッド数を制限すると安定します。

# もし torch が未導入の場合（vLLM 同梱解決が失敗した環境向け）
# CUDA 12.8 の公式ホイールからインストール
# uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0" "torchvision==0.19.0" "torchaudio==2.8.0"
```

---

## 3. リポジトリの取得（1分）

```
git clone https://github.com/takumi0211/tes_grpo_trl.git -b main
cd tes_grpo_trl
```

（既存ワークツリーがある場合はこの手順は不要）

---


## 5. 実行（学習の開始）

```
# 学習を開始
python train_grpo.py
```

学習は以下の構成で行われます:
- `openai/gpt-oss-20b` を MXFP4 ロード → BF16 にデクオンして LoRA 学習
- GRPO の損失計算は `GRPOConfig(use_liger_loss=True, loss_type="bnpo")` で Liger のチャンク化ロスを使用（B×T×V 常駐を回避。Liger は現状 `loss_type="dapo"` に未対応のため `bnpo` に切替済み）
- TRL の GRPOTrainer が vLLM をコロケート起動（`vllm_enable_sleep_mode=True` で VRAM 回収）
- 1 step あたり 12 プロンプト × 各 8 生成（設定値は `train_grpo.py` を参照）

成果物は `output/grpo_gptoss20b_lora4_tes/` に保存されます（LoRA アダプタとトークナイザ）。

作成済み run をローカルへ持ち帰りたい場合は、リポジトリ直下で以下を実行して ZIP 化してください。

```
# run ディレクトリを ZIP に固める（例）
zip -r grpo_run.zip output
```

ZIP 化した `grpo_run.zip` を `scp` / `rsync` / VS Code Remote などでダウンロードすれば完了です。

nvidia chipの使用状況確認
```
watch -n 5 nvidia-smi
```
```
nvidia-smi dmon -s pucvmt -d 5
```

---

## 6. うまくいかない時のチェック

- vRAM 不足: `GRPOConfig.vllm_gpu_memory_utilization` を下げる、`MAX_COMPLETION_LEN` を短くする、`NUM_GENERATIONS` を減らす。
- CUDA/ドライバ不整合: `nvidia-smi` のドライバが 570 未満なら更新（DC GPU で互換パッケージ運用の例外はあるが、基本は 570+）。PyTorch を `cu128` ホイールで再インストール。
- vLLM が起動できない: `pip freeze | grep vllm` で 0.10.2 であること、`torch` が 2.8.0 であることを確認。
- ImportError: MXFP4 関連で `triton` や FlashAttention カーネルが未導入だと失敗します。`uv pip install --no-build-isolation packaging ninja "flash-attn==2.8.3" "triton>=3.4"` を追加実行し、初回実行で `kernels-community/triton_kernels` と FlashAttention の CUDA 拡張がビルドされることを確認してください。

---

## 7. 再現性メモ

環境差分を最小化したい場合は、導入したバージョンをそのまま `requirements.txt` に固定し、`uv pip compile`/`uv pip sync` 等でロックすることを推奨します。

例:

```
uv pip freeze > requirements.txt
```

---

## 8. 参考（要点）

- GPT‑OSS 20B は MXFP4 での軽量ロードを公式にサポートし、学習時は BF16 へデクオンして LoRA を当てる構成が推奨です。
- TRL の GRPO は vLLM の生成エンジンをコロケート起動でき、`sleep mode` により生成⇄学習の切替で VRAM を解放できます。

参考URL:
- CUDA 12.8/ドライバ 570+ 要件（互換パッケージの例外含む）: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#pytorch
- CUDA 12.8 リリースノート（最小ドライバ例: Linux 570.124.06）: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/cuda-toolkit-12-8-0/index.html
- MXFP4 の前提（Accelerate, kernels, Triton≥3.4 と Compute Capability 7.5+）: https://huggingface.co/docs/transformers/main/en/quantization/mxfp4
- vLLM（CUDA 12.8 をデフォルトにビルド）: https://docs.vllm.ai/en/latest/getting_started/installation/cuda.html
- FlashAttention 2（インストール要件と GPU サポート）: https://pypi.org/project/flash-attn/
