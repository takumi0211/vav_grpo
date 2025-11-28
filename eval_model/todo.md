ここでは、学習したモデルを評価することを行いたいと思います。
# 学習したモデルの使い方
run_deploy/run_exported_model.pyをそのまま使えます
MODEL_ID = "takumi0211/vav_grpo"
PROMPT_PATH = Path("data/test.md")
MAX_NEW_TOKENS = 4000
TEMPERATURE = 0.8
TOP_P = 0.95
DO_SAMPLE = True # True: サンプリング, False: 貪欲法
など

# 評価環境
eval_model/simulator-humidディレクトリに入っています。
具体的にはeval_model/simulator-humid/simulator_humid/agents/ollama_agent.pyの実行するところをエージェントを今回のものに置き換えるだけです。
但し、プロンプトはharmony形式にしてからモデルに与えてください。
それ以外の流れ、条件、csvとしてログとる、plotする。は全て同じでお願いします。

# 決まり
- eval_model/ディレクトリ以外は編集しないでください。
- eval_model/main.pyを実装することで評価ができるようにして。（結果の出力 csv pngもeval_model直下に保存されるように）

何かわからないことがあれば質問してください。