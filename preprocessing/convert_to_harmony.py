import argparse
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from transformers import AutoTokenizer

MODEL_ID = "openai/gpt-oss-20b"
_SYS_RE = re.compile(r"^\s*system:\s*(.*)$", re.IGNORECASE)
_USER_RE = re.compile(r"^\s*user:\s*(.*)$", re.IGNORECASE)


def legacy_text_to_messages(text: str):
    system_parts: List[str] = []
    user_parts: List[str] = []
    role = None
    for line in text.splitlines():
        m_sys = _SYS_RE.match(line)
        m_user = _USER_RE.match(line)
        if m_sys:
            role = "system"
            system_parts.append(m_sys.group(1).strip())
            continue
        if m_user:
            role = "user"
            user_parts.append(m_user.group(1).strip())
            continue
        if role == "user":
            user_parts.append(line)
        elif role == "system":
            system_parts.append(line)
        else:
            user_parts.append(line)
    messages = []
    sys_txt = "\n".join(system_parts).strip()
    if sys_txt:
        messages.append({"role": "system", "content": sys_txt})
    usr_txt = "\n".join(user_parts).strip()
    if usr_txt:
        messages.append({"role": "user", "content": usr_txt})
    else:
        messages.append({"role": "user", "content": text.strip()})
    return messages


def render_harmony_prompt(prompt: str, tok: AutoTokenizer) -> str:
    messages = legacy_text_to_messages(prompt)
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def convert_file(path: Path, out_path: Path, tok: AutoTokenizer, overwrite: bool = False):
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists. Pass --overwrite to replace it.")
    df = pd.read_csv(path)
    if "prompt" not in df.columns:
        raise ValueError(f"{path} does not have a 'prompt' column")
    df = df.copy()
    # Convert prompts, handling NaN values
    df["prompt"] = [
        render_harmony_prompt(p, tok) if pd.notna(p) else p 
        for p in df["prompt"]
    ]
    df.to_csv(out_path, index=False)
    return len(df)


def iter_csv_paths(targets: Iterable[Path]):
    for target in targets:
        if target.is_dir():
            for csv_path in sorted(target.rglob("*.csv")):
                yield csv_path
        elif target.suffix.lower() == ".csv":
            yield target
        else:
            raise ValueError(f"Unsupported path (expect CSV or dir): {target}")


def main():
    parser = argparse.ArgumentParser(description="Convert TES prompts to Harmony chat format")
    parser.add_argument("paths", nargs="+", type=Path, help="CSV files or directories to convert")
    parser.add_argument("--suffix", default="_harmony", help="Suffix inserted before .csv")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for csv_path in iter_csv_paths(args.paths):
        out_name = csv_path.stem + args.suffix + csv_path.suffix
        out_path = csv_path.with_name(out_name)
        rows = convert_file(csv_path, out_path, tok, overwrite=args.overwrite)
        print(f"Converted {csv_path} -> {out_path} ({rows} rows)")


if __name__ == "__main__":
    main()
