# build_qwen_dataset.py
from datasets import load_dataset
import json
import os
import re
from pathlib import Path

def extract_answer(text: str) -> str:
    """从 assistant 回答中提取 \boxed{答案} 里的字母/数字"""
    m = re.search(r'\\boxed\{{1,2}(.+?)\}{1,2}', text)
    return m.group(1) if m else ""

def convert_sample(raw: dict) -> dict:
    conversations = raw.get("conversations", [])
    user_turn   = next((c["value"] for c in conversations if c["from"] == "user"), "")
    assist_turn = next((c["value"] for c in conversations if c["from"] == "assistant"), "")
    answer = extract_answer(assist_turn)

    return {
        "reference": answer,
        "prompt": [
            {
                "content": "请用<think>你的思考过程</think>和<answer>最终答案</answer>的格式回答。其中<think>标签里填写尽量简短的思考过程并确保不要超过2000个字，最后用<answer>标签给出最终答案。",
                "role": "system"
            },
            {
                "content": user_turn,
                "role": "user"
            }
        ]
    }

def main(src_path: str, dst_path: str):
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        count = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                out = convert_sample(raw)
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                count += 1
                if count >= 20000:        # 只写 20000 条
                    break
            except Exception as e:
                print(f"跳过坏行：{e} | {line[:80]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",  default="/hy-tmp/RL/code/data/dianjin_data/qwen_finance.jsonl",  help="原始 JSONL 文件")
    parser.add_argument("-o", "--output", default="/hy-tmp/RL/code/data/dianjin_data/qwen_RL.jsonl", help="输出 JSONL 文件")
    args = parser.parse_args()
    main(args.input, args.output)