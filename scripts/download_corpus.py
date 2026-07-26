"""Stream a token-budgeted text corpus from HuggingFace datasets.

Pulls documents from a streaming dataset (no full download), tokenizes with the
Qwen tokenizer to count real tokens, and writes raw text until a target token
count is hit. Used to assemble the English + Chinese sample that drives
scripts/prune_tokenizer.py.

Presets (50/50 English/Chinese, edu-filtered web — the small-model quality mix):
  --lang en  ->  HuggingFaceFW/fineweb-edu        (config sample-10BT)
  --lang zh  ->  opencsg/chinese-fineweb-edu-v2

Example:
  python scripts/download_corpus.py --lang en --target-tokens 50_000_000 --output data/corpus_en.txt
  python scripts/download_corpus.py --lang zh --target-tokens 50_000_000 --output data/corpus_zh.txt
then prune on both:
  python scripts/prune_tokenizer.py --output ... --corpus data/corpus_en.txt data/corpus_zh.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep all HuggingFace downloads inside the project tree.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".cache" / "hf"))

from datasets import load_dataset
from tokenizers import Tokenizer


DEFAULT_TOKENIZER = "/home/miyuki/repos/myk-llm/_download_models/Qwen3.5-0.8B/tokenizer.json"

PRESETS: dict[str, dict[str, str]] = {
    "en": {"path": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "text_key": "text"},
    "zh": {"path": "opencsg/chinese-fineweb-edu-v2", "config": "", "text_key": "text"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--lang", type=str, choices=sorted(PRESETS), default=None)
    parser.add_argument("--dataset", type=str, default="", help="overrides --lang preset path")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--min-doc-tokens", type=int, default=32, help="skip very short documents")
    parser.add_argument("--report-every", type=int, default=2_000_000)
    args = parser.parse_args()

    if args.dataset:
        path, config, text_key = args.dataset, (args.config or None), args.text_key
    elif args.lang:
        preset = PRESETS[args.lang]
        path, config, text_key = preset["path"], (preset["config"] or None), preset["text_key"]
    else:
        raise SystemExit("provide --lang or --dataset")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"streaming {path}{f'/{config}' if config else ''} split={args.split} -> {output} (target {args.target_tokens:,} tokens)", file=sys.stderr)
    dataset = load_dataset(path, name=config, split=args.split, streaming=True)

    total_tokens = 0
    total_docs = 0
    next_report = args.report_every
    with output.open("w", encoding="utf-8") as handle:
        for example in dataset:
            text = example.get(text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            n = len(tokenizer.encode(text, add_special_tokens=False).ids)
            if n < args.min_doc_tokens:
                continue
            handle.write(text)
            handle.write("\n")
            total_tokens += n
            total_docs += 1
            if total_tokens >= next_report:
                print(f"\r  {total_tokens:,} tokens / {total_docs:,} docs", end="", file=sys.stderr)
                next_report += args.report_every
            if total_tokens >= args.target_tokens:
                break

    print(f"\ndone: {total_tokens:,} tokens across {total_docs:,} docs -> {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
