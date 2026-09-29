"""Tokenize chat JSONL / JSON into a packed, loss-masked SFT dataset (myk_llm.data.sft).

    python scripts/tokenize_sft.py --data data/sft/train.jsonl --output data/sft/train
    python scripts/tokenize_sft.py --data data/sft/val.jsonl --output data/sft/val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm
from transformers import AutoTokenizer

from myk_llm.data.sft import write_chat_dataset


def read_examples(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help=".jsonl (one conversation per line) or .json (a list)")
    parser.add_argument("--output", required=True, help="output prefix")
    parser.add_argument("--tokenizer", default="_download_models/Qwen3.5-0.8B-pruned64k")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    examples = tqdm(read_examples(Path(args.data)), desc="tokenizing")
    meta = write_chat_dataset(tokenizer, examples, args.output, args.tokenizer)
    print(f"wrote {meta['num_tokens']:,} tokens in {meta['num_spans']:,} conversations "
          f"(skipped {meta['skipped']} without assistant tokens) to {args.output}.*")


if __name__ == "__main__":
    main()
