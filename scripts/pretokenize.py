"""Tokenize a HuggingFace text corpus into token shards (see myk_llm.data.shards).

Streams the dataset (no full download), applies the same document filter as streaming
training, and writes `<output>/train` and `<output>/val`. With `--workers N`, N processes
each read a disjoint 1/N of the stream and write their own shards into the same
directories.

    python scripts/pretokenize.py --preset en --target-tokens 5e9 --output data/shards/fineweb-edu-en --workers 8
    python scripts/pretokenize.py --preset zh --target-tokens 5e9 --output data/shards/fineweb-edu-zh --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "hf"))

from tokenizers import Tokenizer  # noqa: E402

from myk_llm.data.config import SourceConfig  # noqa: E402
from myk_llm.data.filters import DocFilter  # noqa: E402
from myk_llm.data.shards import ShardWriter, write_meta_file  # noqa: E402
from myk_llm.data.streaming import open_hf_stream  # noqa: E402

PRESETS = {
    "en": SourceConfig(name="en", hf_path="HuggingFaceFW/fineweb-edu", hf_config="sample-100BT", script="latin"),
    "zh": SourceConfig(
        name="zh", hf_path="opencsg/Fineweb-Edu-Chinese-V2.1", hf_data_dir="4_5", script="cjk", curated=True
    ),
}


def run_worker(args: argparse.Namespace, source: SourceConfig, worker: int) -> dict:
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    eos_id = tokenizer.token_to_id(args.eos_token)
    doc_filter = DocFilter(max_doc_tokens=args.max_doc_tokens)
    output = Path(args.output)
    prefix = f"w{worker:03d}"
    train = ShardWriter(output / "train", eos_id, args.tokenizer, int(args.shard_tokens), prefix)
    val = ShardWriter(output / "val", eos_id, args.tokenizer, int(args.shard_tokens), prefix)
    budget = int(args.target_tokens) // args.workers
    val_docs = args.val_docs // args.workers
    stream = iter(open_hf_stream(source, args.seed, args.shuffle_buffer, worker, args.workers))
    kept = dropped = 0
    started = last_report = time.time()
    while train.num_tokens < budget:
        texts = []
        for example in stream:
            text = example.get(source.text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            if not source.curated and doc_filter.reject_text(text, source.script):
                dropped += 1
                continue
            texts.append(text)
            if len(texts) == 256:
                break
        if not texts:
            break  # stream exhausted
        for encoding in tokenizer.encode_batch(texts, add_special_tokens=False):
            ids = encoding.ids[: doc_filter.max_doc_tokens] if doc_filter.max_doc_tokens else encoding.ids
            if not ids or (not source.curated and doc_filter.reject(ids)):
                dropped += 1
                continue
            kept += 1
            (val if kept <= val_docs else train).add(ids)
        if worker == 0 and time.time() - last_report > 30:
            last_report = time.time()
            rate = train.num_tokens / (last_report - started)
            print(f"[worker 0] {train.num_tokens / 1e6:,.0f}M / {budget / 1e6:,.0f}M tokens ({rate / 1e6:.2f}M tok/s)", flush=True)
    return {
        "train": train.close(write_meta=False),
        "val": val.close(write_meta=False),
        "eos_id": eos_id,
        "kept": kept,
        "dropped": dropped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--hf-path", default="", help="instead of --preset")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-data-dir", default=None)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--script", choices=["latin", "cjk"], default=None)
    parser.add_argument("--curated", action="store_true")
    parser.add_argument("--tokenizer", default="_download_models/Qwen3.5-0.8B-pruned64k")
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-tokens", type=float, required=True)
    parser.add_argument("--val-docs", type=int, default=2000)
    parser.add_argument("--shard-tokens", type=float, default=2**28)
    parser.add_argument("--max-doc-tokens", type=int, default=8192)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.preset:
        source = PRESETS[args.preset]
    elif args.hf_path:
        source = SourceConfig(
            name=Path(args.hf_path).name, hf_path=args.hf_path, hf_config=args.hf_config,
            hf_data_dir=args.hf_data_dir, text_key=args.text_key, script=args.script, curated=args.curated,
        )
    else:
        parser.error("pass --preset or --hf-path")

    if args.workers == 1:
        results = [run_worker(args, source, 0)]
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.starmap(run_worker, [(args, source, w) for w in range(args.workers)])

    info = {"source": vars(source), "kept_docs": sum(r["kept"] for r in results), "dropped_docs": sum(r["dropped"] for r in results)}
    for split in ("train", "val"):
        shards = [shard for r in results for shard in r[split]]
        write_meta_file(Path(args.output) / split, shards, results[0]["eos_id"], args.tokenizer, info)
        print(f"{split}: {sum(s['num_tokens'] for s in shards):,} tokens in {len(shards)} shard(s)")
    print(f"kept {info['kept_docs']:,} docs, dropped {info['dropped_docs']:,}")


if __name__ == "__main__":
    main()
