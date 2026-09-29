"""Build a bilingual (en+zh) SFT dataset for MykLLM as normalized chat JSONL.

Pulls a few instruction-tuning sources chosen to fit a small (~200M) model — clean
instruction-following, not reasoning-heavy — and normalizes every row to the
`{"messages": [{"role", "content"}, ...]}` shape that `scripts/tokenize_sft.py`
consumes.

Default sources:
  - HuggingFaceTB/smoltalk            (English; instruction subsets only — the
                                       math/CoT/long-context subsets are dropped as
                                       too hard for a ~200M model)
  - m-a-p/COIG-CQIA                   (Chinese, native high-quality instruction)
  - opencsg/Fineweb-Edu-Chinese-V2.2 (Chinese, native synthetic QA, volume)

Each source is streamed (no full local copy), capped at --max-per-source, converted,
lightly filtered, and deduped. The union is shuffled with a fixed seed and split into
train/val JSONL. Weighting is by cap: raise the zh caps to match the ~0.4 en / 0.6 zh
pretrain ratio.

Example:
  python scripts/build_sft_data.py --out-dir data/sft \
      --max-per-source 200000 --val-frac 0.01

Then:
  python scripts/tokenize_sft.py --tokenizer _download_models/Qwen3.5-0.8B-pruned64k \
      --data data/sft/train.jsonl --output data/sft/train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault(
    "HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "hf")
)

# ShareGPT-style role aliases; keep in sync with tokenization.normalize_chat_messages.
_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "system": "system",
}


@dataclass
class Source:
    """One HF instruction dataset and how to reach/convert it."""

    name: str
    path: str
    lang: str  # "en" | "zh" (informational)
    configs: list[str | None] = field(default_factory=lambda: [None])
    split: str = "train"
    cap: int | None = None  # per-source override of --max-per-source
    # Per-source cap on total assistant-turn characters. Used to keep sources whose
    # answers are long/essay-like (e.g. the synthetic Fineweb-Edu-Chinese QA) from
    # teaching a ~200M model a verbose register it can't sustain. None = no extra cap.
    max_assistant_chars: int | None = None


DEFAULT_SOURCES: list[Source] = [
    # smoltalk ships as per-task subsets plus an "all" superset. We enumerate the
    # clean instruction-following subsets and deliberately OMIT the ones a ~200M model
    # can't learn from: metamathqa-50k + numina-cot-100k (multi-step math CoT it only
    # learns to fake), longalign (>seq_len), apigen-80k (tool-call JSON) and
    # self-oss-instruct (Python). Enumerating also avoids the "all"-superset trap where
    # a --max-per-source cap silently takes only the first shard's subset.
    Source(
        "smoltalk",
        "HuggingFaceTB/smoltalk",
        "en",
        configs=[
            "smol-magpie-ultra", "smol-summarize", "smol-rewrite", "smol-constraints",
            "everyday-conversations", "systemchats-30k", "openhermes-100k",
            "explore-instruct-rewriting",
        ],
    ),
    # COIG-CQIA ships as many task subsets; pull a broad, clean spread. Unknown/failed
    # configs are skipped with a warning so the script survives schema drift.
    Source(
        "coig-cqia",
        "m-a-p/COIG-CQIA",
        "zh",
        configs=[
            "coig_pc", "exam", "human_value", "logi_qa", "ruozhiba",
            "wiki", "wikihow", "zhihu", "xhs", "douban", "finance", "segmentfault",
        ],
    ),
    # Native synthetic Chinese QA (DeepSeek-V3.2, grounded in top-0.1% edu web text).
    # Replaces the noisy machine-translated Belle. Answers are long/structured, so we
    # cap assistant length to keep the verbose register from swamping COIG-CQIA's
    # shorter, more varied instructions. `default`/`sft_qa` is the pure instruction-
    # output split (sft/cleaned/*.jsonl); Alpaca instruction/input/output shape.
    Source(
        "finewebedu-zh",
        "opencsg/Fineweb-Edu-Chinese-V2.2",
        "zh",
        configs=["default"],
        split="sft_qa",
        max_assistant_chars=1200,
    ),
]


def _stringify(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    """Normalize one dataset row to [{role, content}, ...] or None if unusable.

    Handles three common shapes: pre-formatted `messages`, ShareGPT `conversations`
    (from/value), and Alpaca-style `instruction`(+`input`)/`output`.
    """
    raw = row.get("messages") or row.get("conversations")
    if isinstance(raw, list) and raw:
        messages: list[dict[str, str]] = []
        for message in raw:
            if not isinstance(message, dict):
                return None
            role = message.get("role", message.get("from"))
            content = message.get("content", message.get("value"))
            if not isinstance(role, str) or not _stringify(content):
                return None
            messages.append({"role": _ROLE_ALIASES.get(role.lower(), role), "content": content.strip()})
    else:
        # Alpaca style.
        instruction = _stringify(row.get("instruction"))
        extra = _stringify(row.get("input"))
        output = _stringify(row.get("output") or row.get("response") or row.get("answer"))
        if not instruction or not output:
            return None
        user = f"{instruction}\n\n{extra}" if extra else instruction
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": output}]

    # Require at least one user->assistant exchange with a trainable assistant turn.
    if not any(m["role"] == "assistant" for m in messages):
        return None
    if messages[0]["role"] == "system" and len(messages) < 2:
        return None
    return messages


def _passes_filters(
    messages: list[dict[str, str]],
    min_chars: int,
    max_chars: int,
    max_assistant_chars: int | None = None,
) -> bool:
    total = sum(len(m["content"]) for m in messages)
    if total < min_chars or total > max_chars:
        return False
    assistant_chars = sum(len(m["content"]) for m in messages if m["role"] == "assistant")
    if max_assistant_chars is not None and assistant_chars > max_assistant_chars:
        return False
    # Drop turns that are empty after normalization (already stripped) or degenerate.
    for m in messages:
        if m["role"] == "assistant" and len(m["content"]) < 1:
            return False
    return True


def _dedup_key(messages: list[dict[str, str]]) -> str:
    joined = "␟".join(f"{m['role']}:{m['content']}" for m in messages)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def stream_source(source: Source, cap: int, min_chars: int, max_chars: int) -> Iterator[list[dict[str, str]]]:
    from datasets import load_dataset

    produced = 0
    for config in source.configs:
        if produced >= cap:
            break
        try:
            dataset = load_dataset(source.path, name=config, split=source.split, streaming=True)
        except Exception as exc:  # noqa: BLE001 - survive gated/missing configs
            print(f"  [warn] {source.name}:{config} load failed ({type(exc).__name__}: {exc}); skipping")
            continue
        for row in dataset:
            if produced >= cap:
                break
            messages = row_to_messages(dict(row))
            if messages is None or not _passes_filters(
                messages, min_chars, max_chars, source.max_assistant_chars
            ):
                continue
            produced += 1
            yield messages
    print(f"  {source.name}: yielded {produced} conversations")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-per-source", type=int, default=200_000)
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=8000, help="drop very long conversations (small model)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", type=str, default="", help="comma-separated source names to include (default: all)")
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    sources = [s for s in DEFAULT_SOURCES if not only or s.name in only]
    if not sources:
        raise SystemExit(f"no sources selected; known: {[s.name for s in DEFAULT_SOURCES]}")

    seen: set[str] = set()
    conversations: list[list[dict[str, str]]] = []
    per_lang: dict[str, int] = {}
    for source in sources:
        cap = source.cap if source.cap is not None else args.max_per_source
        print(f"streaming {source.name} ({source.path}) cap={cap} ...")
        for messages in stream_source(source, cap, args.min_chars, args.max_chars):
            key = _dedup_key(messages)
            if key in seen:
                continue
            seen.add(key)
            conversations.append(messages)
            per_lang[source.lang] = per_lang.get(source.lang, 0) + 1

    if not conversations:
        raise SystemExit("no conversations produced; check network/HF access and source configs")

    rng = random.Random(args.seed)
    rng.shuffle(conversations)
    num_val = max(1, int(len(conversations) * args.val_frac)) if args.val_frac > 0 else 0
    val = conversations[:num_val]
    train = conversations[num_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for messages in rows:
                handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} -> {path}")
    print(f"totals: {len(conversations)} conversations, by lang: {per_lang}")


if __name__ == "__main__":
    main()
