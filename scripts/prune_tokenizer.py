"""Prune the Qwen3.5 Byte Pair Encoding (BPE) tokenizer to a smaller bilingual vocab.

Qwen3.5 ships a ~248K-token byte-level BPE vocab, of which roughly a fifth is
languages we do not want (Cyrillic, Arabic, Hangul, Kana, ...). For an
English + Chinese model we keep only the tokens that actually earn their place
on a representative corpus, plus the pieces required to keep the tokenizer
*complete* (every byte still encodable) and *consistent* (every kept token still
reachable through surviving merges).

What the script produces, under --output:
  - tokenizer.json / tokenizer_config.json / chat_template.jinja  (a loadable
    pruned tokenizer that reuses Qwen's exact byte-level pipeline)
  - embedding_row_map.npy  (int64 array; row_map[new_id] = old_id, so the model
    can initialize its embedding by gathering the surviving rows out of Qwen's
    matrix: new_embedding = qwen_embedding[row_map])
  - prune_report.json  (what was kept/dropped, by Unicode script)

Selection is frequency-driven on the supplied corpus. Naive "keep the first N
ids" does NOT work here: Qwen's Chinese block lives at ids ~99K-148K, so id
truncation would yield an almost English-only vocab. Always pass a corpus that
reflects the real English/Chinese mix.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer


# ----------------------------------------------------------------------------
# Byte-level helpers + Unicode-script classification (for the report only).
# ----------------------------------------------------------------------------

def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 / Qwen reversible byte-to-unicode map used by byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def script_of_char(ch: str) -> str:
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF:
        return "Han"
    if 0x3040 <= cp <= 0x30FF:
        return "Kana"
    if 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF:
        return "Hangul"
    if 0x0400 <= cp <= 0x04FF:
        return "Cyrillic"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        return "Arabic"
    if (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A) or (0xC0 <= cp <= 0x24F):
        return "Latin"
    if ch.isdigit():
        return "Digit"
    cat = unicodedata.category(ch)
    if cat.startswith("P") or cat.startswith("S"):
        return "Punct/Sym"
    if ch.isspace():
        return "Space"
    return "Other"


def token_script(text: str) -> str:
    counts: Counter[str] = Counter()
    for ch in text:
        s = script_of_char(ch)
        if s != "Space":
            counts[s] += 1
    return counts.most_common(1)[0][0] if counts else "Space"


def decode_token(tok: str, u2b: dict[str, int]) -> str | None:
    """Recover the real text a byte-level token stands for, or None if it is a
    partial multi-byte fragment that does not decode on its own."""
    try:
        return bytes(u2b[c] for c in tok).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


# ----------------------------------------------------------------------------
# Merge graph: closure so every kept token stays reachable.
# ----------------------------------------------------------------------------

def split_merge(merge: str) -> tuple[str, str]:
    # Byte-level tokens never contain a raw space, so the single separating
    # space splits the rule cleanly into its two parents.
    idx = merge.index(" ")
    return merge[:idx], merge[idx + 1 :]


def build_merge_parents(merges: list[str]) -> dict[str, tuple[str, str]]:
    parents: dict[str, tuple[str, str]] = {}
    for merge in merges:
        left, right = split_merge(merge)
        parents[left + right] = (left, right)
    return parents


def closure_of(token: str, parents: dict[str, tuple[str, str]], memo: dict[str, frozenset[str]]) -> frozenset[str]:
    """All tokens required to be able to produce `token`: itself plus, recursively,
    the two parents of every merge in its derivation."""
    cached = memo.get(token)
    if cached is not None:
        return cached
    result: set[str] = set()
    stack = [token]
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        pair = parents.get(current)
        if pair is not None:
            stack.append(pair[0])
            stack.append(pair[1])
    frozen = frozenset(result)
    memo[token] = frozen
    return frozen


# ----------------------------------------------------------------------------
# Corpus frequency counting.
# ----------------------------------------------------------------------------

def iter_line_batches(paths: list[Path], batch_lines: int, max_bytes: int) -> Iterable[list[str]]:
    for path in paths:
        read_bytes = 0
        batch: list[str] = []
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                batch.append(line)
                read_bytes += len(line.encode("utf-8"))
                if len(batch) >= batch_lines:
                    yield batch
                    batch = []
                if max_bytes and read_bytes >= max_bytes:
                    break
        if batch:
            yield batch


def count_token_frequencies(tokenizer: Tokenizer, paths: list[Path], vocab_size: int, batch_lines: int, max_bytes: int) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=np.int64)
    total_lines = 0
    for batch in iter_line_batches(paths, batch_lines, max_bytes):
        encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
        ids: list[int] = []
        for enc in encodings:
            ids.extend(enc.ids)
        if ids:
            counts += np.bincount(np.asarray(ids, dtype=np.int64), minlength=vocab_size)
        total_lines += len(batch)
        print(f"\r  counted {total_lines} lines, {int(counts.sum())} tokens", end="", file=sys.stderr)
    print(file=sys.stderr)
    return counts


# ----------------------------------------------------------------------------
# Pruning.
# ----------------------------------------------------------------------------

def select_base_tokens(
    base_vocab: dict[str, int],
    counts: np.ndarray,
    parents: dict[str, tuple[str, str]],
    reserved: set[str],
    base_budget: int,
) -> set[str]:
    """Greedily keep the highest-frequency tokens (with their merge closure) until
    the budget is full. Starting from `reserved` (the 256 byte tokens) guarantees
    encodability; pulling in each token's closure guarantees reachability."""
    id_to_token = {tid: tok for tok, tid in base_vocab.items()}
    keep: set[str] = set(reserved)
    memo: dict[str, frozenset[str]] = {}

    # Highest frequency first; ties and zero-frequency tokens fall to the end.
    order = np.argsort(counts, kind="stable")[::-1]
    skipped_budget = 0
    for tid in order:
        count = counts[tid]
        token = id_to_token.get(int(tid))
        if token is None or token in keep:
            continue
        if count == 0:
            break  # nothing below this point appeared in the corpus
        needed = closure_of(token, parents, memo) - keep
        if len(keep) + len(needed) <= base_budget:
            keep |= needed
        else:
            skipped_budget += 1
    if skipped_budget:
        print(f"  budget full: {skipped_budget} in-corpus tokens dropped (closure did not fit)", file=sys.stderr)
    return keep


def filter_merges(merges: list[str], keep: set[str]) -> list[str]:
    kept: list[str] = []
    for merge in merges:
        left, right = split_merge(merge)
        if left in keep and right in keep and (left + right) in keep:
            kept.append(merge)
    return kept


def build_report(base_vocab: dict[str, int], keep: set[str], u2b: dict[str, int]) -> dict[str, object]:
    kept_by_script: Counter[str] = Counter()
    dropped_by_script: Counter[str] = Counter()
    for tok in base_vocab:
        text = decode_token(tok, u2b)
        script = "partial-bytes" if text is None else token_script(text)
        if tok in keep:
            kept_by_script[script] += 1
        else:
            dropped_by_script[script] += 1
    return {
        "kept_by_script": dict(kept_by_script.most_common()),
        "dropped_by_script": dict(dropped_by_script.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=str, default="_download_models/Qwen3.5-0.8B")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--corpus", type=str, nargs="+", required=True, help="text files for frequency counting")
    parser.add_argument("--target-vocab", type=int, default=65536, help="total rows incl. special tokens")
    parser.add_argument("--batch-lines", type=int, default=10000)
    parser.add_argument("--max-bytes-per-corpus", type=int, default=0, help="0 = read the whole file")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    corpus_paths = [Path(p) for p in args.corpus]
    for path in corpus_paths:
        if not path.exists():
            raise FileNotFoundError(f"corpus file not found: {path}")

    tokenizer_json = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    model = tokenizer_json["model"]
    base_vocab: dict[str, int] = model["vocab"]  # token -> id, ids 0..N-1
    merges: list[str] = model["merges"]

    # Special / added tokens: union of the fast list and the config decoder, keyed
    # by old id. These always survive and occupy embedding rows above the base vocab.
    added: dict[int, dict[str, object]] = {}
    for entry in tokenizer_json.get("added_tokens", []):
        added[int(entry["id"])] = dict(entry)
    for sid, entry in tokenizer_config.get("added_tokens_decoder", {}).items():
        added.setdefault(int(sid), {"id": int(sid), **entry})

    u2b = {u: b for b, u in bytes_to_unicode().items()}
    byte_tokens = {u for u in u2b if u in base_vocab}  # the 256 single-byte tokens
    if len(byte_tokens) != 256:
        raise ValueError(f"expected 256 byte-level tokens in vocab, found {len(byte_tokens)}")

    base_budget = args.target_vocab - len(added)
    if base_budget <= len(byte_tokens):
        raise ValueError(f"target-vocab {args.target_vocab} too small for {len(added)} specials + 256 bytes")
    print(f"source vocab={len(base_vocab)} merges={len(merges)} added={len(added)} -> target={args.target_vocab} (base budget {base_budget})", file=sys.stderr)

    print("counting token frequencies on corpus...", file=sys.stderr)
    counter = Tokenizer.from_file(str(source / "tokenizer.json"))
    # Size the histogram to cover every id, including special tokens that raw text
    # may contain literally (e.g. "<|im_end|>"), so bincount never overruns.
    counts_len = max(max(base_vocab.values(), default=0), max(added, default=0)) + 1
    counts = count_token_frequencies(counter, corpus_paths, counts_len, args.batch_lines, args.max_bytes_per_corpus)

    parents = build_merge_parents(merges)
    keep = select_base_tokens(base_vocab, counts, parents, byte_tokens, base_budget)
    kept_merges = filter_merges(merges, keep)
    print(f"kept base tokens={len(keep)} kept merges={len(kept_merges)}", file=sys.stderr)

    # Reassign ids: base tokens 0..K-1 (ordered by original id for a stable map),
    # then the special tokens K..K+A-1. row_map lets us gather Qwen's rows.
    base_sorted = sorted(keep, key=lambda tok: base_vocab[tok])
    new_base_vocab = {tok: new_id for new_id, tok in enumerate(base_sorted)}
    added_sorted = sorted(added.values(), key=lambda e: int(e["id"]))
    new_vocab_size = len(base_sorted) + len(added_sorted)
    row_map = np.empty(new_vocab_size, dtype=np.int64)
    for tok, new_id in new_base_vocab.items():
        row_map[new_id] = base_vocab[tok]
    new_added_tokens: list[dict[str, object]] = []
    for offset, entry in enumerate(added_sorted):
        new_id = len(base_sorted) + offset
        row_map[new_id] = int(entry["id"])
        remapped = dict(entry)
        remapped["id"] = new_id
        new_added_tokens.append(remapped)

    # Write the pruned tokenizer, reusing Qwen's normalizer/pre_tokenizer/decoder.
    output.mkdir(parents=True, exist_ok=True)
    model["vocab"] = new_base_vocab
    model["merges"] = kept_merges
    tokenizer_json["added_tokens"] = new_added_tokens
    (output / "tokenizer.json").write_text(json.dumps(tokenizer_json, ensure_ascii=False), encoding="utf-8")

    tokenizer_config["added_tokens_decoder"] = {
        str(entry["id"]): {k: v for k, v in entry.items() if k != "id"} for entry in new_added_tokens
    }
    (output / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, ensure_ascii=False, indent=2), encoding="utf-8")
    for aux in ("chat_template.jinja", "special_tokens_map.json"):
        if (source / aux).exists():
            shutil.copy(source / aux, output / aux)

    np.save(output / "embedding_row_map.npy", row_map)
    report = build_report(base_vocab, keep, u2b)
    report.update({
        "source": str(source),
        "target_vocab": args.target_vocab,
        "new_vocab_size": new_vocab_size,
        "base_tokens": len(base_sorted),
        "added_tokens": len(added_sorted),
        "kept_merges": len(kept_merges),
        "corpus_tokens_counted": int(counts.sum()),
    })
    (output / "prune_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    verify(source, output, row_map)
    print(f"\nwrote pruned tokenizer to {output} (vocab {new_vocab_size}); see prune_report.json", file=sys.stderr)


def verify(source: Path, output: Path, row_map: np.ndarray) -> None:
    """Round-trip a few strings and confirm compression versus the original."""
    original = Tokenizer.from_file(str(source / "tokenizer.json"))
    pruned = Tokenizer.from_file(str(output / "tokenizer.json"))
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "今天天气真好，我们去公园散步吧。",  # Chinese
        "Mixing 中文 and English in 一个 sentence 123!",
        "def f(x): return x ** 2  # rare-ish symbols — éàü",
    ]
    vocab_size = len(row_map)
    orig_total = 0
    pruned_total = 0
    for text in samples:
        enc = pruned.encode(text, add_special_tokens=False)
        if any(tid >= vocab_size for tid in enc.ids):
            raise AssertionError(f"out-of-range id encoding: {text!r}")
        decoded = pruned.decode(enc.ids)
        if decoded != text:
            raise AssertionError(f"round-trip mismatch:\n  in : {text!r}\n  out: {decoded!r}")
        orig_total += len(original.encode(text, add_special_tokens=False).ids)
        pruned_total += len(enc.ids)
    print(f"verify ok: round-trip exact; tokens orig={orig_total} pruned={pruned_total} (+{100*(pruned_total-orig_total)/orig_total:.1f}% on sample)", file=sys.stderr)


if __name__ == "__main__":
    main()
