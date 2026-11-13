"""Synthetic long-context retrieval tasks over a small token vocabulary.

An example is one sequence: a context of records scattered among noise tokens, then
queries. Each of N *items* is a chain of h hops through distinct keys; h = 1 is a plain
key -> value record (associative recall), h > 1 is variable tracking:

  records   k0 k1 ;  k1 k2 ;  k2 v v v v ;     (h = 3; keys are key_len tokens)
  query     ? k0   answer   v v v v $           (cot: k1 k2 v v v v $)

Queries end with the key itself, so a plain induction head (find the earlier occurrence
of the current tokens, copy what follows) answers them. With a separator between key and
value (`? k >`), models plateaued at chance for 8000+ steps: the current token no longer
identifies the record. Keys, values and noise use disjoint alphabets, and record parts
have fixed lengths, so records stay unambiguous without separators.

All records of all items are shuffled together and placed at random positions among
noise tokens; Q of the N items are queried at the end. The loss covers only the answers
(incl. the EOS `$`). Regimes (see notes/lc-retrieval.md):

  needle    N small (>= Q), noise fills the rest: finding a few records far away
  capacity  N large, little or no noise: holding many associations at once

With `scale_records`, N is the count at length `max_len` and scales with the length, so
evaluating longer contexts keeps the record density (load) instead of the count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

PAD, EOS, QUERY, SEP = 0, 1, 2, 3
KEY_BASE, KEY_SYMBOLS = 16, 128
VALUE_BASE, VALUE_SYMBOLS = KEY_BASE + KEY_SYMBOLS, 64
NOISE_BASE, NOISE_SYMBOLS = VALUE_BASE + VALUE_SYMBOLS, 256
VOCAB_SIZE = NOISE_BASE + NOISE_SYMBOLS  # 464


@dataclass
class TaskConfig:
    # Training example lengths (tokens, queries included) lie in [min_len, max_len]; max_len
    # is also the packed row length. log_uniform puts as many examples in [128, 256] as in
    # [1024, 2048]: short, dense examples are what teaches key matching quickly.
    min_len: int = 256
    max_len: int = 2048
    length_dist: Literal["uniform", "log_uniform"] = "uniform"
    num_records: int = 8  # items (chains) per example; at max_len when scale_records
    # Training draws each example's item count uniformly from [min_records, num_records]
    # (scaled alike), with queries capped at that count; evaluation always uses
    # num_records. Retrieval is learned through an abrupt transition after a plateau that
    # grows with the load; easy examples (1-2 items) in the mix shorten it. 0: no mix.
    min_records: int = 0
    # Fraction of training examples that are copy examples instead (a random span of
    # copy_min..copy_max tokens, then the same span again, loss on the repeat). Their
    # dense loss forms the induction circuit that retrieval relies on; with answer-only
    # loss alone it can take far longer than a run (notes/lc-retrieval.md, calibration).
    copy_fraction: float = 0.0
    copy_min: int = 16
    copy_max: int = 128
    scale_records: bool = False
    num_queries: int = 4
    key_len: int = 2
    value_len: int = 4
    hops: list[int] = field(default_factory=lambda: [1])  # each item's hop count is drawn from these
    # Hop counts of lc.train's final scan (empty: `hops`); counts above max(hops) measure
    # hop extrapolation, e.g. of looped models given more iterations.
    eval_hops: list[int] = field(default_factory=list)
    cot: bool = False  # answers spell out the intermediate keys of a chain
    eval_lengths: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    depth_buckets: int = 10

    def __post_init__(self) -> None:
        if not 0 < self.min_len <= self.max_len:
            raise ValueError("need 0 < min_len <= max_len")
        if self.num_queries > self.num_records:
            raise ValueError("num_queries must be <= num_records (queried items are distinct)")
        if not 0 <= self.min_records <= self.num_records:
            raise ValueError("need 0 <= min_records <= num_records")
        if self.copy_fraction and not 1 <= self.copy_min <= self.copy_max <= self.max_len // 2:
            raise ValueError("need 1 <= copy_min <= copy_max <= max_len / 2")
        if not self.hops or min(self.hops) < 1:
            raise ValueError("hops must be a non-empty list of positive counts")
        if self.eval_hops and min(self.eval_hops) < 1:
            raise ValueError("eval_hops must be positive counts")

    def sample_length(self, rng: np.random.Generator) -> int:
        if self.length_dist == "uniform":
            return int(rng.integers(self.min_len, self.max_len + 1))
        return int(np.exp(rng.uniform(np.log(self.min_len), np.log(self.max_len + 1))))

    def records_at(self, length: int, count: int | None = None) -> int:
        count = self.num_records if count is None else count
        if not self.scale_records:
            return count
        return max(1, round(count * length / self.max_len))

    def sample_records(self, rng: np.random.Generator) -> int | None:
        return int(rng.integers(self.min_records, self.num_records + 1)) if self.min_records else None


@dataclass
class Example:
    ids: np.ndarray  # int64 (length,)
    answers: list[tuple[int, int]]  # [start, end) of each query's answer tokens in ids
    depths: list[float]  # position of the queried item's first record / context length
    hops: list[int]


def make_example(rng: np.random.Generator, task: TaskConfig, length: int, records: int | None = None) -> Example:
    """One example of exactly `length` tokens with `records` items (default num_records;
    both scaled with the length when scale_records)."""
    kl = task.key_len
    n = task.records_at(length, records)
    q = min(task.num_queries, n)
    hops = rng.choice(task.hops, size=n)
    total_keys = int(hops.sum())
    if total_keys > KEY_SYMBOLS**kl:
        raise ValueError(f"{total_keys} keys needed, only {KEY_SYMBOLS ** kl} exist; raise key_len")
    # Distinct keys: distinct integers in [0, KEY_SYMBOLS^kl), written in base KEY_SYMBOLS.
    codes = rng.choice(KEY_SYMBOLS**kl, size=total_keys, replace=False)
    digits = (codes[:, None] // KEY_SYMBOLS ** np.arange(kl - 1, -1, -1)) % KEY_SYMBOLS + KEY_BASE
    keys = [list(row) for row in digits.tolist()]
    values = (rng.integers(0, VALUE_SYMBOLS, size=(n, task.value_len)) + VALUE_BASE).tolist()

    records: list[tuple[int, list[int]]] = []  # (item, tokens)
    chains: list[list[list[int]]] = []
    offset = 0
    for item, h in enumerate(hops.tolist()):
        chain = keys[offset : offset + h]
        offset += h
        chains.append(chain)
        for a, b in zip(chain, chain[1:]):
            records.append((item, [*a, *b, SEP]))
        records.append((item, [*chain[-1], *values[item], SEP]))

    queried = rng.choice(n, size=q, replace=False).tolist()
    queries = []
    for item in queried:
        chain = chains[item]
        answer = [t for key in chain[1:] for t in key] if task.cot else []
        queries.append(([QUERY, *chain[0]], [*answer, *values[item], EOS]))
    query_tokens = sum(len(p) + len(a) for p, a in queries)
    context = length - query_tokens
    record_tokens = sum(len(r) for _, r in records)
    noise = context - record_tokens
    if noise < 0:
        raise ValueError(f"{n} items ({record_tokens} record tokens) + queries do not fit in {length} tokens")

    order = rng.permutation(len(records))
    cuts = np.sort(rng.integers(0, noise + 1, size=len(records)))  # noise before each record
    noise_ids = rng.integers(NOISE_BASE, NOISE_BASE + NOISE_SYMBOLS, size=noise)
    ids = np.empty(length, dtype=np.int64)
    first_pos = [context] * n  # position of each item's earliest record
    pos = prev = 0
    for index, cut in zip(order.tolist(), cuts.tolist()):
        ids[pos : pos + cut - prev] = noise_ids[prev:cut]
        pos += cut - prev
        prev = cut
        item, tokens = records[index]
        first_pos[item] = min(first_pos[item], pos)
        ids[pos : pos + len(tokens)] = tokens
        pos += len(tokens)
    ids[pos:context] = noise_ids[prev:]
    pos = context

    answers, depths = [], []
    for (prompt, answer), item in zip(queries, queried):
        ids[pos : pos + len(prompt)] = prompt
        pos += len(prompt)
        ids[pos : pos + len(answer)] = answer
        answers.append((pos, pos + len(answer)))
        pos += len(answer)
        depths.append(first_pos[item] / context)
    return Example(ids, answers, depths, [int(hops[i]) for i in queried])


def make_copy_example(rng: np.random.Generator, task: TaskConfig) -> Example:
    """A random span over all content alphabets followed by itself; the repeat (minus its
    first token, which nothing predicts) is the answer."""
    span = rng.integers(KEY_BASE, VOCAB_SIZE, size=int(rng.integers(task.copy_min, task.copy_max + 1)))
    ids = np.concatenate([span, span])
    return Example(ids, [(len(span) + 1, len(ids))], [0.0], [0])


def resolve(ids: np.ndarray, query_key: list[int], key_len: int) -> list[int]:
    """Follow records from `query_key` to its value by scanning the context (reference
    implementation for tests)."""
    tokens = ids.tolist()
    tokens = tokens[: tokens.index(QUERY)]  # the context ends at the first query
    table: dict[tuple[int, ...], list[int]] = {}
    i = 0
    while i < len(tokens):
        if KEY_BASE <= tokens[i] < VALUE_BASE and (i == 0 or tokens[i - 1] == SEP or tokens[i - 1] >= NOISE_BASE):
            end = tokens.index(SEP, i)  # a record: key, then a key or a value, then SEP
            table[tuple(tokens[i : i + key_len])] = tokens[i + key_len : end]
            i = end + 1
        else:
            i += 1
    target = table[tuple(query_key)]
    while all(KEY_BASE <= t < VALUE_BASE for t in target):
        target = table[tuple(target)]
    return target
