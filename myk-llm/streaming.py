"""On-the-fly streaming corpus for pretraining (no full local copy on disk).

Two or more HuggingFace streaming datasets are interleaved by a token budget,
tokenized on the fly with the pruned tokenizer, and packed into fixed-length
windows. Documents are separated by an end-of-sequence token and every window
carries the per-document segment lengths (`seqlens`), so a later stage can run
document-masked attention (no cross-document attention) without changing the
data path.

Design notes:
  - All mutable iteration state lives on object attributes (not in generator
    locals), so `state_dict()` / `load_state_dict()` capture an exact snapshot
    that torchdata's StatefulDataLoader can checkpoint per worker.
  - Each source is infinite: when a stream is exhausted it reshuffles with a new
    seed and loops, tracking its own epoch.
  - HuggingFace IterableDatasets auto-shard across DataLoader workers, so keep
    num_workers <= the smallest source's shard count (FineWeb-Edu sample-10BT
    has 14) to avoid idle workers.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

# Keep all HuggingFace traffic/metadata inside the project tree.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".cache" / "hf"))

from datasets import load_dataset
from tokenizers import Tokenizer

IGNORE_INDEX = -100

# Same presets as scripts/download_corpus.py (edu-filtered web, English + Chinese).
# `script` is the expected dominant writing system; docs that don't match are dropped
# (the "Chinese" source carries a non-trivial tail of long English papers).
PRESETS: dict[str, dict[str, str]] = {
	"en": {"path": "HuggingFaceFW/fineweb-edu", "config": "sample-100BT", "text_key": "text", "script": "latin"},
	"zh": {"path": "opencsg/chinese-fineweb-edu-v2", "config": "", "text_key": "text", "script": "cjk"},
}


@dataclass
class DocFilter:
	"""Cheap rejection of degenerate / off-language documents.

	Two failure modes seen in the web crawls drive these checks:

	  1. Repetitive spam — repeated characters, phrases, or link farms. A single such
	     doc fills a packed window and shows up as an anomalously low training loss
	     (perplexity ~2). Caught by the token-level checks in `reject`.
	  2. Wrong-language contamination — the "Chinese" source (opencsg/chinese-fineweb)
	     carries a tail of very long English papers. Packed into the zh token budget
	     they transiently wreck Chinese eval loss (a doc up to ~60 windows long can
	     dominate hundreds of steps). Caught by the text-level check in `reject_text`,
	     plus the `max_doc_tokens` cap so no single doc can dominate regardless.

	Filtering runs before a doc is queued, so it doesn't perturb the exact-resume
	snapshot (which captures the already-filtered queue). `reject_text` runs first
	(on raw text, before tokenizing — also skips wasting the tokenizer on a 100k-token
	English paper); `reject` runs on the tokenized ids. All thresholds are loose: they
	target egregious cases and should not touch natural prose.

	  - `max_doc_tokens`: hard cap on a single doc's footprint (0 = no cap); truncation,
	    not rejection, so one giant doc can't dominate many windows.
	  - `min_doc_tokens`: drop tiny fragments (post-tokenization).
	  - `max_top_token_frac`: drop if one token id makes up more than this fraction
	    (catches single-character / single-symbol runs).
	  - `min_distinct_ratio` / `max_dup_bigram_frac`: applied only to docs at least
	    `long_doc_tokens` long, where ratios are meaningful — catch low overall
	    diversity and phrase-level loops respectively.
	  - `min_script_frac`: drop if, among CJK + Latin-alphabetic characters, the source's
	    expected script makes up less than this fraction (catches English in the zh
	    stream and vice-versa). `script_scan_chars` bounds the per-doc scan.
	"""

	enabled: bool = True
	max_doc_tokens: int = 8192
	min_doc_tokens: int = 8
	max_top_token_frac: float = 0.5
	long_doc_tokens: int = 100
	min_distinct_ratio: float = 0.1
	max_dup_bigram_frac: float = 0.5
	min_script_frac: float = 0.3
	script_scan_chars: int = 20000

	def reject(self, ids: list[int]) -> bool:
		"""True if `ids` looks degenerate and should be dropped."""
		if not self.enabled:
			return False
		n = len(ids)
		if n < self.min_doc_tokens:
			return True
		counts = Counter(ids)
		if counts.most_common(1)[0][1] / n > self.max_top_token_frac:
			return True
		if n >= self.long_doc_tokens:
			if len(counts) / n < self.min_distinct_ratio:
				return True
			distinct_bigrams = len(set(zip(ids, ids[1:])))
			if 1.0 - distinct_bigrams / (n - 1) > self.max_dup_bigram_frac:
				return True
		return False

	def reject_text(self, text: str, script: str | None) -> bool:
		"""True if `text` is not predominantly in the source's expected `script`."""
		if not self.enabled or not script:
			return False
		cjk = latin = 0
		for ch in text[: self.script_scan_chars]:
			o = ord(ch)
			if 0x4E00 <= o <= 0x9FFF:
				cjk += 1
			elif o < 0x80 and ch.isalpha():
				latin += 1
		total = cjk + latin
		if total == 0:
			return True  # no real alphabetic/CJK content
		frac = (cjk if script == "cjk" else latin) / total
		return frac < self.min_script_frac


@dataclass
class SourceSpec:
	path: str
	config: str | None
	text_key: str
	weight: float
	script: str | None = None  # expected dominant writing system ("latin" | "cjk" | None)

	@staticmethod
	def from_lang(lang: str, weight: float) -> "SourceSpec":
		if lang not in PRESETS:
			raise ValueError(f"unknown stream lang '{lang}'; known: {sorted(PRESETS)}")
		preset = PRESETS[lang]
		return SourceSpec(
			preset["path"], preset["config"] or None, preset["text_key"], weight, preset.get("script")
		)


class _TokenizedSource:
	"""One HF streaming dataset, tokenized in batches, infinite via reshuffle+loop.

	Holds a small queue of already-tokenized documents so `state_dict` can restore
	exactly: the HF stream position plus the queued-but-unconsumed documents.
	"""

	def __init__(
		self,
		spec: SourceSpec,
		tokenizer: Tokenizer,
		base_seed: int,
		shuffle_buffer: int,
		encode_batch_size: int,
		eos_id: int,
		doc_filter: DocFilter | None = None,
	) -> None:
		self.spec = spec
		self.tokenizer = tokenizer
		self.base_seed = base_seed
		self.shuffle_buffer = shuffle_buffer
		self.encode_batch_size = encode_batch_size
		self.eos_id = eos_id
		self.doc_filter = doc_filter or DocFilter()
		self.n_kept = 0
		self.n_dropped = 0      # degenerate (token-level)
		self.n_dropped_lang = 0  # wrong-language (text-level)
		self.epoch = 0
		self._queue: list[list[int]] = []
		self._ds: Any = None
		self._raw_iter: Iterator[dict[str, Any]] | None = None
		self._pending_hf_state: dict[str, Any] | None = None

	def _new_dataset(self) -> Any:
		ds = load_dataset(self.spec.path, name=self.spec.config, split="train", streaming=True)
		if self.shuffle_buffer > 0:
			ds = ds.shuffle(seed=self.base_seed + self.epoch, buffer_size=self.shuffle_buffer)
		return ds

	def _ensure_iter(self) -> None:
		if self._raw_iter is not None:
			return
		self._ds = self._new_dataset()
		if self._pending_hf_state is not None:
			self._ds.load_state_dict(self._pending_hf_state)
			self._pending_hf_state = None
		self._raw_iter = iter(self._ds)

	def _refill(self) -> None:
		self._ensure_iter()
		texts: list[str] = []
		while len(texts) < self.encode_batch_size:
			try:
				example = next(cast(Iterator[dict[str, Any]], self._raw_iter))
			except StopIteration:
				self.epoch += 1
				self._ds = self._new_dataset()
				self._raw_iter = iter(self._ds)
				continue
			text = example.get(self.spec.text_key)
			if not (isinstance(text, str) and text.strip()):
				continue
			if self.doc_filter.reject_text(text, self.spec.script):
				self.n_dropped_lang += 1
				continue
			texts.append(text)
		for encoding in self.tokenizer.encode_batch(texts, add_special_tokens=False):
			ids = encoding.ids
			if not ids:
				continue
			cap = self.doc_filter.max_doc_tokens
			if cap and len(ids) > cap:
				ids = ids[:cap]
			if self.doc_filter.reject(ids):
				self.n_dropped += 1
				continue
			self.n_kept += 1
			self._queue.append([*ids, self.eos_id])

	def next_doc(self) -> list[int]:
		if not self._queue:
			self._refill()
		return self._queue.pop(0)

	def state_dict(self) -> dict[str, Any]:
		self._ensure_iter()
		return {
			"epoch": self.epoch,
			"queue": [list(doc) for doc in self._queue],
			"hf": cast(Any, self._ds).state_dict(),
		}

	def load_state_dict(self, state: dict[str, Any]) -> None:
		self.epoch = int(state.get("epoch", 0))
		self._queue = [list(doc) for doc in state.get("queue", [])]
		self._pending_hf_state = state.get("hf")
		self._raw_iter = None  # rebuilt lazily, applying the pending HF state


class InterleavedStreamingDataset(IterableDataset):
	"""Token-balanced interleave of several sources, packed into seq_len windows."""

	def __init__(
		self,
		specs: list[SourceSpec],
		tokenizer_path: str,
		seq_len: int,
		eos_id: int,
		shuffle_buffer: int = 10_000,
		base_seed: int = 0,
		encode_batch_size: int = 64,
		mask_seams: bool = False,
		doc_filter: DocFilter | None = None,
	) -> None:
		super().__init__()
		if not specs:
			raise ValueError("at least one source spec is required")
		if seq_len <= 1:
			raise ValueError("seq_len must be > 1")
		self.specs = specs
		self.tokenizer_path = tokenizer_path
		self.seq_len = seq_len
		self.eos_id = eos_id
		self.shuffle_buffer = shuffle_buffer
		self.base_seed = base_seed
		self.encode_batch_size = encode_batch_size
		self.mask_seams = mask_seams
		self.doc_filter = doc_filter or DocFilter()
		self._weights = np.asarray([spec.weight for spec in specs], dtype=np.float64)
		if np.any(self._weights <= 0):
			raise ValueError("source weights must be positive")

		self._sources: list[_TokenizedSource] | None = None
		self._cum_tokens = np.zeros(len(specs), dtype=np.float64)
		self._residual: list[int] = []
		self._residual_segs: list[int] = []
		self._seg_counter = 0
		self._pending_state: dict[str, Any] | None = None

	# -- setup -------------------------------------------------------------
	def _ensure_setup(self) -> None:
		if self._sources is not None:
			return
		worker = get_worker_info()
		# Distinct seed per worker so shuffle buffers (and per-source epochs) decorrelate;
		# stable across resume because worker count is fixed.
		seed = self.base_seed + 1009 * (worker.id if worker is not None else 0)
		tokenizer = Tokenizer.from_file(self.tokenizer_path)
		self._sources = [
			_TokenizedSource(
				spec, tokenizer, seed, self.shuffle_buffer, self.encode_batch_size, self.eos_id, self.doc_filter
			)
			for spec in self.specs
		]
		if self._pending_state is not None:
			self._apply_state(self._pending_state)
			self._pending_state = None

	# -- window production (all state on self, so snapshots are exact) -----
	def _produce_window(self) -> dict[str, np.ndarray]:
		sources = cast(list[_TokenizedSource], self._sources)
		while len(self._residual) < self.seq_len + 1:
			pick = int(np.argmin(self._cum_tokens / self._weights))
			doc = sources[pick].next_doc()
			self._cum_tokens[pick] += len(doc)
			self._residual.extend(doc)
			self._residual_segs.extend([self._seg_counter] * len(doc))
			self._seg_counter += 1

		window = self._residual[: self.seq_len + 1]
		seg_window = self._residual_segs[: self.seq_len + 1]
		self._residual = self._residual[self.seq_len :]
		self._residual_segs = self._residual_segs[self.seq_len :]

		input_ids = np.asarray(window[: self.seq_len], dtype=np.int64)
		targets = np.asarray(window[1 : self.seq_len + 1], dtype=np.int64)
		seg_in = np.asarray(seg_window[: self.seq_len], dtype=np.int64)
		# Run lengths of contiguous segment ids = per-document lengths in this window.
		boundaries = np.flatnonzero(np.diff(seg_in)) + 1
		seqlens = np.diff(np.concatenate(([0], boundaries, [self.seq_len]))).astype(np.int32)

		if self.mask_seams:
			# target[i] crosses a document boundary when seg(window[i]) != seg(window[i+1]).
			seam = np.asarray(seg_window[: self.seq_len], dtype=np.int64) != np.asarray(
				seg_window[1 : self.seq_len + 1], dtype=np.int64
			)
			targets[seam] = IGNORE_INDEX

		return {"input_ids": input_ids, "targets": targets, "seqlens": seqlens}

	def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
		self._ensure_setup()
		while True:
			yield self._produce_window()

	# -- exact resume ------------------------------------------------------
	def state_dict(self) -> dict[str, Any]:
		if self._sources is None:
			# Not yet iterated in this (worker) process; echo back whatever we hold.
			return self._pending_state or {}
		return {
			"cum_tokens": self._cum_tokens.tolist(),
			"residual": list(self._residual),
			"residual_segs": list(self._residual_segs),
			"seg_counter": self._seg_counter,
			"sources": [source.state_dict() for source in self._sources],
		}

	def load_state_dict(self, state: dict[str, Any]) -> None:
		if self._sources is None:
			self._pending_state = state  # applied when the worker builds its sources
			return
		self._apply_state(state)

	def _apply_state(self, state: dict[str, Any]) -> None:
		if not state:
			return
		sources = cast(list[_TokenizedSource], self._sources)
		self._cum_tokens = np.asarray(state["cum_tokens"], dtype=np.float64)
		self._residual = list(state["residual"])
		self._residual_segs = list(state["residual_segs"])
		self._seg_counter = int(state["seg_counter"])
		for source, source_state in zip(sources, state["sources"]):
			source.load_state_dict(source_state)


def collate_windows(batch: list[dict[str, np.ndarray]]) -> dict[str, Any]:
	input_ids = torch.as_tensor(np.stack([item["input_ids"] for item in batch]), dtype=torch.long)
	targets = torch.as_tensor(np.stack([item["targets"] for item in batch]), dtype=torch.long)
	# seqlens kept per-sample (ragged) for document-masked attention in a later stage.
	seqlens = [torch.as_tensor(item["seqlens"], dtype=torch.int32) for item in batch]
	return {"input_ids": input_ids, "targets": targets, "seqlens": seqlens}


def build_stream_dataloader(
	langs: list[str],
	weights: list[float],
	tokenizer_path: str,
	seq_len: int,
	batch_size: int,
	eos_id: int,
	shuffle_buffer: int = 10_000,
	num_workers: int = 4,
	base_seed: int = 0,
	mask_seams: bool = False,
	prefetch_factor: int = 4,
	doc_filter: DocFilter | None = None,
):
	from torchdata.stateful_dataloader import StatefulDataLoader

	if len(langs) != len(weights):
		raise ValueError("langs and weights must have the same length")
	specs = [SourceSpec.from_lang(lang, weight) for lang, weight in zip(langs, weights)]
	dataset = InterleavedStreamingDataset(
		specs,
		tokenizer_path=tokenizer_path,
		seq_len=seq_len,
		eos_id=eos_id,
		shuffle_buffer=shuffle_buffer,
		base_seed=base_seed,
		mask_seams=mask_seams,
		doc_filter=doc_filter,
	)
	return StatefulDataLoader(
		dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		collate_fn=collate_windows,
		pin_memory=True,
		drop_last=True,
		persistent_workers=num_workers > 0,
		prefetch_factor=prefetch_factor if num_workers > 0 else None,
	)


def build_eval_batches(
	lang: str,
	tokenizer_path: str,
	seq_len: int,
	batch_size: int,
	eos_id: int,
	num_batches: int,
	seed: int,
	shuffle_buffer: int = 2000,
	doc_filter: DocFilter | None = None,
	mask_seams: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]:
	"""Pull a fixed set of single-language windows once, for periodic evaluation.

	Drawn from the same source with a dedicated seed: not a strict held-out split,
	but with a different shuffle order over billions of tokens the overlap with the
	training stream is negligible, which is what a monitoring set needs. Returns CPU
	(input_ids, targets) batch tensors.

	The monitoring set is filtered the same way as training (default-on `DocFilter`):
	the source carries wrong-language / degenerate docs, so an unfiltered eval set
	would measure those too. Pass a matching `doc_filter` to keep train/eval aligned.
	"""
	dataset = InterleavedStreamingDataset(
		[SourceSpec.from_lang(lang, 1.0)],
		tokenizer_path=tokenizer_path,
		seq_len=seq_len,
		eos_id=eos_id,
		shuffle_buffer=shuffle_buffer,
		base_seed=seed,
		mask_seams=mask_seams,
		doc_filter=doc_filter,
	)
	iterator = iter(dataset)
	batches: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]] = []
	for _ in range(num_batches):
		window = collate_windows([next(iterator) for _ in range(batch_size)])
		batches.append((window["input_ids"], window["targets"], window["seqlens"]))
	return batches


def _inspect() -> None:
	"""Quick standalone sanity/throughput check: python myk-llm/streaming.py ..."""
	import argparse
	import time

	parser = argparse.ArgumentParser()
	parser.add_argument("--tokenizer", type=str, default="_download_models/Qwen3.5-0.8B-pruned64k")
	parser.add_argument("--langs", type=str, default="en,zh")
	parser.add_argument("--weights", type=str, default="0.5,0.5")
	parser.add_argument("--seq-len", type=int, default=2048)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--batches", type=int, default=100)
	parser.add_argument("--eos-token", type=str, default="<|endoftext|>")
	parser.add_argument("--no-filter", action="store_true", help="disable degenerate-doc filtering")
	args = parser.parse_args()

	tok = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
	eos_id = tok.token_to_id(args.eos_token)
	loader = build_stream_dataloader(
		args.langs.split(","),
		[float(w) for w in args.weights.split(",")],
		tokenizer_path=str(Path(args.tokenizer) / "tokenizer.json"),
		seq_len=args.seq_len,
		batch_size=args.batch_size,
		eos_id=eos_id,
		num_workers=args.num_workers,
		doc_filter=DocFilter(enabled=not args.no_filter),
	)
	start = time.time()
	total_tokens = 0
	docs_per_window = []
	for i, batch in enumerate(loader):
		total_tokens += batch["input_ids"].numel()
		docs_per_window.extend(len(s) for s in batch["seqlens"])
		if i + 1 >= args.batches:
			break
	elapsed = time.time() - start
	print(f"batches={args.batches} tokens={total_tokens} elapsed={elapsed:.1f}s "
	      f"throughput={total_tokens/elapsed:,.0f} tok/s avg_docs_per_window={np.mean(docs_per_window):.1f}")
	# Drop stats are only reachable in-process (workers hold their own sources).
	sources = getattr(loader.dataset, "_sources", None)
	if args.num_workers == 0 and sources:
		print("filter drop rates (run with --num-workers 0 to see):")
		for src in sources:
			seen = src.n_kept + src.n_dropped + src.n_dropped_lang
			deg = src.n_dropped / seen if seen else 0.0
			lang = src.n_dropped_lang / seen if seen else 0.0
			print(f"  {src.spec.path}: kept={src.n_kept} "
			      f"dropped_degenerate={src.n_dropped} ({deg:.2%}) "
			      f"dropped_wrong_lang={src.n_dropped_lang} ({lang:.2%})")


if __name__ == "__main__":
	_inspect()
