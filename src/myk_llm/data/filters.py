"""Cheap document-quality heuristics shared by streaming and pre-tokenization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class DocFilter:
    """Cheap rejection of degenerate / off-language documents.

    Applies to sources that are not marked `curated` (a source already filtered
    upstream, e.g. by an educational-quality score, bypasses these heuristics). Only
    the `max_doc_tokens` cap below still applies to curated sources.

    Two failure modes seen in the web crawls drive these checks:

      1. Repetitive spam — repeated characters, phrases, or link farms. A single such
         doc fills a packed window and shows up as an anomalously low training loss
         (perplexity ~2). Caught by the token-level checks in `reject`.
      2. Wrong-language contamination — a web source can carry a tail of docs in the
         wrong language (e.g. long English papers in a CJK stream). Packed into the
         token budget they transiently wreck eval loss (a doc up to ~60 windows long
         can dominate hundreds of steps). Caught by the text-level check in
         `reject_text`, plus the `max_doc_tokens` cap so no single doc can dominate.

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
