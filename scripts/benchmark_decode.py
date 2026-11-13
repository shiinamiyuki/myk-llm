"""Decode speed of a config (random weights, one GPU), per token and per byte, with and
without multi-token-prediction (MTP) speculative decoding.

One decode iteration is captured as a CUDA graph over static KV caches (attention is
switched to `flash_attn_with_kvcache` on preallocated buffers for this benchmark only;
the model's own decoding grows its cache with torch.cat, which cannot be captured), with
`--context` tokens already cached. A decoder-decoder (model.cross) gets one static
buffer for its shared KV, which every cross-decoder block reads. With K drafted tokens an iteration is:

  verify   the main model on K + 1 tokens (last token + K drafts), output head on all K + 1
  draft    K MTP modules in sequence, each one block + output head on one token

If each draft is accepted with probability a (independently, in order), an iteration
yields E = 1 + a + ... + a^K tokens, so decode speed = batch * E * bytes_per_token / time.
The output head is computed 2K + 1 times per iteration, which is where the vocabulary
size enters.

    python scripts/benchmark_decode.py configs/ngram-small/v64k.yaml --bytes-per-token 4.57
    python scripts/benchmark_decode.py configs/experiments/ablation.yaml model.vocab_size=32768 --bytes-per-token 4.27
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from myk_llm.configuration import load_config
from myk_llm.model import MykLLM
from myk_llm.model.mixers import Attention, CrossAttention, SharedKV
from myk_llm.ops.flash_attn import flash_attn_with_kvcache
from myk_llm.train.config import TrainConfig

_eager_attention = Attention.forward
_eager_shared_kv = SharedKV.forward
_eager_attend = CrossAttention.attend


@dataclass
class StaticKV:
    """A decoder-decoder's shared KV in a static cache, plus this pass's new keys / values
    (each cross-decoder block writes them again before attending: same values, same slots)."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    cache_seqlens: torch.Tensor


def _static_attention(self, x, position_ids, varlen=None, state=None, use_cache=False):
    """Attention over this module's static cache (`self._static`), appending the new keys
    at `cache_seqlens` in place; falls back to the model's own path without one."""
    if not hasattr(self, "_static"):
        return _eager_attention(self, x, position_ids, varlen, state, use_cache)
    k_cache, v_cache, cache_seqlens = self._static
    batch, seq_len, _ = x.shape
    q, k, v = self.qkv_proj(x).split(self.split, dim=-1)
    q = q.view(batch, seq_len, self.num_heads, self.head_dim)
    k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
    v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)
    if self.q_norm is not None and self.k_norm is not None:
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
    if self.rope is not None:
        q, k = self.rope(q, k, position_ids)
    window = (self.window - 1, 0) if self.window is not None else (-1, -1)
    out = flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens, causal=True,
                                  window_size=window)
    out = out.reshape(batch, seq_len, -1)
    if self.gate_proj is not None:
        out = out * torch.sigmoid(self.gate_proj(x))
    shared = StaticKV(k_cache, v_cache, k, v, cache_seqlens) if self.shares_kv else None  # cross.kv=layer
    return self.o_proj(out), shared


def _static_shared_kv(self, x, position_ids, state=None):
    if not hasattr(self, "_static"):
        return _eager_shared_kv(self, x, position_ids, state)
    new = _eager_shared_kv(self, x, position_ids)  # keys / values of the new positions
    return StaticKV(*self._static[:2], new.keys, new.values, self._static[2])


def _static_attend(self, q, kv, varlen):
    if not isinstance(kv, StaticKV):
        return _eager_attend(self, q, kv, varlen)
    return flash_attn_with_kvcache(q, kv.k_cache, kv.v_cache, k=kv.keys, v=kv.values,
                                   cache_seqlens=kv.cache_seqlens, causal=True)


Attention.forward = _static_attention
SharedKV.forward = _static_shared_kv
CrossAttention.attend = _static_attend


def attach_caches(model: MykLLM, batch: int, max_len: int, cache_seqlens: torch.Tensor) -> None:
    for module in model.modules():
        if isinstance(module, (Attention, SharedKV)):
            shape = (batch, max_len, module.num_kv_heads, module.head_dim)
            k = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
            module._static = (k, torch.zeros_like(k), cache_seqlens)


@torch.no_grad()
def check_static_attention(model: MykLLM) -> float:
    """Max |logit difference| between a full forward and prefill + one cached step."""
    vocab, length = model.config.vocab_size, 64
    ids = torch.randint(0, vocab, (2, length + 1), device="cuda")
    for module in model.modules():
        module.__dict__.pop("_static", None)
    reference = model.logits(model(ids).hidden[:, -1]).float()
    seqlens = torch.zeros(2, dtype=torch.int32, device="cuda")
    attach_caches(model, 2, length + 1, seqlens)
    model(ids[:, :length])
    seqlens.fill_(length)
    position = torch.full((2, 1), length, device="cuda")
    out = model(ids[:, length:], position_ids=position, prefix_ids=ids[:, :length])
    return float((model.logits(out.hidden[:, -1]).float() - reference).abs().max())


@torch.no_grad()
def iteration_time(model: MykLLM, batch: int, drafts: int, context: int, repeats: int) -> float:
    """Seconds per decode iteration (verify drafts + 1 tokens, draft `drafts` more)."""
    vocab, ctx = model.config.vocab_size, model.embed.context
    seqlens = torch.full((batch,), context, dtype=torch.int32, device="cuda")
    attach_caches(model, batch, context + drafts + 1, seqlens)
    tokens = torch.randint(0, vocab, (batch, drafts + 1), device="cuda")
    prefix = torch.randint(0, vocab, (batch, ctx), device="cuda") if ctx else None
    positions = (context + torch.arange(drafts + 1, device="cuda")).expand(batch, -1).contiguous()
    draft_position = positions[:, -1:].contiguous()
    result = torch.empty(batch, drafts + 1, dtype=torch.long, device="cuda")

    def step() -> None:
        out = model(tokens, position_ids=positions, prefix_ids=prefix)
        verified = model.logits(out.hidden).argmax(-1)  # (batch, drafts + 1)
        hidden, token = out.residual[:, -1:], verified[:, -1:]
        for k, module in enumerate(model.mtp[:drafts]):
            hidden = module(hidden, model.embed.token(token), draft_position, None)
            token = model.logits(module.final_norm(hidden)).argmax(-1)
            result[:, k : k + 1] = token
        result[:, -1:] = verified[:, -1:]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--drafts", type=int, nargs="+", default=[0, 1, 2, 3], help="MTP depths K to time")
    parser.add_argument("--accept", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--context", type=int, default=1024, help="tokens already in the cache")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--bytes-per-token", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true", help="torch.compile each block (fuses elementwise ops)")
    args = parser.parse_args()

    overrides = [*args.overrides, f"model.mtp.depth={max(args.drafts)}"]
    config = load_config(TrainConfig, args.config, overrides).model
    # Room in the rotary position table for the cached context plus one iteration's tokens.
    config.max_seq_len = max(config.max_seq_len, args.context + max(args.drafts) + 1)
    torch.manual_seed(0)
    model = MykLLM(config).cuda().to(torch.bfloat16).eval()
    if args.compile:  # one graph per (block kind, token count): raise the recompile limit
        torch._dynamo.config.recompile_limit = 256
        model.compile_blocks(dynamic=False)
    body = model.num_parameters(exclude_embeddings=True) - sum(p.numel() for p in model.mtp.parameters())
    print(f"vocab {config.vocab_size}, {config.layer_pattern} x {config.num_layers}, hidden {config.hidden_size}, "
          f"{body / 1e6:.1f}M non-embedding params, n-gram {config.ngram.orders or 'off'}, "
          f"static-cache check max|dlogit| = {check_static_attention(model):.3f}")
    unit = "KB/s" if args.bytes_per_token != 1.0 else "Ktok/s"
    print(f"| batch | K | ms/iter | " + " | ".join(f"{unit} a={a}" for a in args.accept) + " |")
    print("|---|---|---|" + "---|" * len(args.accept))
    for batch in args.batch:
        for drafts in args.drafts:
            seconds = iteration_time(model, batch, drafts, args.context, args.repeats)
            speeds = [batch * sum(a**i for i in range(drafts + 1)) * args.bytes_per_token / seconds / 1e3
                      for a in args.accept]
            print(f"| {batch} | {drafts} | {seconds * 1e3:.3f} | " + " | ".join(f"{s:.1f}" for s in speeds) + " |")


if __name__ == "__main__":
    main()
