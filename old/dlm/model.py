"""Character-level Block Diffusion LM (BD3-LM style), CUDA + FlexAttention only.

Faithful to block_dlm.py:
  - Shared-weight, single-pass *dual-stream* training over [clean ; noised] of length 2L.
  - Block-causal clean conditioning (no answer leakage) expressed as a FlexAttention mask_mod.
  - Block-by-block ancestral sampling with within-block bidirectional denoising.

Requires a CUDA device and torch >= 2.5 (FlexAttention). There is no CPU fallback.
"""
import argparse
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    create_block_mask as _create_block_mask,
    flex_attention as _flex_attention,
)

flex_attention = torch.compile(_flex_attention)


def alpha(t: torch.Tensor | float) -> torch.Tensor | float:
    return 1.0 - t


def loss_weight(t: torch.Tensor) -> torch.Tensor:
    return 1.0 / t


class CharTokenizer:
    def __init__(self, chars: Sequence[str]):
        ordered = list(chars)
        if len(ordered) != len(set(ordered)):
            raise ValueError("characters must be unique")
        self.stoi = {ch: idx for idx, ch in enumerate(ordered)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long, device=device)

    def decode(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(self.itos[int(idx)] for idx in token_ids)


@dataclass
class MykAttentionConfig:
    num_attention_heads: int = 8
    head_dim: int = 64
    num_kv_heads: int = 8

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.head_dim


@dataclass
class BlockDiffusionConfig:
    vocab_size: int
    seq_len: int = 128
    block_size: int = 16
    num_layers: int = 6
    num_attention_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int = 64
    ff_mult: int = 4
    dropout: float = 0.0
    max_position_embeddings: int = 512

    def __post_init__(self) -> None:
        if self.seq_len % self.block_size != 0:
            raise ValueError("seq_len must be divisible by block_size")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        if self.max_position_embeddings < self.seq_len:
            raise ValueError("max_position_embeddings must cover seq_len")

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def ff_hidden_size(self) -> int:
        return self.hidden_size * self.ff_mult

    def attention_config(self) -> MykAttentionConfig:
        return MykAttentionConfig(
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
        )


# --------------------------------------------------------------------------------------
# FlexAttention mask_mods (True = attending allowed).
# --------------------------------------------------------------------------------------
def _dual_stream_mask_mod(seq_len: int, block_size: int):
    """Dual-stream mask over [clean ; noised], length 2L.

    clean q  -> clean k, block-causal (kb <= qb)          : AR context (bidirectional in-block)
    noised q -> clean k, strictly earlier block (kb < qb) : clean conditioning, no answer leak
    noised q -> noised k, same block (kb == qb)           : within-block bidirectional denoising
    """
    L, K = seq_len, block_size

    def mask_mod(b, h, q_idx, kv_idx):
        q_noised = q_idx >= L
        kv_noised = kv_idx >= L
        qb = (q_idx % L) // K
        kb = (kv_idx % L) // K
        r1 = (~q_noised) & (~kv_noised) & (kb <= qb)
        r2 = q_noised & (~kv_noised) & (kb < qb)
        r3 = q_noised & kv_noised & (kb == qb)
        return r1 | r2 | r3

    return mask_mod


def _gen_mask_mod(block_size: int, current_block: int):
    """Single-stream generation mask. Block `current_block` is the block being denoised.

    context q -> context k, block-causal (kb <= qb)
    current q -> everything (priors are clean; current block is bidirectional)
    """
    K = block_size

    def mask_mod(b, h, q_idx, kv_idx):
        qb = q_idx // K
        kb = kv_idx // K
        is_cur_q = qb == current_block
        is_cur_k = kb == current_block
        rc = (~is_cur_q) & (~is_cur_k) & (kb <= qb)
        return rc | is_cur_q

    return mask_mod


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size: int, ff_hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(hidden_size, ff_hidden_size)
        self.up = torch.nn.Linear(hidden_size, ff_hidden_size)
        self.down = torch.nn.Linear(ff_hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate(x)) * self.up(x)
        return self.dropout(self.down(x))


class MykAttention(torch.nn.Module):
    def __init__(self, config: MykAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.kv_hidden_size = config.head_dim * config.num_kv_heads
        self.enable_gqa = config.num_attention_heads != config.num_kv_heads
        self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.kv_hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.kv_hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, block_mask) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_size}")

        nh, nkv, hd = (
            self.config.num_attention_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        q = self.q_proj(x).view(batch_size, seq_len, nh, hd).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, nkv, hd).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, nkv, hd).transpose(1, 2)
        out = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=self.enable_gqa)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        return self.o_proj(out)


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: BlockDiffusionConfig) -> None:
        super().__init__()
        self.attn_norm = torch.nn.LayerNorm(config.hidden_size)
        self.attn = MykAttention(config.attention_config())
        self.ffn_norm = torch.nn.LayerNorm(config.hidden_size)
        self.ffn = FeedForward(config.hidden_size, config.ff_hidden_size, config.dropout)

    def forward(self, x: torch.Tensor, block_mask) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), block_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BlockDiffusionModel(torch.nn.Module):
    def __init__(self, config: BlockDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.mask_token_id = config.vocab_size
        self.token_embedding = torch.nn.Embedding(config.vocab_size + 1, config.hidden_size)
        self.position_embedding = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_dropout = torch.nn.Dropout(config.dropout)
        self.layers = torch.nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = torch.nn.LayerNorm(config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._mask_cache: dict = {}

    @property
    def device(self) -> torch.device:
        return self.token_embedding.weight.device

    def _dual_stream_block_mask(self, seq_len: int):
        key = ("train", seq_len, self.config.block_size)
        cached = self._mask_cache.get(key)
        if cached is None:
            cached = _create_block_mask(
                _dual_stream_mask_mod(seq_len, self.config.block_size),
                B=None,
                H=None,
                Q_LEN=2 * seq_len,
                KV_LEN=2 * seq_len,
                device=self.device,
            )
            self._mask_cache[key] = cached
        return cached

    def _gen_block_mask(self, total_len: int, current_block: int):
        key = ("gen", total_len, current_block, self.config.block_size)
        cached = self._mask_cache.get(key)
        if cached is None:
            cached = _create_block_mask(
                _gen_mask_mod(self.config.block_size, current_block),
                B=None,
                H=None,
                Q_LEN=total_len,
                KV_LEN=total_len,
                device=self.device,
            )
            self._mask_cache[key] = cached
        return cached

    def backbone(self, tokens: torch.Tensor, positions: torch.Tensor, block_mask) -> torch.Tensor:
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        x = self.embedding_dropout(x)
        for layer in self.layers:
            x = layer(x, block_mask)
        return self.lm_head(self.final_norm(x))

    def diffusion_loss(self, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        batch_size, seq_len = x.shape
        if seq_len != self.config.seq_len:
            raise ValueError(f"expected sequence length {self.config.seq_len}, got {seq_len}")
        K = self.config.block_size
        num_blocks = seq_len // K

        t_blk = torch.rand(batch_size, num_blocks, device=x.device) * (1.0 - eps) + eps
        t_tok = t_blk.repeat_interleave(K, dim=1)                       # (B, L)
        mask_rate = 1.0 - alpha(t_tok)                                  # = t_tok (linear)
        masked = torch.rand(batch_size, seq_len, device=x.device) < mask_rate
        x_t = torch.where(masked, torch.full_like(x, self.mask_token_id), x)

        z = torch.cat([x, x_t], dim=1)                                 # (B, 2L)
        positions = torch.arange(seq_len, device=x.device).repeat(2)   # clean & noised share positions
        block_mask = self._dual_stream_block_mask(seq_len)
        logits = self.backbone(z, positions, block_mask)[:, seq_len:, :]  # noised half -> (B, L, V)

        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x.reshape(-1),
            reduction="none",
        ).view(batch_size, seq_len)
        weights = loss_weight(t_tok)
        loss = (ce * weights * masked.float()).sum() / batch_size
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_blocks: int,
        steps: int = 16,
        temperature: float = 1.0,
        prefix_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        K = self.config.block_size
        if prefix_tokens is None:
            context = torch.empty(1, 0, dtype=torch.long, device=self.device)
        else:
            if prefix_tokens.dim() == 1:
                prefix_tokens = prefix_tokens.unsqueeze(0)
            if prefix_tokens.size(0) != 1:
                raise ValueError("prefix_tokens must have batch size 1")
            if prefix_tokens.size(1) % K != 0:
                raise ValueError("prefix length must align to block_size")
            context = prefix_tokens.to(self.device)

        for _ in range(num_blocks):
            current = torch.full((1, K), self.mask_token_id, dtype=torch.long, device=self.device)
            ctx_len = context.size(1)
            current_block = ctx_len // K
            total_len = ctx_len + K
            block_mask = self._gen_block_mask(total_len, current_block)
            positions = torch.arange(total_len, device=self.device)
            timesteps = torch.linspace(1.0, 0.0, steps + 1, device=self.device)

            for idx in range(steps):
                t = timesteps[idx].item()
                s = timesteps[idx + 1].item()
                seq = torch.cat([context, current], dim=1)
                logits = self.backbone(seq, positions, block_mask)[:, ctx_len:, :]  # (1, K, V)
                probs = (logits / temperature).softmax(dim=-1)
                proposals = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, K)
                a_t, a_s = alpha(t), alpha(s)
                unmask_prob = (a_s - a_t) / max(1.0 - a_t, 1e-6)
                do_unmask = (
                    torch.rand(1, K, device=self.device) < unmask_prob
                ) & (current == self.mask_token_id)
                current = torch.where(do_unmask, proposals, current)

            if (current == self.mask_token_id).any():
                seq = torch.cat([context, current], dim=1)
                logits = self.backbone(seq, positions, block_mask)[:, ctx_len:, :]
                final_tokens = logits.argmax(dim=-1)
                current = torch.where(current == self.mask_token_id, final_tokens, current)

            context = torch.cat([context, current], dim=1)

        return context[0]


def build_char_dataset(text: str, device: Optional[torch.device] = None) -> tuple[torch.Tensor, CharTokenizer]:
    tokenizer = CharTokenizer.from_text(text)
    data = tokenizer.encode(text, device=device)
    return data, tokenizer


def get_batch(data: torch.Tensor, seq_len: int, batch_size: int, device: torch.device | str) -> torch.Tensor:
    upper = data.size(0) - seq_len
    if upper <= 0:
        raise ValueError("dataset must be longer than seq_len")
    indices = torch.randint(0, upper, (batch_size,), device=data.device)
    batch = torch.stack([data[idx:idx + seq_len] for idx in indices.tolist()])
    return batch.to(device)


def _load_text(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    return (
        "the wheels on the bus go round and round, round and round, all through the town. "
        "the people on the bus go up and down, up and down, all through the town. "
    ) * 400


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this model is CUDA-only; no GPU is available")
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    text = _load_text(args.data)
    data, tokenizer = build_char_dataset(text)

    config = BlockDiffusionConfig(
        vocab_size=tokenizer.vocab_size,
        seq_len=args.seq_len,
        block_size=args.block_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        max_position_embeddings=max(512, args.seq_len),
    )
    model = BlockDiffusionModel(config).to(device).to(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_blocks = config.seq_len // config.block_size
    print(
        f"device={device}  vocab={config.vocab_size}  L={config.seq_len}  "
        f"K={config.block_size}  blocks={num_blocks}  "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    model.train()
    for step in range(1, args.steps + 1):
        x = get_batch(data, config.seq_len, args.batch, device)
        loss = model.diffusion_loss(x)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    out = model.sample(num_blocks, steps=args.sample_steps)
    print("\n--- sample ---")
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
