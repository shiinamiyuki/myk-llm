"""
Character-level Block Diffusion Language Model (BD3-LM style), minimal & educational.

Block Diffusion (Arriola et al., ICLR 2025) interpolates between autoregressive (AR)
and (masked) diffusion LMs:
  - Split the sequence into blocks of size K.
  - Generate blocks left-to-right (AR across blocks)  -> arbitrary length + KV cache
  - Within each block, masked (absorbing-state) diffusion -> parallel denoising

      log p(x) = sum_b log p_theta(x^b | x^{<b})
  each conditional = masked diffusion over the block, conditioned on CLEAN prior blocks.

  K = 1  -> recovers autoregressive next-token prediction
  K = L  -> recovers a pure masked diffusion LM (MDLM)

This file: tiny Transformer + dual-stream single-pass training + block-by-block sampling.
It is for *learning*, not for competitive perplexity.

Run:
  python block_dlm.py --data input.txt           # e.g. tiny shakespeare
  python block_dlm.py                             # falls back to a tiny built-in corpus
"""
import argparse, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_ID = None  # set in main() once vocab size is known; equals vocab_size (an extra token)

# --------------------------------------------------------------------------------------
# Noise schedule: linear ("uniform") masked diffusion.
# Convention: t in [0,1]; alpha(t) = prob a token stays clean. t=0 -> clean, t=1 -> all mask.
# --------------------------------------------------------------------------------------
def alpha(t):              # linear schedule
    return 1.0 - t

def loss_weight(t):        # w(t) = -alpha'(t)/(1-alpha(t)); for linear -> 1/t
    return 1.0 / t


# --------------------------------------------------------------------------------------
# Attention masks (True = attending is allowed).
# --------------------------------------------------------------------------------------
def train_mask(L, K, device):
    """Dual-stream mask over [clean x ; noised x_t], shape (2L, 2L).

    clean q  -> clean k, block-causal (kb <= qb)         : provides AR context
    noised q -> clean k, strictly earlier block (kb < qb): CLEAN conditioning, no answer leak
    noised q -> noised k, same block (kb == qb)          : within-block bidirectional denoising
    """
    blk = torch.arange(L, device=device) // K            # block id per position
    qb = torch.cat([blk, blk])                           # (2L,)
    noised = torch.cat([torch.zeros(L, device=device),
                        torch.ones(L, device=device)]).bool()
    QB, KB = qb[:, None], qb[None, :]
    QN, KN = noised[:, None], noised[None, :]
    r1 = (~QN) & (~KN) & (KB <= QB)
    r2 = ( QN) & (~KN) & (KB <  QB)
    r3 = ( QN) & ( KN) & (KB == QB)
    return r1 | r2 | r3


def gen_mask(N, K, device):
    """Single-stream generation mask, length N (a multiple of K). Last K = current block.

    context q -> context k, block-causal (kb <= qb)
    current q -> everything (all prior blocks are clean; current block is bidirectional)
    """
    blk = torch.arange(N, device=device) // K
    cur = blk.max()
    is_cur = (blk == cur)
    QB, KB = blk[:, None], blk[None, :]
    ISC_q, ISC_k = is_cur[:, None], is_cur[None, :]
    rc = (~ISC_q) & (~ISC_k) & (KB <= QB)
    rk = ISC_q                                            # current query attends all positions
    return rc | rk


# --------------------------------------------------------------------------------------
# Tiny Transformer (decoder-only style, but attention pattern is supplied externally).
# --------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.h, self.dh = h, d // h
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, addmask):                        # addmask: (S,S), 0 / -inf
        B, S, D = x.shape
        y = self.ln1(x)
        q = self.q(y).view(B, S, self.h, self.dh).transpose(1, 2)
        k = self.k(y).view(B, S, self.h, self.dh).transpose(1, 2)
        v = self.v(y).view(B, S, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh) + addmask
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.o(out)
        x = x + self.mlp(self.ln2(x))
        return x


class BlockDLM(nn.Module):
    def __init__(self, vocab, d=256, h=4, layers=4, ff=1024, maxlen=512):
        super().__init__()
        self.tok = nn.Embedding(vocab + 1, d)            # +1 row for [MASK]
        self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, h, ff) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)                  # predict real vocab only (never [MASK])

    def forward(self, tokens, attn_bool, pos):
        x = self.tok(tokens) + self.pos(pos)[None]
        add = torch.zeros_like(attn_bool, dtype=x.dtype).masked_fill(~attn_bool, float('-inf'))
        for b in self.blocks:
            x = b(x, add)
        return self.head(self.lnf(x))


# --------------------------------------------------------------------------------------
# Training loss: one forward pass, all blocks, faithful clean conditioning.
# --------------------------------------------------------------------------------------
def diffusion_loss(model, x, K, eps=1e-3):
    Bsz, L = x.shape
    nblk = L // K
    dev = x.device
    t_blk = torch.rand(Bsz, nblk, device=dev) * (1 - eps) + eps   # per-block t in [eps,1]
    t_tok = t_blk.repeat_interleave(K, dim=1)                     # (Bsz, L)
    mask_rate = 1.0 - alpha(t_tok)                                # = t_tok (linear)
    masked = torch.rand(Bsz, L, device=dev) < mask_rate
    x_t = torch.where(masked, torch.full_like(x, MASK_ID), x)

    z = torch.cat([x, x_t], dim=1)                               # (Bsz, 2L)
    pos = torch.arange(L, device=dev).repeat(2)                  # clean & noised share positions
    attn = train_mask(L, K, dev)
    logits = model(z, attn, pos)[:, L:, :]                       # noised half only -> (Bsz,L,V)

    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                         x.reshape(-1), reduction='none').view(Bsz, L)
    w = loss_weight(t_tok)
    loss = (ce * w * masked.float()).sum() / Bsz                 # NELBO per sequence (batch-avg)
    return loss


# --------------------------------------------------------------------------------------
# Sampling: block-by-block AR, with ancestral masked-diffusion denoising inside each block.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def sample(model, K, n_blocks, steps=16, device='cpu', temperature=1.0):
    model.eval()
    ctx = torch.empty(1, 0, dtype=torch.long, device=device)
    for _ in range(n_blocks):
        cur = torch.full((1, K), MASK_ID, dtype=torch.long, device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t, s = ts[i].item(), ts[i + 1].item()
            seq = torch.cat([ctx, cur], dim=1)
            N = seq.size(1)
            pos = torch.arange(N, device=device)
            logits = model(seq, gen_mask(N, K, device), pos)[:, ctx.size(1):, :]   # (1,K,V)
            probs = (logits / temperature).softmax(-1)
            x0 = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, K)
            a_t, a_s = alpha(t), alpha(s)
            unmask_p = (a_s - a_t) / max(1.0 - a_t, 1e-6)          # linear -> (t-s)/t
            do_un = (torch.rand(1, K, device=device) < unmask_p) & (cur == MASK_ID)
            cur = torch.where(do_un, x0, cur)
        if (cur == MASK_ID).any():                                # final cleanup -> argmax
            seq = torch.cat([ctx, cur], dim=1); N = seq.size(1)
            pos = torch.arange(N, device=device)
            x0 = model(seq, gen_mask(N, K, device), pos)[:, ctx.size(1):, :].argmax(-1)
            cur = torch.where(cur == MASK_ID, x0, cur)
        ctx = torch.cat([ctx, cur], dim=1)
    return ctx[0]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def get_data(path):
    if path and os.path.exists(path):
        text = open(path, encoding='utf-8').read()
    else:
        # Tiny public-domain fallback so the script runs; replace with real data (e.g. tiny shakespeare).
        text = ("the wheels on the bus go round and round, round and round, all through the town. "
                "the people on the bus go up and down, up and down, all through the town. ") * 400
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, stoi, itos, len(chars)


def get_batch(data, L, bs, device):
    ix = torch.randint(0, len(data) - L, (bs,))
    x = torch.stack([data[i:i + L] for i in ix]).to(device)
    return x


# --------------------------------------------------------------------------------------
def main():
    global MASK_ID
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='')
    ap.add_argument('--seq_len', type=int, default=128)   # L, must be a multiple of block_size
    ap.add_argument('--block_size', type=int, default=16) # K  (try 1 for AR-like, 128 for pure MDLM)
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--ff', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--sample_steps', type=int, default=16)
    args = ap.parse_args()

    assert args.seq_len % args.block_size == 0, "seq_len must be a multiple of block_size"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data, stoi, itos, vocab = get_data(args.data)
    MASK_ID = vocab
    print(f"device={device}  vocab={vocab}  L={args.seq_len}  K={args.block_size}  "
          f"blocks={args.seq_len // args.block_size}")

    model = BlockDLM(vocab, args.d, args.heads, args.layers, args.ff,
                     maxlen=max(512, args.seq_len)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    model.train()
    for step in range(1, args.steps + 1):
        x = get_batch(data, args.seq_len, args.batch, device)
        loss = diffusion_loss(model, x, args.block_size)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    n_blocks = args.seq_len // args.block_size
    out = sample(model, args.block_size, n_blocks, steps=args.sample_steps, device=device)
    print("\n--- sample ---")
    print(''.join(itos[i.item()] for i in out))


if __name__ == '__main__':
    main()