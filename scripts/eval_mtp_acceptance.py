"""Draft acceptance of a model's multi-token-prediction (MTP) modules, for speculative
decoding with K = 1..depth drafts.

At position i the main model's greedy token is g(i) = argmax p(. | x_<=i) and MTP depth k
drafts d_k(i), the token at i + 1 + k. With greedy verification, draft k is accepted when
d_j(i) == g(i + j) for every j <= k. Two text sources:

  text       teacher-forced on validation text (the MTP modules see the text's tokens);
             a proxy, since real decoding conditions on the model's own continuation
  greedy     the model first continues validation prompts greedily, then the same
             measurement runs on its own continuation: the exact acceptance of greedy
             speculative decoding on that text (small models loop, so check `repeat`)

Per K it prints the chained acceptance a_k, the expected tokens per iteration
E = 1 + a_1 + ... + a_K and the expected bytes per iteration; `p_accept` is the
sampling (T = 1) acceptance with greedy drafts, prod_j p(d_j(i) | x_<=i+j), on the text.

    python scripts/eval_mtp_acceptance.py runs/sweeps/ngram-mtp/v32k-none
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from myk_llm.configuration import from_dict
from myk_llm.data import build_fixed_batches
from myk_llm.data.config import IGNORE_INDEX
from myk_llm.model.generation import generate
from myk_llm.model.io import load_model
from myk_llm.train.config import TrainConfig
from myk_llm.train.objectives import mtp_targets
from myk_llm.train.trainer import token_byte_lengths


@torch.no_grad()
def acceptance(model, input_ids, targets, seqlens, token_bytes, region_start: int = 0) -> dict[str, np.ndarray]:
    """Per-position chained greedy acceptance (depth x positions), sampling acceptance,
    and bytes of the token each draft would emit, for one row."""
    depth = len(model.mtp)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids, seqlens=seqlens)
        mtp_hidden = model.mtp_hidden(output, input_ids)
    main_logits = model.logits(output.hidden[0]).float()  # (T, V)
    greedy = main_logits.argmax(-1)
    log_probs = main_logits.log_softmax(-1)
    valid = mtp_targets(targets, depth)[0] != IGNORE_INDEX  # tokens i+1 .. i+1+depth in one document
    valid[:region_start] = False
    length = greedy.size(0)
    chain = torch.ones(length, dtype=torch.bool, device=greedy.device)
    p_chain = torch.ones(length, device=greedy.device)
    accepted, p_accepted, draft_bytes = [], [], []
    for k, hidden in enumerate(mtp_hidden, start=1):
        draft = model.logits(hidden[0]).float().argmax(-1)  # predicts token i + 1 + k
        target = torch.full_like(greedy, -1)
        target[:-k] = greedy[k:]
        chain &= draft == target
        lp = torch.full((length,), float("-inf"), device=greedy.device)
        lp[:-k] = log_probs[k:].gather(1, draft[:-k, None]).squeeze(1)
        p_chain *= lp.exp()
        accepted.append(chain[valid].float().cpu().numpy())
        p_accepted.append(p_chain[valid].cpu().numpy())
        draft_bytes.append(token_bytes[draft.cpu()][valid.cpu()].numpy())
    return {
        "accepted": np.stack(accepted),
        "p_accepted": np.stack(p_accepted),
        "bytes": np.stack(draft_bytes),
        "first_bytes": token_bytes[greedy.cpu()][valid.cpu()].numpy(),  # the verified token
    }


def summarize(parts: list[dict[str, np.ndarray]]) -> dict:
    cat = {key: np.concatenate([p[key] for p in parts], axis=-1) for key in parts[0]}
    depth = cat["accepted"].shape[0]
    rows = []
    for K in range(depth + 1):
        tokens = 1 + cat["accepted"][:K].sum(0)
        nbytes = cat["first_bytes"] + (cat["accepted"][:K] * cat["bytes"][:K]).sum(0)
        rows.append({
            "K": K,
            "a_k": float(cat["accepted"][K - 1].mean()) if K else 1.0,
            "tokens_per_iter": float(tokens.mean()),
            "bytes_per_iter": float(nbytes.mean()),
            "p_accept_k": float(cat["p_accepted"][K - 1].mean()) if K else 1.0,
            "p_tokens_per_iter": float(1 + cat["p_accepted"][:K].sum(0).mean()),
        })
    return {"positions": int(cat["accepted"].shape[1]), "rows": rows}


def repeat_fraction(ids: torch.Tensor, n: int = 4) -> float:
    """Share of n-grams that already occurred earlier in the same row."""
    seen, repeats, total = set(), 0, 0
    values = ids.tolist()
    for i in range(len(values) - n + 1):
        gram = tuple(values[i : i + n])
        repeats += gram in seen
        seen.add(gram)
        total += 1
    return repeats / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--rows", type=int, default=64, help="validation rows (2048 tokens) per language")
    parser.add_argument("--prompts", type=int, default=64, help="greedy continuations per language")
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--new", type=int, default=256)
    args = parser.parse_args()

    for run_dir in map(Path, args.run_dirs):
        config = from_dict(TrainConfig, yaml.safe_load((run_dir / "config.yaml").read_text()))
        checkpoint = sorted((run_dir / "checkpoints").glob("step_*"))[-1]
        model = load_model(checkpoint)
        from transformers import AutoTokenizer

        token_bytes = token_byte_lengths(AutoTokenizer.from_pretrained(config.data.tokenizer)).float()
        print(f"\n## {run_dir.name} ({checkpoint.name}, vocab {model.config.vocab_size}, MTP depth {len(model.mtp)})")
        for val in config.eval.val_sets:
            data = replace(config.data, sources=val.sources, micro_batch_size=1)
            batches = build_fixed_batches(data, args.rows, config.eval.seed)
            text = summarize([
                acceptance(model, b.input_ids.cuda(), b.targets.cuda(), [s.cuda() for s in b.seqlens], token_bytes)
                for b in batches
            ])
            parts, repeats = [], []
            for b in batches[: args.prompts]:
                prompt = b.input_ids[:, : args.prompt_len].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ids = generate(model, prompt, args.new, temperature=0.0)
                targets = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], IGNORE_INDEX)], dim=1)
                seqlens = [torch.tensor([ids.size(1)], device="cuda")]
                parts.append(acceptance(model, ids, targets, seqlens, token_bytes, region_start=args.prompt_len))
                repeats.append(repeat_fraction(ids[0, args.prompt_len :]))
            greedy = summarize(parts)
            print(f"\n{val.name}: text {text['positions']} positions; greedy {greedy['positions']} positions, "
                  f"repeated 4-grams in continuations {100 * np.mean(repeats):.0f}%")
            print("| K | text a_k | text tok/iter | text B/iter | p_accept_k (T=1) | greedy a_k | greedy tok/iter | greedy B/iter |")
            print("|---|---|---|---|---|---|---|---|")
            for t, g in zip(text["rows"], greedy["rows"]):
                print(f"| {t['K']} | {t['a_k']:.3f} | {t['tokens_per_iter']:.3f} | {t['bytes_per_iter']:.2f} | "
                      f"{t['p_accept_k']:.3f} | {g['a_k']:.3f} | {g['tokens_per_iter']:.3f} | {g['bytes_per_iter']:.2f} |")


if __name__ == "__main__":
    main()
