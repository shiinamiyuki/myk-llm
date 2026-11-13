"""Autoregressive sampling with per-layer caches."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .transformer import MykLLM, ModelOutput


def _last_logits(model: MykLLM, output: ModelOutput) -> tuple[ModelOutput, torch.Tensor]:
    # Runs as the forward's head: under FSDP the output head is only gathered there.
    return output, model.logits(output.hidden[:, -1]).float()


@torch.no_grad()
@torch.compiler.set_stance("force_eager")  # compiled blocks would recompile for every length
def generate(
    model: MykLLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Sample continuations for a (batch, prompt_len) block of equal-length prompts.

    temperature <= 0 means greedy. `repetition_penalty` (CTRL-style) applies only to
    tokens generated so far, not the prompt: penalizing the prompt would also penalize
    chat-template tokens such as the end-of-turn token and keep the model from stopping.
    Returns prompt + generated tokens; finished rows are padded with `eos_token_id`.
    """
    if input_ids.dim() != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must be a non-empty (batch, seq_len) tensor")
    max_len = model.config.max_seq_len
    if input_ids.size(1) + max_new_tokens > max_len:
        raise ValueError(f"prompt ({input_ids.size(1)}) + max_new_tokens ({max_new_tokens}) exceeds max_seq_len {max_len}")

    tokens = input_ids
    batch = tokens.size(0)
    context = model.embed.context
    output, logits = model(tokens, use_cache=True, keep_last=1, head=_last_logits)
    cache = output.cache
    finished = torch.zeros(batch, dtype=torch.bool, device=tokens.device)
    prompt_len = tokens.size(1)

    for _ in range(max_new_tokens):
        if repetition_penalty != 1.0 and tokens.size(1) > prompt_len:
            seen = tokens[:, prompt_len:]
            scores = logits.gather(1, seen)
            scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
            logits = logits.scatter(1, seen, scores)
        next_token = _sample(logits, temperature, top_k, top_p)
        if eos_token_id is not None:
            next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            finished |= next_token == eos_token_id
        tokens = torch.cat([tokens, next_token[:, None]], dim=1)
        if eos_token_id is not None and bool(finished.all()):
            break
        position = torch.full((batch, 1), tokens.size(1) - 1, device=tokens.device, dtype=torch.long)
        prefix = tokens[:, -1 - context : -1] if context else None
        output, logits = model(
            tokens[:, -1:], position_ids=position, cache=cache, use_cache=True, prefix_ids=prefix, head=_last_logits
        )
        cache = output.cache
    return tokens


def _sample(logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None and top_k < logits.size(-1):
        cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        drop = cumulative - F.softmax(sorted_logits, dim=-1) > top_p  # keep the token that crosses top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
