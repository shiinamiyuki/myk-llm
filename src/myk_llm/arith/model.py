"""MykLLM plus an optional digit-position embedding, and batched greedy evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data.config import Batch
from ..model import MykLLM, ModelConfig
from .tasks import EOS, CharTokenizer, TaskConfig, parse_answer, render, sample_operand, true_answer


@dataclass
class DigitEmbeddingConfig:
    """Abacus-style embedding (McLeish et al., 2024): each digit gets a learned vector for
    its index within its number (1 = first digit written). A random offset in
    [0, max_offset] is added during training so that indices beyond the training
    lengths are also trained, which is what lets models add much longer numbers."""

    enabled: bool = False
    max_offset: int = 50
    max_index: int = 128


def digit_indices(input_ids: torch.Tensor, digit_ids: torch.Tensor) -> torch.Tensor:
    """1-based index of each digit within its run of digits, 0 for other tokens (causal:
    depends only on the token and those before it)."""
    is_digit = torch.isin(input_ids, digit_ids)
    count = is_digit.long().cumsum(dim=1)
    run_start = torch.where(is_digit, torch.zeros_like(count), count).cummax(dim=1).values
    return torch.where(is_digit, count - run_start, torch.zeros_like(count))


class ArithModel(nn.Module):
    def __init__(self, config: ModelConfig, digits: DigitEmbeddingConfig) -> None:
        super().__init__()
        self.lm = MykLLM(config)
        self.digits = digits
        self.tokenizer = CharTokenizer()
        if len(self.tokenizer) > config.vocab_size:
            raise ValueError(f"model.vocab_size must be >= {len(self.tokenizer)}")
        self.register_buffer("digit_ids", torch.tensor(self.tokenizer.digit_ids), persistent=False)
        self.digit_embed = nn.Embedding(digits.max_index + digits.max_offset + 1, config.hidden_size) if digits.enabled else None
        if self.digit_embed is not None:
            nn.init.normal_(self.digit_embed.weight, std=config.initializer_range)
            with torch.no_grad():
                self.digit_embed.weight[0].zero_()  # index 0 = not a digit

    def embed_offset(self, input_ids: torch.Tensor, random_offset: bool) -> torch.Tensor | None:
        if self.digit_embed is None:
            return None
        index = digit_indices(input_ids, self.digit_ids)
        if random_offset and self.digits.max_offset:
            offset = torch.randint(0, self.digits.max_offset + 1, (input_ids.size(0), 1), device=input_ids.device)
            index = torch.where(index > 0, index + offset, index)
        return self.digit_embed(index.clamp(max=self.digit_embed.num_embeddings - 1))

    def forward(self, batch: Batch, head):
        return self.lm(
            batch.input_ids, seqlens=batch.seqlens, head=head,
            embed_offset=self.embed_offset(batch.input_ids, random_offset=self.training),
        )

    @torch.no_grad()
    def greedy_complete(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Greedy completions, recomputing the full (right-padded) prefix each step; fine
        for the short sequences here and valid with any positional scheme."""
        tokenizer, device = self.tokenizer, self.digit_ids.device
        lengths = torch.tensor([len(p) for p in prompts], device=device)
        buffer = torch.full((len(prompts), int(lengths.max()) + max_new_tokens), tokenizer.pad_id, dtype=torch.long, device=device)
        for row, prompt in enumerate(prompts):
            buffer[row, : len(prompt)] = torch.tensor(tokenizer.encode(prompt), device=device)
        done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
        rows = torch.arange(len(prompts), device=device)
        for _ in range(max_new_tokens):
            width = int(lengths.max())
            ids = buffer[:, :width]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = self.lm(ids, embed_offset=self.embed_offset(ids, random_offset=False)).hidden
            # vocab_size may exceed the character set; never emit the unused ids.
            logits = self.lm.logits(hidden[rows, lengths - 1])[:, : len(tokenizer)]
            next_token = logits.argmax(-1)
            active = rows[~done]  # rows still generating append one token (possibly EOS)
            buffer[active, lengths[active]] = next_token[active]
            lengths[active] += 1
            done[active] = next_token[active] == tokenizer.eos_id
            if bool(done.all()):
                break
        return [
            tokenizer.decode(buffer[row, len(prompt) : int(lengths[row])].tolist())
            for row, prompt in enumerate(prompts)
        ]


def completion_budget(task: TaskConfig, digits: int) -> int:
    """Upper bound on completion length for two `digits`-digit operands."""
    a = b = 10**digits - 1
    return len(render(a, b, task)[1]) + 4


@torch.no_grad()
def evaluate_lengths(
    model: ArithModel, task: TaskConfig, lengths: list[int], examples: int, seed: int, batch_size: int = 512
) -> dict[int, float]:
    """Exact-match accuracy of the final answer, both operands `d` digits, for each d."""
    was_training = model.training
    model.eval()
    rng = random.Random(seed)
    accuracy = {}
    for digits in lengths:
        problems = [(sample_operand(rng, digits), sample_operand(rng, digits)) for _ in range(examples)]
        budget = completion_budget(task, digits)
        correct = 0
        for start in range(0, examples, batch_size):
            chunk = problems[start : start + batch_size]
            completions = model.greedy_complete([render(a, b, task)[0] for a, b in chunk], budget)
            correct += sum(
                parse_answer(c, task) == true_answer(a, b, task) and c.endswith(EOS)
                for c, (a, b) in zip(completions, chunk)
            )
        accuracy[digits] = correct / examples
    model.train(was_training)
    return accuracy
