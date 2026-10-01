"""Arithmetic problems as character strings, with optional chain-of-thought (CoT).

Every example is `<prompt>=<completion>$`: the model reads the prompt up to and
including "=", and is trained on (and must generate) the completion and the "$"
end-of-sequence (EOS) marker. The final answer is whatever follows the last "|" in
the completion (or the whole completion when there is no CoT).

Formats (`TaskConfig.format`):

  direct   123+456=579$
  reverse  123+456=975$                         answer written least-significant digit first
  cot      addition: one step per digit, least significant first, "a+b+carry:two-digit sum"
             123+456=3+6+0:09,2+5+0:07,1+4+0:05|579$
           multiplication: partial products a*b_i*10^i and running sums, all written
           least-significant digit first; then the answer in normal order
             37*24=841;047>888|888$     (148 + 740 = 888)

`reverse_operands` writes the prompt operands least-significant digit first too.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

EOS = "$"
PAD = "#"
FILLER = "_"  # meaningless character randomly inserted into prompts (task.filler_prob)
# Fixed character vocabulary; id 0 is padding. New characters go at the end so the ids
# of existing ones never change.
VOCAB = [PAD, EOS, *"0123456789", "+", "*", "=", "|", ":", ",", ";", ">", FILLER]


class CharTokenizer:
    def __init__(self) -> None:
        self.chars = VOCAB
        self.index = {c: i for i, c in enumerate(VOCAB)}
        self.pad_id, self.eos_id = self.index[PAD], self.index[EOS]
        self.digit_ids = [self.index[str(d)] for d in range(10)]

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.index[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


@dataclass
class TaskConfig:
    op: Literal["add", "mul"] = "add"
    format: Literal["direct", "reverse", "cot"] = "cot"
    # Training operands have min_digits..train_digits digits, each operand's length drawn
    # independently and uniformly (balanced lengths matter; Lee et al., 2023).
    min_digits: int = 1
    train_digits: int = 10
    reverse_operands: bool = False
    # Data augmentation: each gap between prompt characters gets, with this probability,
    # 1..max_fillers FILLER characters. Distances between digits (and to "=") then vary
    # from problem to problem, so a solution that counts fixed offsets cannot work.
    filler_prob: float = 0.0
    max_fillers: int = 3
    # Evaluation lengths (both operands this many digits); empty -> 1..2*train_digits.
    eval_digits: list[int] = field(default_factory=list)

    def eval_lengths(self) -> list[int]:
        return self.eval_digits or list(range(1, 2 * self.train_digits + 1))


def _rev(number: int) -> str:
    return str(number)[::-1]


def _cot_add(a: int, b: int) -> str:
    sa, sb = str(a)[::-1], str(b)[::-1]
    steps, carry = [], 0
    for i in range(max(len(sa), len(sb))):
        da = int(sa[i]) if i < len(sa) else 0
        db = int(sb[i]) if i < len(sb) else 0
        total = da + db + carry
        steps.append(f"{da}+{db}+{carry}:{total:02d}")
        carry = total // 10
    return ",".join(steps)


def _cot_mul(a: int, b: int) -> str:
    steps, running = [], 0
    for i, digit in enumerate(str(b)[::-1]):
        partial = a * int(digit)
        partial_text = "0" * i + _rev(partial)  # a * b_i * 10^i, least significant first
        running += partial * 10**i
        steps.append(partial_text if i == 0 else f"{partial_text}>{_rev(running)}")
    return ";".join(steps)


def render(a: int, b: int, config: TaskConfig) -> tuple[str, str]:
    """(prompt ending in '=', completion ending in EOS)."""
    symbol = "+" if config.op == "add" else "*"
    answer = a + b if config.op == "add" else a * b
    left, right = (_rev(a), _rev(b)) if config.reverse_operands else (str(a), str(b))
    prompt = f"{left}{symbol}{right}="
    if config.format == "direct":
        completion = str(answer)
    elif config.format == "reverse":
        completion = _rev(answer)
    else:
        completion = (_cot_add(a, b) if config.op == "add" else _cot_mul(a, b)) + "|" + str(answer)
    return prompt, completion + EOS


def add_fillers(prompt: str, rng: random.Random, prob: float, max_fillers: int) -> str:
    """Insert FILLER runs into the gaps of `prompt` (before every character except the
    leading one; the prompt still ends with '=')."""
    if prob <= 0:
        return prompt
    out = [prompt[0]]
    for char in prompt[1:]:
        if rng.random() < prob:
            out.append(FILLER * rng.randint(1, max_fillers))
        out.append(char)
    return "".join(out)


def parse_answer(completion: str, config: TaskConfig) -> int | None:
    """The integer answer in a generated completion (None if malformed)."""
    text = completion.split(EOS, 1)[0].rsplit("|", 1)[-1]
    if config.format == "reverse":
        text = text[::-1]
    return int(text) if text.isdigit() else None


def sample_operand(rng: random.Random, digits: int) -> int:
    return rng.randint(10 ** (digits - 1) if digits > 1 else 0, 10**digits - 1)


def sample_problem(rng: random.Random, config: TaskConfig) -> tuple[int, int]:
    return (
        sample_operand(rng, rng.randint(config.min_digits, config.train_digits)),
        sample_operand(rng, rng.randint(config.min_digits, config.train_digits)),
    )


def true_answer(a: int, b: int, config: TaskConfig) -> int:
    return a + b if config.op == "add" else a * b
