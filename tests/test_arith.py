import random

import pytest
import torch

from myk_llm.arith.data import PackedArithmetic, encode_example
from myk_llm.arith.model import digit_indices
from myk_llm.arith.tasks import CharTokenizer, TaskConfig, parse_answer, render, sample_problem, true_answer
from myk_llm.data.config import IGNORE_INDEX


@pytest.mark.parametrize("op", ["add", "mul"])
@pytest.mark.parametrize("fmt", ["direct", "reverse", "cot"])
@pytest.mark.parametrize("reverse_operands", [False, True])
def test_rendered_answers_parse_back(op, fmt, reverse_operands):
    task = TaskConfig(op=op, format=fmt, train_digits=6, reverse_operands=reverse_operands)
    rng = random.Random(0)
    for _ in range(500):
        a, b = sample_problem(rng, task)
        prompt, completion = render(a, b, task)
        assert prompt.endswith("=") and completion.endswith("$")
        assert parse_answer(completion, task) == true_answer(a, b, task)


def test_cot_formats():
    assert render(123, 989, TaskConfig(op="add"))[1] == "3+9+0:12,2+8+1:11,1+9+1:11|1112$"
    assert render(37, 24, TaskConfig(op="mul"))[1] == "841;047>888|888$"


def test_loss_covers_only_the_completion():
    tokenizer = CharTokenizer()
    ids, targets = encode_example(tokenizer, "12+3=", "15$")
    assert tokenizer.decode(ids) == "12+3=15$"
    supervised = [tokenizer.decode([t]) for t in targets if t != IGNORE_INDEX]
    assert supervised == ["1", "5", "$"] and targets[-1] == IGNORE_INDEX


def test_packed_rows_are_full_and_segmented():
    batch = next(iter(PackedArithmetic(TaskConfig(), batch_rows=4, row_len=256, seed=0)))
    assert batch.input_ids.shape == (4, 256)
    for lengths in batch.seqlens:
        assert int(lengths.sum()) == 256


def test_digit_indices_count_within_numbers():
    tokenizer = CharTokenizer()
    ids = torch.tensor([tokenizer.encode("123+45=5")])
    assert digit_indices(ids, torch.tensor(tokenizer.digit_ids)).tolist() == [[1, 2, 3, 0, 1, 2, 0, 1]]


def test_greedy_never_emits_ids_outside_the_character_set():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from conftest import tiny_model_config
    from myk_llm.arith.model import ArithModel, DigitEmbeddingConfig

    torch.manual_seed(0)
    model = ArithModel(tiny_model_config(vocab_size=32), DigitEmbeddingConfig()).cuda().eval()
    original = model.lm.logits

    def unused_id_wins(hidden):  # make an id outside the character set the argmax everywhere
        logits = original(hidden)
        logits[..., 31] = logits.max() + 100
        return logits

    model.lm.logits = unused_id_wins
    completions = model.greedy_complete(["12+34=", "5*6="], max_new_tokens=8)
    assert all(set(c) <= set(CharTokenizer().chars) for c in completions)
