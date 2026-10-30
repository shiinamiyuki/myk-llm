from types import SimpleNamespace

import numpy as np
import pytest
import torch

from myk_llm.data.config import IGNORE_INDEX
from myk_llm.lc.data import PackedRetrieval, answer_targets
from myk_llm.lc.sweep import effective_length
from myk_llm.lc.tasks import EOS, KEY_BASE, NOISE_BASE, QUERY, SEP, VOCAB_SIZE, TaskConfig, make_copy_example, make_example, resolve


@pytest.mark.parametrize("hops, cot", [([1], False), ([1, 2, 3, 4], False), ([1, 2, 3, 4], True)])
def test_answers_resolve_through_the_context(hops, cot):
    task = TaskConfig(num_records=12, num_queries=5, hops=hops, cot=cot)
    rng = np.random.default_rng(0)
    for length in (300, 2048, 5000):
        example = make_example(rng, task, length)
        assert len(example.ids) == length and example.ids.max() < VOCAB_SIZE
        assert len(example.answers) == len(example.depths) == len(example.hops) == 5
        for (start, end), depth, h in zip(example.answers, example.depths, example.hops):
            prompt = example.ids[start - task.key_len - 1 : start].tolist()
            assert prompt[0] == QUERY
            answer = example.ids[start:end].tolist()
            assert answer[-1] == EOS
            assert answer[-1 - task.value_len : -1] == resolve(example.ids, prompt[1:], task.key_len)
            assert len(answer) == task.value_len + 1 + (task.key_len * (h - 1) if cot else 0)
            assert 0.0 <= depth < 1.0 and h in hops


def test_keys_are_distinct_and_noise_fills_the_rest():
    task = TaskConfig(num_records=16, num_queries=4, hops=[1, 3])
    example = make_example(np.random.default_rng(1), task, 4096)
    context = example.ids[: example.ids.tolist().index(QUERY)]
    is_key = (context >= KEY_BASE) & (context < KEY_BASE + 128)
    # Every record is "key ... ;" after noise or another record: count the keys that start one.
    starts = [i for i in range(len(context)) if is_key[i] and (i == 0 or context[i - 1] == SEP or context[i - 1] >= NOISE_BASE)]
    record_keys = {tuple(context[i : i + 2]) for i in starts}
    assert len(record_keys) == int((context == SEP).sum())  # one distinct key per record
    assert (context >= NOISE_BASE).sum() > 0.9 * len(context)


def test_records_scale_with_length_when_asked():
    task = TaskConfig(min_len=2048, max_len=2048, num_records=128, scale_records=True, num_queries=16)
    assert task.records_at(2048) == 128 and task.records_at(8192) == 512 and task.records_at(256) == 16
    with pytest.raises(ValueError):
        make_example(np.random.default_rng(0), TaskConfig(num_records=300, num_queries=4), 2048)  # does not fit


def test_targets_cover_only_answers():
    example = make_example(np.random.default_rng(2), TaskConfig(), 1000)
    targets = answer_targets(example)
    supervised = np.flatnonzero(targets != IGNORE_INDEX)
    expected = [p for start, end in example.answers for p in range(start - 1, end - 1)]
    assert supervised.tolist() == expected
    assert (targets[supervised] == example.ids[supervised + 1]).all()


def test_packed_rows_are_full_and_segmented():
    batch = next(iter(PackedRetrieval(TaskConfig(min_len=256, max_len=1024), batch_rows=4, seed=0)))
    assert batch.input_ids.shape == (4, 1024)
    assert all(int(s.sum()) == 1024 for s in batch.seqlens)
    assert batch.num_targets > 0


def test_copy_examples_supervise_the_predictable_part_of_the_repeat():
    task = TaskConfig(copy_fraction=0.5, copy_min=8, copy_max=8)
    example = make_copy_example(np.random.default_rng(0), task)
    targets = answer_targets(example)
    assert len(example.ids) == 16 and (example.ids[:8] == example.ids[8:]).all()
    assert np.flatnonzero(targets != IGNORE_INDEX).tolist() == list(range(8, 15))  # all but the repeat's first token
    batch = next(iter(PackedRetrieval(TaskConfig(max_len=512, copy_fraction=0.5, copy_min=8, copy_max=8), batch_rows=8, seed=0)))
    assert any(16 in s.tolist() for s in batch.seqlens)  # copy documents were packed in


def test_mixing_options_keep_the_task_valid():
    task = TaskConfig(num_records=8, num_queries=4, min_records=1)
    example = make_example(np.random.default_rng(0), task, 200, records=2)
    assert len(example.answers) == 2  # queries capped at the item count
    with pytest.raises(ValueError):
        TaskConfig(max_len=128, copy_fraction=0.1, copy_max=100)


def test_late_mean_skips_evaluations_before_the_transition():
    from myk_llm.lc.train import late_mean

    late = [(2500, {512: 0.1}), (3000, {512: 0.95}), (3500, {512: 0.97})]
    assert late_mean(late, 3000, [512]) == {512: pytest.approx(0.96)}
    assert late_mean(late, None, [512]) == {512: pytest.approx((0.1 + 0.95 + 0.97) / 3)}
    assert late_mean([], None, [512]) == {}


def test_effective_length_stops_at_the_first_failure():
    assert effective_length({"512": 1.0, "1024": 0.95, "2048": 0.4, "4096": 0.95}, 0.9) == 1024
    assert effective_length({"512": 0.2}, 0.5) == 0


class Oracle(torch.nn.Module):
    """Predicts the true next token everywhere except at the given (row, position)s."""

    def __init__(self, wrong: set[tuple[int, int]]) -> None:
        super().__init__()
        self.wrong, self.device = wrong, torch.device("cpu")

    def forward(self, ids, seqlens=None):
        following = torch.roll(ids, -1, dims=1)
        for row, pos in self.wrong:
            following[row, pos] = (following[row, pos] + 1) % VOCAB_SIZE
        return SimpleNamespace(hidden=torch.nn.functional.one_hot(following, VOCAB_SIZE).float())

    def logits(self, hidden):
        return hidden


def test_scoring_marks_exactly_the_queries_with_a_wrong_answer_token():
    from myk_llm.lc.eval import score_batch

    task = TaskConfig(num_records=6, num_queries=3)
    examples = [make_example(np.random.default_rng(i), task, 256) for i in range(2)]
    assert score_batch(Oracle(set()), examples) == [True] * 6
    start, end = examples[1].answers[2]
    wrong = {(1, end - 2), (0, 5)}  # last answer token of row 1's third query; a context position
    assert score_batch(Oracle(wrong), examples) == [True] * 5 + [False]


def test_evaluation_can_draw_other_hop_counts():
    from myk_llm.lc.eval import evaluate

    task = TaskConfig(num_records=8, num_queries=4, hops=[1, 2])
    result = evaluate(Oracle(set()), task, [512], queries=40, seed=0, hops=[5, 6])
    assert set(result["hops"][512]) == {5, 6} and result["accuracy"][512] == 1.0


def test_report_tabulates_final_scans(tmp_path):
    import json

    from myk_llm.lc.sweep import report

    for seed, six_hops in ((0, 0.5), (1, 0.7)):
        run = f"chain-loop-s{seed}"
        result = {
            "run_name": run, "train_len": 2048, "train_minutes": 1.0,
            "accuracy": {"2048": 1.0}, "depth": {"2048": [1.0]}, "hops": {"2048": {"1": 1.0}},
            "scan": [{"iterations": 8, "accuracy": {"2048": 0.8}, "hops": {"2048": {"1": 1.0, "6": six_hops}}}],
        }
        (tmp_path / run).mkdir()
        (tmp_path / run / "results.json").write_text(json.dumps(result))
    assert "| chain-loop | 8 | 2 | 100 | 60 |" in report(tmp_path, at=2048)
