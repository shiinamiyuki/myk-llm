"""Lazy-Adam n-gram tables (train/ngram_tables.py): the GPU placement matches a dense
reference restricted to the touched rows, and the host placement (with and without
prefetch) matches the GPU placement bit for bit."""

import copy

import pytest
import torch

from conftest import requires_cuda
from myk_llm.model.config import NgramEmbeddingConfig
from myk_llm.model.embeddings import NgramEmbedding
from myk_llm.train.config import NgramOptimConfig, OptimizerConfig
from myk_llm.train.ngram_tables import SparseNgramTables

pytestmark = requires_cuda

VOCAB, DIM, BUCKETS, ORDERS = 50, 8, 97, [2, 3]
LR, BETAS, EPS = 1e-2, (0.9, 0.95), 1e-8


def make_module(seed: int = 0) -> NgramEmbedding:
    torch.manual_seed(seed)
    module = NgramEmbedding(NgramEmbeddingConfig(orders=ORDERS, num_buckets=BUCKETS, dim=DIM), VOCAB, hidden_size=DIM)
    for table in module.tables:
        torch.nn.init.normal_(table.weight, std=0.02)
    return module


def make_steps(num_steps: int = 6, accum: int = 2, seed: int = 1) -> list[list[torch.Tensor]]:
    """Token ids per step and micro-batch. A small vocabulary over 97 buckets makes
    consecutive steps share many rows (the host prefetch patch path) but not all."""
    gen = torch.Generator().manual_seed(seed)
    return [[torch.randint(0, VOCAB, (2, 12), generator=gen) for _ in range(accum)] for _ in range(num_steps)]


def loss_of(features: torch.Tensor, step: int, micro: int) -> torch.Tensor:
    gen = torch.Generator(device=features.device).manual_seed(1000 * step + micro)
    target = torch.randn(features.shape, generator=gen, device=features.device)
    return (features * target).sum()


def reference(steps, rowwise: bool = False, weight_decay: float = 0.0) -> torch.Tensor:
    """Adam applied only to the rows each step looks up, on the dense stacked table."""
    module = make_module().cuda()
    weight = torch.cat([t.weight.detach() for t in module.tables]).clone()
    exp_avg = torch.zeros_like(weight)
    exp_avg_sq = torch.zeros(weight.size(0), device="cuda") if rowwise else torch.zeros_like(weight)
    for t, micro_batches in enumerate(steps, start=1):
        leaf = weight.clone().requires_grad_()
        touched = torch.zeros(weight.size(0), dtype=torch.bool, device="cuda")
        for m, ids in enumerate(micro_batches):
            rows = module.row_ids(ids.cuda())
            touched[rows.flatten()] = True
            loss_of(torch.nn.functional.embedding(rows, leaf).sum(0), t, m).backward()
        g = leaf.grad
        with torch.no_grad():
            lr = LR
            new_w = weight * (1 - lr * weight_decay)
            new_m = exp_avg.lerp(g, 1 - BETAS[0])
            if rowwise:
                new_v = exp_avg_sq * BETAS[1] + (1 - BETAS[1]) * g.square().mean(1)
                second = new_v[:, None]
            else:
                new_v = exp_avg_sq * BETAS[1] + (1 - BETAS[1]) * g * g
                second = new_v
            denom = second.sqrt() / (1 - BETAS[1] ** t) ** 0.5 + EPS
            new_w = new_w - lr / (1 - BETAS[0] ** t) * new_m / denom
            weight = torch.where(touched[:, None], new_w, weight)
            exp_avg = torch.where(touched[:, None], new_m, exp_avg)
            exp_avg_sq = torch.where(touched if rowwise else touched[:, None], new_v, exp_avg_sq)
    return weight


def run_sparse(steps, placement: str, prefetch: bool = True, rowwise: bool = False, weight_decay: float = 0.0,
               validate_every: int = 0) -> tuple[torch.Tensor, SparseNgramTables]:
    module = make_module()
    config = NgramOptimConfig(update="lazy", placement=placement, prefetch=prefetch, weight_decay=weight_decay,
                              second_moment="rowwise" if rowwise else "elementwise")
    tables = SparseNgramTables(module, config, LR, BETAS, EPS, torch.device("cuda"), max_rows_per_step=256)
    module.cuda()
    for t, micro_batches in enumerate(steps, start=1):
        upcoming = steps[t] if t < len(steps) else None
        tables.begin_step(micro_batches, upcoming)
        for m, ids in enumerate(micro_batches):
            loss_of(module.lookup(ids.cuda(), None), t, m).backward()
        tables.step()
        if validate_every and t % validate_every == 0:  # a no-grad forward between steps
            with torch.no_grad():
                module.lookup(steps[0][0].cuda(), None)
    tables.flush()
    return tables.weight.cuda(), tables


@pytest.mark.parametrize("rowwise", [False, True])
def test_gpu_placement_matches_lazy_reference(rowwise):
    steps = make_steps()
    weight, _ = run_sparse(steps, "gpu", rowwise=rowwise, weight_decay=0.1)
    torch.testing.assert_close(weight, reference(steps, rowwise=rowwise, weight_decay=0.1), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("prefetch", [True, False])
@pytest.mark.parametrize("rowwise", [False, True])
def test_host_placement_matches_gpu_exactly(prefetch, rowwise):
    steps = make_steps(num_steps=8)
    gpu, gpu_tables = run_sparse(steps, "gpu", rowwise=rowwise)
    host, host_tables = run_sparse(steps, "host", prefetch=prefetch, rowwise=rowwise, validate_every=3)
    assert torch.equal(gpu, host)
    assert torch.equal(gpu_tables.exp_avg_sq.cpu(), host_tables.exp_avg_sq)
    host_tables.close()


def test_no_grad_lookup_reads_the_current_tables():
    steps = make_steps(num_steps=3)
    _, gpu_tables = run_sparse(steps, "gpu")
    _, host_tables = run_sparse(steps, "host")
    ids = steps[0][0].cuda()
    with torch.no_grad():
        expected = torch.nn.functional.embedding(gpu_tables.module.row_ids(ids), gpu_tables.weight).sum(0)
        assert torch.equal(gpu_tables.module.lookup(ids, None), expected)
        assert torch.equal(host_tables.module.lookup(ids, None), expected)
    host_tables.close()


def test_state_dict_round_trip_and_export(tmp_path):
    steps = make_steps(num_steps=3)
    _, tables = run_sparse(steps, "host")
    tables.save(tmp_path)
    fresh = SparseNgramTables(make_module(seed=5), NgramOptimConfig(update="lazy", placement="host"), LR, BETAS, EPS,
                              torch.device("cuda"), max_rows_per_step=256)
    fresh.load(tmp_path)
    assert torch.equal(fresh.weight, tables.weight) and torch.equal(fresh.exp_avg, tables.exp_avg)
    assert fresh.step_count == tables.step_count == 3
    exported = tables.export_weights()
    assert torch.equal(torch.cat([exported[f"embed.ngram.tables.{k}.weight"] for k in range(2)]), tables.weight)
    tables.close()
    fresh.close()


def test_dense_tables_get_their_own_adamw_group():
    from myk_llm.train.optim import build_optimizers

    module = make_module()
    model = torch.nn.Module()
    model.embed = torch.nn.Module()
    model.embed.ngram = module
    model.proj = torch.nn.Linear(4, 4)
    config = OptimizerConfig(name="adamw", lr=1e-3, ngram=NgramOptimConfig(lr_scale=5.0, weight_decay=0.1))
    (adamw,) = build_optimizers(copy.deepcopy(model), config)
    ngram_group = [g for g in adamw.param_groups if len(g["params"]) == 2 and g["params"][0].shape == (BUCKETS, DIM)]
    assert len(ngram_group) == 1 and ngram_group[0]["lr"] == 5e-3 and ngram_group[0]["weight_decay"] == 0.1


def test_host_copies_do_not_race_with_other_gpu_work():
    """The staging slots are recycled allocator blocks: the copy streams must wait for
    kernels still using them (this failed in ~60% of trials before the fix)."""
    for n in [1, 2, 3, 2, 3]:
        steps = make_steps(num_steps=n)
        before = reference(steps)
        weight, tables = run_sparse(steps, "host", prefetch=False)
        torch.testing.assert_close(weight, before, rtol=1e-5, atol=1e-6)
        tables.close()
        assert torch.equal(reference(steps), before)


def dense_reference(steps, start_step: int, weight_decay: float, rowwise: bool = False) -> torch.Tensor:
    """Dense AdamW (every row, every step) on the stacked table, from step `start_step`."""
    module = make_module().cuda()
    weight = torch.cat([t.weight.detach() for t in module.tables]).clone()
    exp_avg = torch.zeros_like(weight)
    exp_avg_sq = torch.zeros(weight.size(0), 1, device="cuda") if rowwise else torch.zeros_like(weight)
    for i, micro_batches in enumerate(steps, start=1):
        t = start_step + i
        leaf = weight.clone().requires_grad_()
        for m, ids in enumerate(micro_batches):
            loss_of(torch.nn.functional.embedding(module.row_ids(ids.cuda()), leaf).sum(0), i, m).backward()
        g = leaf.grad
        with torch.no_grad():
            weight = weight * (1 - LR * weight_decay)
            exp_avg = exp_avg.lerp(g, 1 - BETAS[0])
            exp_avg_sq = exp_avg_sq * BETAS[1] + (1 - BETAS[1]) * (g.square().mean(1, keepdim=True) if rowwise else g * g)
            denom = exp_avg_sq.sqrt() / (1 - BETAS[1] ** t) ** 0.5 + EPS
            weight = weight - LR / (1 - BETAS[0] ** t) * exp_avg / denom
    return weight


@pytest.mark.parametrize("placement", ["gpu", "host"])
@pytest.mark.parametrize("rowwise", [False, True])
def test_catch_up_matches_dense_adamw(placement, rowwise):
    """With catch_up, lazy Adam equals dense AdamW (up to the closed form's eps and
    bias-correction approximations, which vanish this late in training: step 10^4)."""
    steps = make_steps(num_steps=10)
    module = make_module()
    config = NgramOptimConfig(update="lazy", placement=placement, weight_decay=0.1, catch_up=True,
                              second_moment="rowwise" if rowwise else "elementwise")
    tables = SparseNgramTables(module, config, LR, BETAS, EPS, torch.device("cuda"), max_rows_per_step=256)
    tables.step_count = 10_000
    tables.state[3].fill_(10_000)  # every row up to date at the start
    module.cuda()
    for t, micro_batches in enumerate(steps, start=1):
        tables.begin_step(micro_batches, steps[t] if t < len(steps) else None)
        for m, ids in enumerate(micro_batches):
            loss_of(module.lookup(ids.cuda(), None), t, m).backward()
        tables.step()
    exported = tables.export_weights()
    weight = torch.cat([exported[f"embed.ngram.tables.{k}.weight"] for k in range(2)]).cuda()
    expected = dense_reference(steps, start_step=10_000, weight_decay=0.1, rowwise=rowwise)
    torch.testing.assert_close(weight, expected, rtol=1e-4, atol=1e-6)
    # Without catch_up the rows a step skipped are behind dense AdamW.
    lazy = reference(steps, rowwise=rowwise, weight_decay=0.1)
    assert (lazy - expected).abs().max() > 100 * (weight - expected).abs().max()
    with torch.no_grad():  # no-grad lookups see the caught-up rows
        ids = steps[0][0].cuda()
        rows = module.row_ids(ids)
        torch.testing.assert_close(module.lookup(ids, None), torch.nn.functional.embedding(rows, expected).sum(0),
                                   rtol=1e-4, atol=1e-6)
    tables.close()
