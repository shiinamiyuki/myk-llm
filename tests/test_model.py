import pytest
import torch
import torch.nn.functional as F

from conftest import requires_cuda, tiny_model_config
from myk_llm.model import MykLLM
from myk_llm.model.embeddings import ngram_bucket_ids
from myk_llm.train.objectives import mtp_targets

pytestmark = requires_cuda

VARIANTS = {
    "baseline": {},
    "qknorm_gate_rope_all": {"attention.qk_norm": "true", "attention.output_gate": "true", "attention.global_rope": "all"},
    "deltanet": {"layer_pattern": "GD"},
    "deltanet_no_conv": {"layer_pattern": "GD", "deltanet.use_short_conv": "false"},
    "kda": {"layer_pattern": "GD", "deltanet.kind": "kda"},
    "ngram": {"ngram.orders": "[2, 3]", "ngram.num_buckets": "1024", "ngram.dim": "32"},
    "ngram_mid_layers": {"ngram.orders": "[2]", "ngram.num_buckets": "1024", "ngram.dim": "128", "ngram.layers": "[0, 2]"},
    "residual_input": {"residual.gate": "input"},
    "residual_channel": {"residual.gate": "channel"},
    "mtp": {"mtp.depth": "2"},
}


def build(**overrides) -> MykLLM:
    torch.manual_seed(0)
    return MykLLM(tiny_model_config(**overrides)).cuda().eval()


def kl(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.kl_div(F.log_softmax(b.float(), -1), F.log_softmax(a.float(), -1), log_target=True, reduction="batchmean"))


@pytest.mark.parametrize("name", VARIANTS)
@torch.no_grad()
def test_incremental_decoding_matches_full_forward(name):
    """Prefill (longer than the sliding window) + token-by-token decode == one full pass."""
    model = build(**VARIANTS[name])
    ids = torch.randint(0, 500, (2, 200), device="cuda")
    prompt = 150
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full = model.logits(model(ids).hidden)
        out = model(ids[:, :prompt], use_cache=True)
        steps = [model.logits(out.hidden)]
        cache, context = out.cache, model.embed.context
        for t in range(prompt, ids.size(1)):
            prefix = ids[:, t - context : t] if context else None
            out = model(ids[:, t : t + 1], position_ids=torch.full((2, 1), t, device="cuda"), cache=cache,
                        use_cache=True, prefix_ids=prefix)
            cache = out.cache
            steps.append(model.logits(out.hidden))
    assert kl(full, torch.cat(steps, 1)) < 1e-3


@pytest.mark.parametrize("name", VARIANTS)
@torch.no_grad()
def test_document_masking_isolates_documents(name):
    """A packed row with document masking == each document run on its own."""
    if name.startswith("ngram"):
        pytest.skip("n-grams read across document seams by design")
    model = build(**VARIANTS[name])
    ids = torch.randint(0, 500, (1, 256), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        packed = model.logits(model(ids, seqlens=[torch.tensor([100, 156])]).hidden)
        alone = torch.cat([model.logits(model(ids[:, :100]).hidden), model.logits(model(ids[:, 100:]).hidden)], 1)
    assert kl(alone, packed) < 1e-3


def test_gates_start_as_the_baseline():
    base, gated = build(), build(**{"residual.gate": "input"})
    ids = torch.randint(0, 500, (1, 64), device="cuda")
    state = {k: v for k, v in gated.state_dict().items() if k in base.state_dict()}
    base.load_state_dict(state)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert kl(base.logits(base(ids).hidden), gated.logits(gated(ids).hidden)) < 1e-4


def test_ngram_hash_depends_only_on_the_ngram():
    tokens = torch.tensor([[7, 1, 2, 3, 9, 1, 2, 3]])
    ids = ngram_bucket_ids(tokens, order=3, context=2, num_buckets=10_007)
    assert ids[0, 1] == ids[0, 5]  # both are the 3-gram (1, 2, 3)
    assert ids[0, 0] != ids[0, 4]


def test_mtp_targets_skip_unsupervised_gaps():
    targets = torch.tensor([[10, 11, 12, -100, 14, 15]])
    assert mtp_targets(targets, 1).tolist() == [[11, 12, -100, -100, 15, -100]]
    assert mtp_targets(targets, 2).tolist() == [[12, -100, -100, -100, -100, -100]]


def test_tied_embeddings_share_storage():
    model = build()
    assert model.lm_head.weight.data_ptr() == model.embed.token.weight.data_ptr()


@pytest.mark.parametrize("name", VARIANTS)
def test_training_step_produces_finite_gradients(name):
    from myk_llm.data import Batch
    from myk_llm.train.config import ObjectiveConfig
    from myk_llm.train.objectives import Objective

    model = build(**VARIANTS[name]).train()
    ids = torch.randint(0, 500, (2, 129), device="cuda")
    batch = Batch(ids[:, :-1], ids[:, 1:], [torch.tensor([60, 68]), torch.tensor([128])])
    objective = Objective(ObjectiveConfig(z_loss=1e-4), torch.device("cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch.input_ids, seqlens=batch.seqlens, head=objective.head(batch, float(batch.num_targets)))
    out.loss.backward()
    assert torch.isfinite(out.loss)
    for param_name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), param_name


def test_ngram_layers_must_be_valid_block_indices():
    with pytest.raises(ValueError):
        tiny_model_config(**{"ngram.orders": "[2]", "ngram.layers": "[4]"})  # num_layers is 4


def test_ngram_update_reaches_only_its_layer():
    """With layers=[2], the n-gram update enters the residual stream at block 2, so the
    outputs of blocks 0 and 1 do not depend on it."""
    model = build(**{"ngram.orders": "[2]", "ngram.num_buckets": "64", "ngram.dim": "32", "ngram.layers": "[2]"})
    ids = torch.randint(0, 500, (1, 32), device="cuda")
    seen = []
    hook = model.layers[1].register_forward_hook(lambda _m, _i, out: seen.append(out[0].detach().clone()))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(ids)
        model.embed.ngram.projs[0].weight.mul_(10)
        model(ids)
    hook.remove()
    torch.testing.assert_close(seen[0], seen[1])
