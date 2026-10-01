from pathlib import Path

import pytest

from myk_llm.configuration import apply_overrides, from_dict, load_config
from myk_llm.model import ModelConfig
from myk_llm.train.config import TrainConfig

ROOT = Path(__file__).parent.parent / "configs"
CONFIGS = sorted(p for p in ROOT.rglob("*.yaml") if "sweeps" not in p.parts)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_config_parses(path):
    from myk_llm.arith.train import ArithConfig

    config = load_config(ArithConfig if path.relative_to(ROOT).parts[0] == "arith" else TrainConfig, path)
    assert config.model.num_layers > 0


def test_arith_sweeps_expand():
    import yaml

    from myk_llm.arith.sweep import plan

    for path in (ROOT / "arith" / "sweeps").glob("*.yaml"):
        for _, config_path, overrides in plan(yaml.safe_load(path.read_text())):
            from myk_llm.arith.train import ArithConfig

            load_config(ArithConfig, ROOT.parent / config_path, overrides)


def test_overrides_are_typed():
    config = load_config(TrainConfig, None, ["optimizer.lr=3e-4", "model.ngram.orders=[2, 3]", "data.document_masking=false"])
    assert config.optimizer.lr == 3e-4 and config.model.ngram.orders == [2, 3] and not config.data.document_masking


def test_unknown_key_is_an_error():
    with pytest.raises(KeyError):
        from_dict(ModelConfig, {"num_layer": 3})


def test_type_mismatch_is_an_error():
    with pytest.raises(TypeError):
        from_dict(ModelConfig, {"num_layers": "three"})


def test_base_inheritance_merges_mappings(tmp_path):
    (tmp_path / "a.yaml").write_text("model: {num_layers: 3, attention: {num_heads: 4, num_kv_heads: 2}}\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\nmodel: {attention: {qk_norm: true}}\n")
    config = load_config(TrainConfig, tmp_path / "b.yaml")
    assert config.model.num_layers == 3 and config.model.attention.num_heads == 4 and config.model.attention.qk_norm


def test_apply_overrides_creates_nested_keys():
    assert apply_overrides({}, ["a.b.c=1"]) == {"a": {"b": {"c": 1}}}
