import numpy as np
import pytest

from myk_llm.data import DataConfig, SourceConfig
from myk_llm.data.config import IGNORE_INDEX
from myk_llm.data.shards import ShardLoader, ShardWriter

EOS = 0


@pytest.fixture
def shard_dir(tmp_path):
    """Documents of 1..40 tokens; token values encode (doc id, position)."""
    writer = ShardWriter(tmp_path / "src", EOS, "test-tokenizer", shard_tokens=300)
    rng = np.random.default_rng(0)
    for doc in range(200):
        writer.add([1000 * (doc + 1) + i for i in range(int(rng.integers(1, 40)))])
    writer.close()
    return tmp_path / "src"


def config(path, **kwargs) -> DataConfig:
    return DataConfig(kind="shards", seq_len=32, micro_batch_size=kwargs.pop("micro", 4),
                      sources=[SourceConfig(name="a", path=str(path))], **kwargs)


def test_windows_carry_document_layout(shard_dir):
    batch = ShardLoader(config(shard_dir)).next_batch()
    for row in range(batch.input_ids.size(0)):
        ids, targets, lengths = batch.input_ids[row], batch.targets[row], batch.seqlens[row]
        assert int(lengths.sum()) == 32
        ends = np.cumsum(lengths.numpy())[:-1]
        for end in ends:
            assert ids[end - 1] == EOS  # every internal cut follows an EOS
            assert targets[end - 1] == IGNORE_INDEX  # ...and its target is masked
        doc_of = ids // 1000
        for start, length in zip(np.concatenate(([0], ends)), lengths.numpy()):
            segment = doc_of[start : start + length]
            assert len(set(segment[segment > 0].tolist())) <= 1  # one document per segment


def test_rank_slices_tile_the_global_batch(shard_dir):
    whole = ShardLoader(config(shard_dir, micro=4), rank=0, world_size=1)
    ranks = [ShardLoader(config(shard_dir, micro=2), rank=r, world_size=2) for r in range(2)]
    for _ in range(3):
        expected = whole.next_batch().input_ids
        got = np.concatenate([loader.next_batch().input_ids for loader in ranks])
        np.testing.assert_array_equal(expected, got)


def test_resume_reproduces_the_stream(shard_dir):
    loader = ShardLoader(config(shard_dir))
    stream = iter(loader)
    for _ in range(5):
        next(stream)
    state = loader.state_dict()
    expected = [next(stream).input_ids for _ in range(3)]
    loader.close()
    resumed = ShardLoader(config(shard_dir))
    resumed.load_state_dict(state)
    stream = iter(resumed)
    for want in expected:
        np.testing.assert_array_equal(want, next(stream).input_ids)
    resumed.close()


def test_sources_are_token_balanced(shard_dir, tmp_path):
    cfg = config(shard_dir)
    cfg.sources = [SourceConfig(name="a", weight=3.0, path=str(shard_dir)), SourceConfig(name="b", weight=1.0, path=str(shard_dir))]
    loader = ShardLoader(cfg)
    for _ in range(25):
        loader.next_batch()
    a, b = (c.consumed for c in loader.cursors)
    assert abs(a / (a + b) - 0.75) < 0.02
