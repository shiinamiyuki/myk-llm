"""Multi-process checks: CPU (gloo) ones run without GPUs; FSDP-on-CUDA ones use one GPU."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from myk_llm.train.optim import Muon


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reference_state(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(24, 40, bias=False), nn.Linear(40, 16, bias=False))
    optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.1)
    for step in range(3):
        torch.manual_seed(100 + step)
        model(torch.randn(8, 24)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _sharded_worker(rank: int, world_size: int, port: int, out_path: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(24, 40, bias=False), nn.Linear(40, 16, bias=False))
    fully_shard(model, mesh=init_device_mesh("cpu", (world_size,)))
    optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.1)
    for step in range(3):
        torch.manual_seed(100 + step)  # same batch on every rank -> same averaged gradient
        model(torch.randn(8, 24)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    full = {k: v.full_tensor().clone() for k, v in model.state_dict().items()}
    if rank == 0:
        torch.save(full, out_path)
    dist.barrier()
    dist.destroy_process_group()


def test_muon_on_fsdp_shards_matches_unsharded(tmp_path):
    out_path = str(tmp_path / "sharded.pt")
    mp.spawn(_sharded_worker, args=(2, _free_port(), out_path), nprocs=2, join=True)
    sharded = torch.load(out_path)
    for name, expected in _reference_state(0).items():
        torch.testing.assert_close(sharded[name], expected, rtol=1e-4, atol=1e-5)


def _fsdp_generate_worker(rank: int, port: int, out_path: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=0, world_size=1, device_id=device)
    from conftest import tiny_model_config
    from myk_llm.model import MykLLM, generate
    from myk_llm.train.config import ParallelConfig
    from myk_llm.train.distributed import DistInfo, parallelize

    torch.manual_seed(0)
    model = parallelize(
        MykLLM(tiny_model_config()).to(device),
        ParallelConfig(strategy="fsdp", compile=False),
        DistInfo(0, 1, 0, device),
    ).eval()
    ids = torch.arange(1, 9, device=device)[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = generate(model, ids, max_new_tokens=8, temperature=0.0)
    torch.save(tokens.cpu(), out_path)
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_generate_under_fsdp_matches_unsharded(tmp_path):
    """Sampling during training: the output head is sharded outside forward under FSDP."""
    from conftest import tiny_model_config
    from myk_llm.model import MykLLM, generate

    out_path = str(tmp_path / "tokens.pt")
    mp.spawn(_fsdp_generate_worker, args=(_free_port(), out_path), nprocs=1, join=True)
    torch.manual_seed(0)
    model = MykLLM(tiny_model_config()).cuda().eval()
    ids = torch.arange(1, 9, device="cuda")[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = generate(model, ids, max_new_tokens=8, temperature=0.0)
    assert torch.equal(torch.load(out_path), expected.cpu())


CROSS_DECODERS = {"project": {"layer_pattern": "LLXX"}, "layer": {"layer_pattern": "GLGX", "cross.kv": "layer"}}


def _cross_decoder_gradients(model, device) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    ids = torch.randint(0, 500, (2, 128), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(ids, seqlens=[torch.tensor([60, 68]), torch.tensor([128])]).hidden
    hidden.float().square().mean().backward()
    grads = {}
    for name, param in model.named_parameters():
        grad = param.grad.full_tensor() if hasattr(param.grad, "full_tensor") else param.grad
        grads[name] = grad.float().cpu()
    return grads


def _fsdp_cross_decoder_worker(rank: int, port: int, kv: str, out_path: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=0, world_size=1, device_id=device)
    from conftest import tiny_model_config
    from myk_llm.model import MykLLM
    from myk_llm.train.config import ParallelConfig
    from myk_llm.train.distributed import DistInfo, parallelize

    torch.manual_seed(0)
    model = parallelize(
        MykLLM(tiny_model_config(**CROSS_DECODERS[kv])).to(device),
        ParallelConfig(strategy="fsdp", compile=False, activation_checkpointing=True),
        DistInfo(0, 1, 0, device),
    )
    torch.save(_cross_decoder_gradients(model, device), out_path)
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("kv", CROSS_DECODERS)
def test_cross_decoder_gradients_under_fsdp_match_unsharded(tmp_path, kv):
    """The shared KV reaches every cross-decoder block (and, with cross.kv=layer, carries
    gradients back into the reused self-decoder block) through FSDP units and checkpointing."""
    from conftest import tiny_model_config
    from myk_llm.model import MykLLM

    out_path = str(tmp_path / "grads.pt")
    mp.spawn(_fsdp_cross_decoder_worker, args=(_free_port(), kv, out_path), nprocs=1, join=True)
    torch.manual_seed(0)
    expected = _cross_decoder_gradients(MykLLM(tiny_model_config(**CROSS_DECODERS[kv])).cuda(), "cuda")
    sharded = torch.load(out_path)
    assert sharded.keys() == expected.keys()
    for name, grad in expected.items():
        torch.testing.assert_close(sharded[name], grad, rtol=1e-3, atol=1e-5, msg=name)
