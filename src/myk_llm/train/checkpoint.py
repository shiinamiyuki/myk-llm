"""Training checkpoints with torch.distributed.checkpoint (DCP).

    <run>/checkpoints/step_0000500/
        .metadata, __*_*.distcp     sharded model + optimizer state (DCP; any GPU count can load)
        trainer.json                step, tokens seen, world size
        data_rank{r}.pt             each rank's data-loader state
        model.safetensors, model.yaml, tokenizer files   portable model (optional; for eval / chat)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)

from ..model import MykLLM, save_model
from .distributed import DistInfo, barrier

STEP_PREFIX = "step_"


def step_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"{STEP_PREFIX}{step:07d}"


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "checkpoints").glob(f"{STEP_PREFIX}*"))
    complete = [c for c in candidates if (c / "trainer.json").exists()]
    return complete[-1] if complete else None


def _app_state(model: MykLLM, optimizers: list[torch.optim.Optimizer]) -> dict[str, Any]:
    model_state, optim_state = get_state_dict(model, optimizers)
    return {"model": model_state, "optim": optim_state}


def save_checkpoint(
    directory: Path,
    model: MykLLM,
    optimizers: list[torch.optim.Optimizer],
    trainer_state: dict[str, Any],
    data_state: dict[str, Any],
    info: DistInfo,
    export_weights: bool,
    tokenizer: str | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dcp.save(_app_state(model, optimizers), checkpoint_id=str(directory))
    torch.save(data_state, directory / f"data_rank{info.rank}.pt")
    if export_weights:
        export_model(directory, model, info, tokenizer)
    barrier()
    if info.is_main:  # written last: its presence marks the checkpoint complete
        (directory / "trainer.json").write_text(json.dumps(trainer_state, indent=2), encoding="utf-8")
    barrier()


def export_model(directory: Path, model: MykLLM, info: DistInfo, tokenizer: str | None = None) -> None:
    """Gather full weights on rank 0 and write a portable model directory."""
    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if info.is_main:
        save_model(directory, model.config, state, tokenizer)


def load_checkpoint(
    directory: Path, model: MykLLM, optimizers: list[torch.optim.Optimizer] | None, info: DistInfo
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load model (+ optimizer) state in place. Returns (trainer_state, this rank's data state)."""
    if optimizers:
        state = _app_state(model, optimizers)
        dcp.load(state, checkpoint_id=str(directory))
        set_state_dict(model, optimizers, model_state_dict=state["model"], optim_state_dict=state["optim"])
    else:
        model_state = get_model_state_dict(model)
        dcp.load({"model": model_state}, checkpoint_id=str(directory))
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        set_model_state_dict(model, model_state)
    trainer_state = json.loads((directory / "trainer.json").read_text(encoding="utf-8"))
    data_path = directory / f"data_rank{info.rank}.pt"
    data_state = torch.load(data_path, weights_only=False) if data_path.exists() else None
    return trainer_state, data_state


def prune_checkpoints(run_dir: Path, keep_last: int, keep_every: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted((run_dir / "checkpoints").glob(f"{STEP_PREFIX}*"))
    for path in checkpoints[:-keep_last]:
        step = int(path.name.removeprefix(STEP_PREFIX))
        if keep_every and step % keep_every == 0:
            continue
        shutil.rmtree(path)
