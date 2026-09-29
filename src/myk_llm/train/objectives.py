"""Training objectives: next-token cross entropy (+ z-loss), multi-token prediction (MTP)
and logit distillation.

An objective runs as the model's `head` callback (inside forward, see MykLLM.forward).
All losses are divided by one `normalizer`: the number of supervised next-token
targets in the whole optimizer step (all micro-batches, all ranks), so gradient
accumulation and data parallelism give exactly the per-token mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..data.config import IGNORE_INDEX, Batch
from ..model import MykLLM, ModelOutput
from ..ops.fused_ce import fused_linear_cross_entropy
from .config import DistillConfig, ObjectiveConfig


@dataclass
class LossOutput:
    loss: torch.Tensor
    # Detached per-token sums (divide by the global token count to log a mean).
    sums: dict[str, torch.Tensor] = field(default_factory=dict)


def mtp_targets(targets: torch.Tensor, depth: int) -> torch.Tensor:
    """Targets for MTP depth k: position i predicts token i + 1 + k, supervised only when
    every next-token target in between is supervised (no document seam, no prompt)."""
    shifted = F.pad(targets[:, depth:], (0, depth), value=IGNORE_INDEX)
    valid = targets != IGNORE_INDEX
    for offset in range(1, depth + 1):
        valid &= F.pad(targets[:, offset:], (0, offset), value=IGNORE_INDEX) != IGNORE_INDEX
    return torch.where(valid, shifted, IGNORE_INDEX)


class Teacher:
    """Frozen HuggingFace teacher whose output head is restricted to the student vocab."""

    def __init__(self, config: DistillConfig, device: torch.device) -> None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(config.teacher, dtype=torch.bfloat16).to(device).eval()
        model.requires_grad_(False)
        self.decoder = model.get_decoder()
        weight = model.get_output_embeddings().weight
        if config.row_map:
            row_map = torch.from_numpy(np.load(config.row_map).astype(np.int64)).to(device)
            weight = weight.index_select(0, row_map)
        self.weight = weight.detach()

    @torch.no_grad()
    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(input_ids=input_ids).last_hidden_state


def distillation_kl_sum(
    hidden: torch.Tensor, weight: torch.Tensor, teacher_hidden: torch.Tensor, teacher_weight: torch.Tensor,
    valid: torch.Tensor, temperature: float, chunk_size: int,
) -> torch.Tensor:
    """Sum over valid tokens of T^2 * KL(teacher || student), chunked and recomputed in
    backward so only O(chunk * vocab) logits are alive at a time."""

    def chunk_kl(h: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        student = F.log_softmax((h @ weight.t()).float() / temperature, dim=-1)
        teacher = F.log_softmax((t @ teacher_weight.t()).float() / temperature, dim=-1)
        kl = (teacher.exp() * (teacher - student)).sum(-1)
        return (kl * mask).sum() * temperature**2

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.size(0), chunk_size):
        end = start + chunk_size
        total = total + checkpoint(chunk_kl, hidden[start:end], teacher_hidden[start:end], valid[start:end], use_reentrant=False)
    return total


class Objective:
    def __init__(self, config: ObjectiveConfig, device: torch.device) -> None:
        self.config = config
        self.teacher = Teacher(config.distill, device) if config.distill.teacher else None

    def head(self, batch: Batch, normalizer: float) -> Callable[[MykLLM, ModelOutput], LossOutput]:
        config = self.config

        def run(model: MykLLM, output: ModelOutput) -> LossOutput:
            weight = model.output_weight
            sums: dict[str, torch.Tensor] = {}
            ce_scale = 1.0 - config.distill.weight if self.teacher is not None else 1.0
            loss, sums["ce"], sums["z"] = fused_linear_cross_entropy(
                output.hidden, weight, batch.targets, normalizer / ce_scale, config.z_loss,
                chunk_size=config.loss_chunk_size,
            )
            if self.teacher is not None:
                valid = (batch.targets != IGNORE_INDEX).reshape(-1).float()
                kl = distillation_kl_sum(
                    output.hidden.reshape(-1, output.hidden.size(-1)), weight,
                    self.teacher.hidden(batch.input_ids).reshape(valid.numel(), -1), self.teacher.weight,
                    valid, config.distill.temperature, config.distill.chunk_size,
                )
                loss = loss + config.distill.weight * kl / normalizer
                sums["kl"] = kl.detach()
            mtp_hidden = model.mtp_hidden(output, batch.input_ids) if config.mtp_weight > 0 else []
            for depth, hidden in enumerate(mtp_hidden, start=1):
                mtp_loss, sums[f"mtp{depth}_ce"], _ = fused_linear_cross_entropy(
                    hidden, weight, mtp_targets(batch.targets, depth), normalizer / config.mtp_weight,
                    chunk_size=config.loss_chunk_size,
                )
                loss = loss + mtp_loss
            return LossOutput(loss, sums)

        return run
