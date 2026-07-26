from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from safetensors import safe_open
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm
from model import MykLLM, MykLLMConfig


DEFAULT_TOKENIZER_PATH = "/home/miyuki/repos/myk-llm/_download_models/Qwen3.5-0.8B"
IGNORE_INDEX = -100
PRETRAINED_EMBEDDING_KEYS = (
	"model.language_model.embed_tokens.weight",
	"model.embed_tokens.weight",
)


class CombinedOptimizer(torch.optim.Optimizer):
	def __init__(self, *optimizers: torch.optim.Optimizer) -> None:
		if not optimizers:
			raise ValueError("CombinedOptimizer requires at least one optimizer")

		self.optimizers = optimizers
		all_params: list[torch.nn.Parameter] = []
		seen: set[int] = set()
		for optimizer in optimizers:
			for group in optimizer.param_groups:
				for parameter in cast(list[torch.nn.Parameter], group["params"]):
					if id(parameter) in seen:
						continue
					seen.add(id(parameter))
					all_params.append(parameter)

		super().__init__(all_params, defaults={})
		self.param_groups = [group for optimizer in optimizers for group in optimizer.param_groups]
		self.defaults = {}

	def zero_grad(self, set_to_none: bool = True) -> None:
		for optimizer in self.optimizers:
			optimizer.zero_grad(set_to_none=set_to_none)

	def step(self, closure=None) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
		if closure is not None:
			with torch.enable_grad():
				closure()
		for optimizer in self.optimizers:
			optimizer.step()
		return None

	def state_dict(self) -> dict[str, object]:
		return {
			"optimizer_type": type(self).__name__,
			"optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
		}

	def load_state_dict(self, state_dict: dict[str, object]) -> None:
		optimizer_states = state_dict.get("optimizers")
		if not isinstance(optimizer_states, list) or len(optimizer_states) != len(self.optimizers):
			raise ValueError("combined optimizer checkpoint does not match the configured optimizers")
		for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
			optimizer.load_state_dict(cast(dict[str, object], optimizer_state))


@dataclass
class OptimizerParameterSplit:
	muon_params: list[torch.nn.Parameter]
	adamw_decay_params: list[torch.nn.Parameter]
	adamw_no_decay_params: list[torch.nn.Parameter]

	@property
	def adamw_params(self) -> list[torch.nn.Parameter]:
		return [*self.adamw_decay_params, *self.adamw_no_decay_params]


@dataclass
class TokenDataset:
	token_ids: np.ndarray
	spans: np.ndarray
	loss_mask: np.ndarray | None = None

	@property
	def num_tokens(self) -> int:
		return int(self.token_ids.shape[0])


def load_json_data(path: str) -> object:
	with Path(path).open(encoding="utf-8") as handle:
		return json.load(handle)


def tokenized_dataset_paths(prefix: Path) -> tuple[Path, Path, Path, Path]:
	return (
		prefix.with_suffix(".tokens.bin"),
		prefix.with_suffix(".loss_mask.bin"),
		prefix.with_suffix(".spans.npy"),
		prefix.with_suffix(".meta.json"),
	)


def load_tokenized_dataset(prefix: str) -> TokenDataset:
	tokens_path, loss_mask_path, spans_path, meta_path = tokenized_dataset_paths(Path(prefix))
	if not meta_path.exists():
		raise FileNotFoundError(
			f"missing tokenized dataset metadata {meta_path}; run myk-llm/tokenization.py first"
		)
	if not tokens_path.exists():
		raise FileNotFoundError(f"missing token ids file {tokens_path}")
	if not spans_path.exists():
		raise FileNotFoundError(f"missing spans file {spans_path}")

	metadata = cast(dict[str, object], load_json_data(str(meta_path)))
	num_tokens = int(cast(int, metadata["num_tokens"]))
	token_dtype = np.dtype(cast(str, metadata.get("token_dtype", "uint32")))
	if token_dtype != np.dtype(np.uint32):
		raise ValueError(f"unsupported token dtype {token_dtype}; expected uint32")
	token_ids = np.memmap(tokens_path, dtype=token_dtype, mode="r", shape=(num_tokens,))
	spans = np.load(spans_path)
	if spans.ndim != 2 or spans.shape[1] != 2:
		raise ValueError(f"tokenized spans must have shape [num_spans, 2], got {spans.shape}")
	has_loss_mask = bool(metadata.get("has_loss_mask", False))
	loss_mask = None
	if has_loss_mask:
		if not loss_mask_path.exists():
			raise FileNotFoundError(f"missing loss mask file {loss_mask_path}")
		loss_mask = np.memmap(loss_mask_path, dtype=np.uint8, mode="r", shape=(num_tokens,))
	return TokenDataset(token_ids=token_ids, spans=spans, loss_mask=loss_mask)


def build_sampling_weights(spans: np.ndarray, seq_len: int) -> torch.Tensor:
	span_lengths = spans[:, 1] - spans[:, 0]
	valid_windows = np.maximum(span_lengths - seq_len, 0)
	if not np.any(valid_windows > 0):
		raise ValueError("dataset is too short for the configured sequence length")
	return torch.tensor(valid_windows, dtype=torch.float32)


def get_batch(
	dataset: TokenDataset,
	sampling_weights: torch.Tensor,
	seq_len: int,
	batch_size: int,
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
	span_indices = torch.multinomial(sampling_weights, num_samples=batch_size, replacement=True).tolist()
	input_batch: list[np.ndarray] = []
	target_batch: list[np.ndarray] = []

	for span_index in span_indices:
		for _ in range(32):
			span_start = int(dataset.spans[span_index, 0])
			span_end = int(dataset.spans[span_index, 1])
			max_local_start = span_end - span_start - seq_len - 1
			if max_local_start < 0:
				raise ValueError("encountered a span shorter than the configured sequence length")
			start = span_start + int(torch.randint(0, max_local_start + 1, (1,)).item())
			inputs = np.asarray(dataset.token_ids[start : start + seq_len], dtype=np.int64)
			targets = np.asarray(dataset.token_ids[start + 1 : start + seq_len + 1], dtype=np.int64)
			if dataset.loss_mask is not None:
				target_mask = np.asarray(dataset.loss_mask[start + 1 : start + seq_len + 1], dtype=np.bool_)
				if not target_mask.any():
					continue
				targets = np.where(target_mask, targets, IGNORE_INDEX)
			input_batch.append(inputs)
			target_batch.append(targets)
			break
		else:
			raise ValueError("unable to sample a chat window containing assistant tokens")

	inputs = torch.as_tensor(np.stack(input_batch), dtype=torch.long, device=device)
	targets = torch.as_tensor(np.stack(target_batch), dtype=torch.long, device=device)
	return inputs, targets


def window_segment_lengths(bounds: np.ndarray, start: int, seq_len: int) -> torch.Tensor:
	"""Per-document lengths for a packed window [start, start+seq_len).

	`bounds` are the conversation end offsets (spans[:, 1]); a boundary strictly
	inside the window is a segment cut point. The returned int32 lengths sum to
	seq_len and feed the model's block-diagonal (document-masked) attention so
	packed conversations never attend across each other.
	"""
	lo, hi = start, start + seq_len
	left = int(np.searchsorted(bounds, lo, side="right"))
	right = int(np.searchsorted(bounds, hi, side="left"))
	points = np.concatenate(([lo], bounds[left:right], [hi])).astype(np.int64)
	return torch.from_numpy(np.diff(points).astype(np.int32))


def get_chat_batch(
	dataset: TokenDataset,
	seq_len: int,
	batch_size: int,
	device: torch.device,
	document_masking: bool,
	generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
	"""Sample packed windows across the concatenated SFT stream.

	Windows are drawn uniformly over token positions (standard packed-SFT sampling);
	loss is masked to assistant tokens via the dataset loss mask; per-window document
	lengths drive block-diagonal attention when `document_masking` is set. Unlike
	`get_batch`, windows span conversation boundaries, so short conversations (the
	norm for SFT) are used rather than dropped.
	"""
	if dataset.loss_mask is None:
		raise ValueError("chat batching requires a loss mask; run tokenization.py --mode chat")
	num_tokens = dataset.num_tokens
	if num_tokens < seq_len + 1:
		raise ValueError(
			f"chat dataset has {num_tokens} tokens but needs at least seq_len+1={seq_len + 1}; "
			"use a smaller --seq-len or more data"
		)
	bounds = dataset.spans[:, 1]
	starts = torch.randint(0, num_tokens - seq_len, (batch_size,), generator=generator).tolist()
	input_batch: list[np.ndarray] = []
	target_batch: list[np.ndarray] = []
	seqlens_batch: list[torch.Tensor] = []
	for start in starts:
		inputs = np.asarray(dataset.token_ids[start : start + seq_len], dtype=np.int64)
		targets = np.asarray(dataset.token_ids[start + 1 : start + seq_len + 1], dtype=np.int64)
		mask = np.asarray(dataset.loss_mask[start + 1 : start + seq_len + 1], dtype=np.bool_)
		targets = np.where(mask, targets, IGNORE_INDEX)
		input_batch.append(inputs)
		target_batch.append(targets)
		if document_masking:
			seqlens_batch.append(window_segment_lengths(bounds, start, seq_len))

	inputs_t = torch.as_tensor(np.stack(input_batch), dtype=torch.long, device=device)
	targets_t = torch.as_tensor(np.stack(target_batch), dtype=torch.long, device=device)
	return inputs_t, targets_t, (seqlens_batch if document_masking else None)


def build_chat_eval_batches(
	dataset: TokenDataset,
	seq_len: int,
	batch_size: int,
	num_batches: int,
	seed: int,
	document_masking: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]:
	"""A fixed, seeded set of packed SFT windows for periodic held-out loss."""
	generator = torch.Generator().manual_seed(seed)
	batches: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]] = []
	for _ in range(num_batches):
		input_ids, targets, seqlens = get_chat_batch(
			dataset, seq_len, batch_size, torch.device("cpu"), document_masking, generator=generator
		)
		batches.append((input_ids, targets, seqlens if seqlens is not None else []))
	return batches


def build_config(args: argparse.Namespace, vocab_size: int) -> MykLLMConfig:
	return MykLLMConfig(
		vocab_size=vocab_size,
		max_seq_len=args.seq_len,
		head_dim=args.head_dim,
		num_kv_heads=args.num_kv_heads,
		num_attention_heads=args.num_attention_heads,
		num_layers=args.num_layers,
		full_attn_period=args.full_attn_period,
		sliding_window_size=args.sliding_window_size,
		ffn_multiplier=args.ffn_multiplier,
		ffn_multiple_of=args.ffn_multiple_of,
		dropout=args.dropout,
		tie_embeddings=args.tie_embeddings,
		gradient_checkpointing=args.gradient_checkpointing,
	)


def resolve_resume_model_config(resume_path: str) -> MykLLMConfig | None:
	if not resume_path:
		return None
	checkpoint = cast(dict[str, object], torch.load(resume_path, map_location="cpu"))
	model_config = checkpoint.get("model_config")
	if not isinstance(model_config, dict):
		raise ValueError("resume checkpoint is missing a valid model_config")
	model_config_values = cast(dict[str, Any], model_config)
	return MykLLMConfig(**model_config_values)


def load_pretrained_token_embedding(source_path: str) -> torch.Tensor:
	path = Path(source_path)
	tensor_path: Path | None = None
	tensor_key: str | None = None

	if path.is_dir():
		index_path = path / "model.safetensors.index.json"
		if index_path.exists():
			index = cast(dict[str, object], load_json_data(str(index_path)))
			weight_map = index.get("weight_map")
			if not isinstance(weight_map, dict):
				raise ValueError(f"invalid safetensors index: {index_path}")
			for candidate_key in PRETRAINED_EMBEDDING_KEYS:
				filename = weight_map.get(candidate_key)
				if isinstance(filename, str):
					tensor_path = path / filename
					tensor_key = candidate_key
					break
		else:
			single_tensor_path = path / "model.safetensors"
			if single_tensor_path.exists():
				tensor_path = single_tensor_path

	elif path.suffix == ".safetensors" and path.exists():
		tensor_path = path

	if tensor_path is None:
		raise FileNotFoundError(f"could not find a safetensors model under {source_path}")

	with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
		if tensor_key is None:
			for candidate_key in PRETRAINED_EMBEDDING_KEYS:
				if candidate_key in handle.keys():
					tensor_key = candidate_key
					break
		if tensor_key is None:
			raise KeyError(
				f"could not find any supported embedding tensor in {tensor_path}; tried {PRETRAINED_EMBEDDING_KEYS}"
			)
		return cast(torch.Tensor, handle.get_tensor(tensor_key))


def gather_pretrained_embedding_rows(embedding_weight: torch.Tensor, row_map_path: str) -> torch.Tensor:
	"""Select the surviving rows of a full pretrained embedding for a pruned vocab.

	`row_map_path` points at the int array written by scripts/prune_tokenizer.py,
	where row_map[new_id] = old_id. The result has one row per pruned-vocab token,
	warm-started from the matching Qwen row.
	"""
	row_map = np.load(row_map_path)
	if row_map.ndim != 1:
		raise ValueError(f"embedding row map must be 1-D, got shape {tuple(row_map.shape)}")
	row_index = torch.from_numpy(np.ascontiguousarray(row_map).astype(np.int64))
	if int(row_index.min()) < 0 or int(row_index.max()) >= embedding_weight.shape[0]:
		raise ValueError(
			f"embedding row map ids out of range [0, {embedding_weight.shape[0]}); "
			f"got min {int(row_index.min())} max {int(row_index.max())}"
		)
	return embedding_weight.index_select(0, row_index)


def initialize_token_embedding_from_pretrained(model: MykLLM, embedding_weight: torch.Tensor) -> None:
	model_weight = model.token_embedding.weight
	if tuple(embedding_weight.shape) != tuple(model_weight.shape):
		raise ValueError(
			"pretrained embedding shape does not match model token embedding shape: "
			f"expected {tuple(model_weight.shape)}, got {tuple(embedding_weight.shape)}"
		)
	with torch.no_grad():
		model_weight.copy_(embedding_weight.to(dtype=model_weight.dtype, device=model_weight.device))


def maybe_freeze_token_embedding(model: MykLLM, freeze_token_embedding: bool) -> None:
	if freeze_token_embedding:
		model.token_embedding.weight.requires_grad_(False)


def checkpoint_payload(
	model: MykLLM,
	optimizer: torch.optim.Optimizer,
	scheduler: torch.optim.lr_scheduler.LRScheduler,
	args: argparse.Namespace,
	step: int,
	data_loader: object | None = None,
) -> dict[str, object]:
	payload: dict[str, object] = {
		"step": step,
		"model_config": vars(model.config),
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict(),
		"training_args": vars(args),
	}
	if data_loader is not None:
		payload["data_loader_state"] = cast(Any, data_loader).state_dict()
	return payload


def save_checkpoint(
	checkpoint_path: Path,
	model: MykLLM,
	optimizer: torch.optim.Optimizer,
	scheduler: torch.optim.lr_scheduler.LRScheduler,
	args: argparse.Namespace,
	step: int,
	data_loader: object | None = None,
) -> None:
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(checkpoint_payload(model, optimizer, scheduler, args, step, data_loader), checkpoint_path)


def prune_old_step_checkpoints(save_dir: Path, keep_last: int) -> None:
	step_checkpoints = sorted(save_dir.glob("step_*.pt"))
	if keep_last > 0:
		step_checkpoints = step_checkpoints[:-keep_last]
	for checkpoint_path in step_checkpoints:
		if checkpoint_path.exists():
			checkpoint_path.unlink()


def build_progress_checkpoint_steps(total_steps: int, progress_save_ratio: float) -> dict[int, int]:
	if progress_save_ratio <= 0.0:
		return {}

	progress_steps: dict[int, int] = {}
	num_full_intervals = int(math.floor(1.0 / progress_save_ratio))
	for interval_index in range(1, num_full_intervals + 1):
		progress = min(1.0, interval_index * progress_save_ratio)
		step = max(1, min(total_steps, math.ceil(total_steps * progress)))
		percent = int(round(progress * 100))
		progress_steps[step] = max(progress_steps.get(step, 0), percent)

	progress_steps[total_steps] = max(progress_steps.get(total_steps, 0), 100)
	return progress_steps


def load_checkpoint(
	checkpoint_path: str,
	model: MykLLM,
	optimizer: torch.optim.Optimizer | None = None,
	scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
	device: torch.device | None = None,
	data_loader: object | None = None,
) -> int:
	checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
	model.load_state_dict(cast(dict[str, torch.Tensor], checkpoint["model_state_dict"]))
	if optimizer is not None and "optimizer_state_dict" in checkpoint:
		optimizer.load_state_dict(cast(dict[str, object], checkpoint["optimizer_state_dict"]))
		move_optimizer_state_to_device(optimizer, device or model.device)
	if scheduler is not None and "scheduler_state_dict" in checkpoint:
		scheduler.load_state_dict(cast(dict[str, object], checkpoint["scheduler_state_dict"]))
	if data_loader is not None and "data_loader_state" in checkpoint:
		cast(Any, data_loader).load_state_dict(checkpoint["data_loader_state"])
	return int(checkpoint.get("step", 0))


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
	if isinstance(optimizer, CombinedOptimizer):
		for inner_optimizer in optimizer.optimizers:
			move_optimizer_state_to_device(inner_optimizer, device)
		return

	for state in optimizer.state.values():
		if not isinstance(state, dict):
			continue
		for key, value in state.items():
			if isinstance(value, torch.Tensor) and value.is_floating_point():
				state[key] = value.to(device=device, dtype=torch.float32)


def split_optimizer_parameters(model: MykLLM, args: argparse.Namespace) -> OptimizerParameterSplit:
	muon_params: list[torch.nn.Parameter] = []
	adamw_decay_params: list[torch.nn.Parameter] = []
	adamw_no_decay_params: list[torch.nn.Parameter] = []
	seen: set[int] = set()
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad or id(parameter) in seen:
			continue
		seen.add(id(parameter))
		use_muon = (
			args.optimizer == "muon"
			and parameter.ndim == 2
			and name not in {"token_embedding.weight", "lm_head.weight"}
		)
		if use_muon:
			muon_params.append(parameter)
		elif name == "token_embedding.weight" or parameter.ndim < 2:
			adamw_no_decay_params.append(parameter)
		else:
			adamw_decay_params.append(parameter)
	return OptimizerParameterSplit(
		muon_params=muon_params,
		adamw_decay_params=adamw_decay_params,
		adamw_no_decay_params=adamw_no_decay_params,
	)


def count_parameter_elements(parameters: list[torch.nn.Parameter]) -> int:
	return sum(parameter.numel() for parameter in parameters)


def build_optimizer(
	model: MykLLM,
	args: argparse.Namespace,
	parameter_split: OptimizerParameterSplit | None = None,
) -> torch.optim.Optimizer:
	parameter_split = parameter_split or split_optimizer_parameters(model, args)

	adamw_optimizer = torch.optim.AdamW(
		[
			{"params": parameter_split.adamw_decay_params, "weight_decay": args.weight_decay},
			{"params": parameter_split.adamw_no_decay_params, "weight_decay": 0.0},
		],
		lr=args.lr,
		betas=(0.9, 0.95),
	)

	if args.optimizer != "muon":
		return adamw_optimizer

	muon_cls = getattr(torch.optim, "Muon", None)
		
	if muon_cls is None:
		raise RuntimeError(
			"--optimizer muon requires torch.optim.Muon, but the installed PyTorch does not provide it"
		)

	if not parameter_split.muon_params:
		return adamw_optimizer

	muon_adjust_lr_fn = None if args.muon_adjust_lr_fn == "none" else args.muon_adjust_lr_fn
	muon_optimizer = muon_cls(
		parameter_split.muon_params,
		lr=args.muon_lr if args.muon_lr is not None else args.lr,
		weight_decay=args.muon_weight_decay if args.muon_weight_decay is not None else args.weight_decay,
		momentum=args.muon_momentum,
		nesterov=args.muon_nesterov,
		ns_steps=args.muon_ns_steps,
		adjust_lr_fn=muon_adjust_lr_fn,
	)
	return CombinedOptimizer(muon_optimizer, adamw_optimizer)


def build_scheduler(
	optimizer: torch.optim.Optimizer,
	total_steps: int,
	warmup_ratio: float,
	min_lr_ratio: float,
	schedule: str = "cosine",
	decay_ratio: float = 0.2,
) -> torch.optim.lr_scheduler.LambdaLR:
	warmup_steps = min(total_steps, max(0, int(total_steps * warmup_ratio)))

	def cosine_decay(progress: float) -> float:
		# progress in [0, 1]: 1.0 -> min_lr_ratio
		cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
		return min_lr_ratio + ((1.0 - min_lr_ratio) * cosine)

	if schedule == "cosine":

		def lr_lambda(step_index: int) -> float:
			current_step = step_index + 1
			if warmup_steps > 0 and current_step <= warmup_steps:
				return current_step / warmup_steps
			decay_steps = max(1, total_steps - warmup_steps)
			return cosine_decay((current_step - warmup_steps) / decay_steps)

	elif schedule == "wsd":
		# Warmup -> Stable (constant peak) -> Decay over the last `decay_ratio` of the run.
		# decay_ratio=0 stays at peak forever (train open-ended, anneal later from a
		# checkpoint); warmup_ratio=0 + decay_ratio=1 is a pure decay-anneal run.
		decay_steps = min(total_steps - warmup_steps, max(0, int(total_steps * decay_ratio)))
		decay_start = total_steps - decay_steps

		def lr_lambda(step_index: int) -> float:
			current_step = step_index + 1
			if warmup_steps > 0 and current_step <= warmup_steps:
				return current_step / warmup_steps
			if current_step <= decay_start or decay_steps == 0:
				return 1.0
			return cosine_decay((current_step - decay_start) / decay_steps)

	else:
		raise ValueError(f"unknown lr schedule '{schedule}'")

	return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def sample_text(
	model: MykLLM,
	tokenizer: PreTrainedTokenizerBase,
	prompt: str,
	max_new_tokens: int,
	temperature: float,
	top_k: int | None,
	mode: str,
) -> str:
	if mode == "chat":
		templated = tokenizer.apply_chat_template(
			[{"role": "user", "content": prompt}],
			tokenize=True,
			add_generation_prompt=True,
			return_tensors="pt",
			return_dict=True,
		)
		prompt_ids = cast(torch.Tensor, templated["input_ids"]).to(model.device)
	else:
		batch = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", verbose=False)
		prompt_ids = cast(torch.Tensor, batch["input_ids"]).to(model.device)
	with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
		output_ids = model.generate(
			prompt_ids,
			max_new_tokens=max_new_tokens,
			temperature=temperature,
			top_k=top_k,
		)
	return cast(str, tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False))


@torch.no_grad()
def evaluate(
	model: MykLLM,
	eval_sets: dict[str, list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]],
	device: torch.device,
	loss_chunk_size: int,
	document_masking: bool = False,
) -> dict[str, float]:
	"""Mean held-out loss + perplexity per language (and overall)."""
	model.eval()
	results: dict[str, float] = {}
	per_lang_losses: list[float] = []
	for lang, batches in eval_sets.items():
		total = 0.0
		for input_ids, targets, seqlens in batches:
			input_ids = input_ids.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)
			with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
				_, loss, _ = model(
					input_ids,
					targets=targets,
					seqlens=seqlens if document_masking else None,
					loss_chunk_size=loss_chunk_size,
				)
			total += float(cast(torch.Tensor, loss).item())
		avg = total / max(1, len(batches))
		results[f"val_loss_{lang}"] = avg
		results[f"val_ppl_{lang}"] = math.exp(avg)
		per_lang_losses.append(avg)
	model.train()
	if per_lang_losses:
		mean = sum(per_lang_losses) / len(per_lang_losses)
		results["val_loss"] = mean
		results["val_ppl"] = math.exp(mean)
	return results


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--tokenized-data", type=str, default="")
	parser.add_argument("--mode", type=str, choices=["pretrain", "chat"], default="pretrain")
	parser.add_argument("--chat-val-data", type=str, default="", help="chat mode: tokenized held-out prefix for periodic SFT loss (uses --eval-interval)")
	parser.add_argument("--chat-document-masking", dest="chat_document_masking", action="store_true", default=True, help="chat mode: block-diagonal attention over packed conversations (default on)")
	parser.add_argument("--chat-no-document-masking", dest="chat_document_masking", action="store_false")
	parser.add_argument("--stream", action="store_true", help="stream + tokenize the corpus on the fly instead of a tokenized memmap")
	parser.add_argument("--stream-langs", type=str, default="en,zh")
	parser.add_argument("--stream-weights", type=str, default="0.5,0.5", help="per-language token weights, comma separated")
	parser.add_argument("--stream-shuffle-buffer", type=int, default=10000)
	parser.add_argument("--stream-num-workers", type=int, default=4, help="keep <= the smallest source's shard count")
	parser.add_argument("--stream-seed", type=int, default=0)
	parser.add_argument("--stream-eos-token", type=str, default="<|endoftext|>", help="document separator token")
	parser.add_argument("--stream-mask-seams", action="store_true", help="mask cross-document next-token targets (pairs with document-masked attention)")
	parser.add_argument("--document-masking", action="store_true", help="block-diagonal (document-masked) attention over packed windows so tokens never attend across EOS seams; implies --stream-mask-seams")
	parser.add_argument("--stream-no-doc-filter", action="store_true", help="disable degenerate/wrong-language document filtering on the stream and eval sets")
	parser.add_argument("--stream-max-doc-tokens", type=int, default=8192, help="truncate any single streamed document to this many tokens so one doc can't dominate many packed windows; 0 disables the cap (lift for long-context runs)")
	parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH)
	parser.add_argument("--seq-len", type=int, default=1024)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--grad-accum-steps", type=int, default=1)
	parser.add_argument("--steps", type=int, default=1000)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--optimizer", type=str, choices=["adamw", "muon"], default="muon")
	parser.add_argument("--lr-schedule", type=str, choices=["cosine", "wsd"], default="cosine")
	parser.add_argument("--warmup-ratio", type=float, default=0.01)
	parser.add_argument("--min-lr-ratio", type=float, default=0.1)
	parser.add_argument("--decay-ratio", type=float, default=0.2, help="wsd only: fraction of steps spent decaying at the end (0 = stay at peak)")
	parser.add_argument("--weight-decay", type=float, default=0.1)
	parser.add_argument("--muon-lr", type=float, default=None)
	parser.add_argument("--muon-weight-decay", type=float, default=None)
	parser.add_argument("--muon-momentum", type=float, default=0.95)
	parser.add_argument("--muon-ns-steps", type=int, default=5)
	parser.add_argument(
		"--muon-adjust-lr-fn",
		type=str,
		choices=["original", "match_rms_adamw", "none"],
		default="match_rms_adamw",
	)
	parser.add_argument("--muon-nesterov", dest="muon_nesterov", action="store_true", default=True)
	parser.add_argument("--muon-no-nesterov", dest="muon_nesterov", action="store_false")
	parser.add_argument("--loss-chunk-size", type=int, default=2048)
	parser.add_argument("--log-interval", type=int, default=20)
	parser.add_argument("--tensorboard", action="store_true", help="also write scalar curves to <save-dir>/tb")
	parser.add_argument("--eval-interval", type=int, default=0, help="held-out val loss every N steps (0 disables; streaming only)")
	parser.add_argument("--eval-batches", type=int, default=8, help="batches per language in the fixed eval set")
	parser.add_argument("--eval-seed", type=int, default=999983)
	parser.add_argument("--sample-interval", type=int, default=200)
	parser.add_argument("--sample-prompt", type=str, default="Hello from MykLLM")
	parser.add_argument("--sample-tokens", type=int, default=64)
	parser.add_argument("--temperature", type=float, default=0.8)
	parser.add_argument("--top-k", type=int, default=50)
	parser.add_argument("--num-layers", type=int, default=33)
	parser.add_argument("--num-attention-heads", type=int, default=8)
	parser.add_argument("--num-kv-heads", type=int, default=2)
	parser.add_argument("--head-dim", type=int, default=128)
	parser.add_argument(
		"--full-attn-period",
		type=int,
		default=4,
		help="full-attention period P: layer i is full attention when i %% P == 0, else sliding-window (P=4 is the default schedule)",
	)
	parser.add_argument("--sliding-window-size", type=int, default=512)
	parser.add_argument("--ffn-multiplier", type=float, default=4.0)
	parser.add_argument("--ffn-multiple-of", type=int, default=256)
	parser.add_argument("--dropout", type=float, default=0.0)
	parser.add_argument("--tie-embeddings", dest="tie_embeddings", action="store_true", default=True)
	parser.add_argument("--untie-embeddings", dest="tie_embeddings", action="store_false")
	parser.add_argument("--gradient-checkpointing", action="store_true")
	parser.add_argument("--grad-clip", type=float, default=1.0)
	parser.add_argument("--save-dir", type=str, default="checkpoints")
	parser.add_argument("--save-interval", type=int, default=200)
	parser.add_argument("--keep-last-checkpoints", type=int, default=5)
	parser.add_argument("--progress-save-ratio", type=float, default=0.1)
	parser.add_argument("--reuse-pretrained-embeddings", action="store_true")
	parser.add_argument("--freeze-token-embedding", action="store_true")
	parser.add_argument("--pretrained-embedding-source", type=str, default="")
	parser.add_argument("--pretrained-embedding-row-map", type=str, default="")
	parser.add_argument("--resume", type=str, default="", help="continue a run exactly (model + optimizer + schedule + step + data position)")
	parser.add_argument("--init-from", type=str, default="", help="warm-start a fresh run from a checkpoint's weights (fresh schedule, step 1, fresh data)")
	parser.add_argument("--init-from-optimizer", action="store_true", help="also carry optimizer momentum when using --init-from")
	parser.add_argument("--override-resume-seq-len", action="store_true")
	args = parser.parse_args()
	if args.grad_accum_steps <= 0:
		raise ValueError("--grad-accum-steps must be positive")
	if not 0.0 <= args.warmup_ratio < 1.0:
		raise ValueError("--warmup-ratio must be in [0, 1)")
	if not 0.0 < args.min_lr_ratio <= 1.0:
		raise ValueError("--min-lr-ratio must be in (0, 1]")
	if not 0.0 <= args.decay_ratio <= 1.0:
		raise ValueError("--decay-ratio must be in [0, 1]")
	if args.resume and args.init_from:
		raise ValueError("--resume and --init-from are mutually exclusive")
	if args.init_from_optimizer and not args.init_from:
		raise ValueError("--init-from-optimizer requires --init-from")
	if args.loss_chunk_size <= 0:
		raise ValueError("--loss-chunk-size must be positive")
	if args.keep_last_checkpoints < 0:
		raise ValueError("--keep-last-checkpoints must be non-negative")
	if not 0.0 <= args.progress_save_ratio <= 1.0:
		raise ValueError("--progress-save-ratio must be in [0, 1]")
	if args.pretrained_embedding_source and not args.reuse_pretrained_embeddings:
		raise ValueError("--pretrained-embedding-source requires --reuse-pretrained-embeddings")
	if args.pretrained_embedding_row_map and not args.reuse_pretrained_embeddings:
		raise ValueError("--pretrained-embedding-row-map requires --reuse-pretrained-embeddings")
	if args.stream:
		if args.mode == "chat":
			raise ValueError("--stream is pretrain-only; chat mode uses tokenized data with loss masks")
		stream_langs = [lang.strip() for lang in args.stream_langs.split(",") if lang.strip()]
		stream_weights = [float(weight) for weight in args.stream_weights.split(",")]
		if len(stream_langs) != len(stream_weights):
			raise ValueError("--stream-langs and --stream-weights must have the same length")
	elif not args.tokenized_data:
		raise ValueError("provide --tokenized-data, or use --stream")

	if not torch.cuda.is_available():
		raise SystemExit("training is CUDA-only; no GPU is available")

	device = torch.device("cuda")
	torch.set_float32_matmul_precision("high")
	resume_model_config = resolve_resume_model_config(args.resume or args.init_from)
	if resume_model_config is not None and args.override_resume_seq_len:
		resume_model_config.max_seq_len = args.seq_len

	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
	tokenizer.model_max_length = int(1e30)
	if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
		tokenizer.pad_token = tokenizer.eos_token
	if args.mode == "chat" and tokenizer.chat_template is None:
		raise ValueError("chat mode requires a tokenizer with a chat template")

	dataset: TokenDataset | None = None
	sampling_weights: torch.Tensor | None = None
	stream_loader = None
	doc_filter = None
	if args.stream:
		from streaming import DocFilter, build_stream_dataloader

		# One filter shared by the training stream and the eval sets (below).
		doc_filter = DocFilter(enabled=not args.stream_no_doc_filter, max_doc_tokens=args.stream_max_doc_tokens)
		eos_id = int(tokenizer.convert_tokens_to_ids(args.stream_eos_token))
		if eos_id < 0 or eos_id == tokenizer.unk_token_id:
			raise ValueError(f"--stream-eos-token {args.stream_eos_token!r} is not in the tokenizer vocab")
		stream_loader = build_stream_dataloader(
			stream_langs,
			stream_weights,
			tokenizer_path=str(Path(args.tokenizer) / "tokenizer.json"),
			seq_len=args.seq_len,
			batch_size=args.batch_size,
			eos_id=eos_id,
			shuffle_buffer=args.stream_shuffle_buffer,
			num_workers=args.stream_num_workers,
			base_seed=args.stream_seed,
			mask_seams=args.stream_mask_seams or args.document_masking,
			doc_filter=doc_filter,
		)
	else:
		dataset = load_tokenized_dataset(args.tokenized_data)
		if args.mode == "chat":
			if dataset.loss_mask is None:
				raise ValueError("chat mode requires tokenized data with a loss mask; run myk-llm/tokenization.py --mode chat")
			# Chat uses packed cross-conversation windows (get_chat_batch), not within-span
			# sampling, so short conversations are used rather than dropped.
		else:
			sampling_weights = build_sampling_weights(dataset.spans, args.seq_len)

	pretrained_embedding_weight: torch.Tensor | None = None
	if args.reuse_pretrained_embeddings and resume_model_config is None:
		embedding_source = args.pretrained_embedding_source or args.tokenizer
		pretrained_embedding_weight = load_pretrained_token_embedding(embedding_source)
		if pretrained_embedding_weight.ndim != 2:
			raise ValueError(
				f"pretrained embedding tensor must be rank 2, got shape {tuple(pretrained_embedding_weight.shape)}"
			)
		if args.pretrained_embedding_row_map:
			pretrained_embedding_weight = gather_pretrained_embedding_rows(
				pretrained_embedding_weight, args.pretrained_embedding_row_map
			)

	if resume_model_config is not None:
		model_config = resume_model_config
	else:
		model_vocab_size = int(pretrained_embedding_weight.shape[0]) if pretrained_embedding_weight is not None else len(tokenizer)
		model_config = build_config(args, vocab_size=model_vocab_size)

	if len(tokenizer) > model_config.vocab_size:
		raise ValueError(
			f"tokenizer vocab size {len(tokenizer)} exceeds model vocab size {model_config.vocab_size}"
		)

	model = MykLLM(model_config).to(device=device)
	if pretrained_embedding_weight is not None:
		initialize_token_embedding_from_pretrained(model, pretrained_embedding_weight)
	maybe_freeze_token_embedding(model, args.freeze_token_embedding)
	optimizer_split = split_optimizer_parameters(model, args)
	optimizer = build_optimizer(model, args, parameter_split=optimizer_split)
	scheduler = build_scheduler(optimizer, args.steps, args.warmup_ratio, args.min_lr_ratio, schedule=args.lr_schedule, decay_ratio=args.decay_ratio)
	progress_checkpoint_steps = build_progress_checkpoint_steps(args.steps, args.progress_save_ratio)
	save_dir = Path(args.save_dir)
	start_step = 1
	if args.resume:
		start_step = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler, device=device, data_loader=stream_loader) + 1
	elif args.init_from:
		# Warm-start weights only (and optionally optimizer momentum); fresh schedule, step 1, fresh data stream.
		load_checkpoint(args.init_from, model, optimizer=(optimizer if args.init_from_optimizer else None), scheduler=None, device=device)
		print(f"init-from {args.init_from}: weights loaded, optimizer={'carried' if args.init_from_optimizer else 'fresh'}, fresh schedule from step 1")

	effective_tokens_per_step = args.batch_size * args.seq_len * args.grad_accum_steps
	muon_tensor_count = len(optimizer_split.muon_params)
	muon_element_count = count_parameter_elements(optimizer_split.muon_params)
	adamw_tensor_count = len(optimizer_split.adamw_params)
	adamw_element_count = count_parameter_elements(optimizer_split.adamw_params)

	data_source = f"stream[{args.stream_langs}]" if args.stream else f"{args.tokenized_data} ({dataset.num_tokens} tokens)"
	print(
		f"device={device} mode={args.mode} vocab={model_config.vocab_size} seq_len={model_config.max_seq_len} "
		f"params={model.num_parameters() / 1e6:.2f}M tokenizer={args.tokenizer} optimizer={args.optimizer} start_step={start_step} "
		f"grad_accum={args.grad_accum_steps} effective_tokens={effective_tokens_per_step} data={data_source}"
	)
	print(
		f"optimizer_split muon_tensors={muon_tensor_count} muon_params={muon_element_count} "
		f"adamw_tensors={adamw_tensor_count} adamw_params={adamw_element_count}"
	)

	from metrics import MetricsLogger

	metrics_logger = MetricsLogger(save_dir / "metrics.jsonl", (save_dir / "tb") if args.tensorboard else None)
	eval_sets: dict[str, list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]] = {}
	if args.stream and args.eval_interval > 0:
		from streaming import build_eval_batches

		tokenizer_json = str(Path(args.tokenizer) / "tokenizer.json")
		for lang in stream_langs:
			eval_sets[lang] = build_eval_batches(
				lang,
				tokenizer_path=tokenizer_json,
				seq_len=args.seq_len,
				batch_size=args.batch_size,
				eos_id=eos_id,
				num_batches=args.eval_batches,
				seed=args.eval_seed,
				doc_filter=doc_filter,
				mask_seams=args.stream_mask_seams or args.document_masking,
			)
		print(f"built eval sets: {{ {', '.join(f'{k}:{len(v)}x{args.batch_size}' for k, v in eval_sets.items())} }}")

	if args.mode == "chat" and args.chat_val_data and args.eval_interval > 0:
		val_dataset = load_tokenized_dataset(args.chat_val_data)
		if val_dataset.loss_mask is None:
			raise ValueError("--chat-val-data must be tokenized with a loss mask (tokenization.py --mode chat)")
		eval_sets["sft"] = build_chat_eval_batches(
			val_dataset,
			seq_len=args.seq_len,
			batch_size=args.batch_size,
			num_batches=args.eval_batches,
			seed=args.eval_seed,
			document_masking=args.chat_document_masking,
		)
		print(f"built chat eval set: sft:{len(eval_sets['sft'])}x{args.batch_size}")

	stream_iter = iter(stream_loader) if stream_loader is not None else None
	loss_ema: float | None = None

	model.train()
	for step in tqdm(range(start_step, args.steps + 1)):
		step_start = time.perf_counter()
		optimizer.zero_grad(set_to_none=True)
		loss_value = 0.0
		for _ in range(args.grad_accum_steps):
			seqlens = None
			if stream_iter is not None:
				batch = next(stream_iter)
				input_ids = batch["input_ids"].to(device, non_blocking=True)
				targets = batch["targets"].to(device, non_blocking=True)
				if args.document_masking:
					seqlens = batch["seqlens"]
			elif args.mode == "chat":
				input_ids, targets, seqlens = get_chat_batch(
					cast(TokenDataset, dataset), args.seq_len, args.batch_size, device,
					document_masking=args.chat_document_masking,
				)
			else:
				input_ids, targets = get_batch(cast(TokenDataset, dataset), cast(torch.Tensor, sampling_weights), args.seq_len, args.batch_size, device)

			with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
				_, loss, _ = model(input_ids, targets=targets, seqlens=seqlens, loss_chunk_size=args.loss_chunk_size)

			loss_value += float(cast(torch.Tensor, loss).detach().item())
			cast(torch.Tensor, loss).div(args.grad_accum_steps).backward()

		grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
		optimizer.step()
		scheduler.step()
		torch.cuda.synchronize()
		step_loss = loss_value / args.grad_accum_steps
		loss_ema = step_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * step_loss
		step_time = time.perf_counter() - step_start
		tokens_per_second = effective_tokens_per_step / step_time

		if step % args.log_interval == 0:
			lr = scheduler.get_last_lr()[0]
			metrics_logger.log(
				step,
				{
					"loss": step_loss,
					"loss_ema": loss_ema,
					"lr": lr,
					"grad_norm": grad_norm,
					"tok_per_s": tokens_per_second,
					"step_time_s": step_time,
					"tokens": step * effective_tokens_per_step,
				},
			)
			print(
				f"step={step:5d} loss={step_loss:.4f} ema={loss_ema:.4f} lr={lr:.6e} "
				f"grad_norm={grad_norm:.2f} tok/s={tokens_per_second:,.0f} tokens={step * effective_tokens_per_step}"
			)

		if eval_sets and args.eval_interval > 0 and step % args.eval_interval == 0:
			eval_document_masking = args.document_masking or (args.mode == "chat" and args.chat_document_masking)
			val_metrics = evaluate(model, eval_sets, device, args.loss_chunk_size, document_masking=eval_document_masking)
			metrics_logger.log(step, val_metrics)
			summary = " ".join(f"{key}={value:.4f}" for key, value in val_metrics.items() if key.startswith("val_loss"))
			ppl = " ".join(f"{key}={value:.1f}" for key, value in val_metrics.items() if key.startswith("val_ppl"))
			print(f"step={step:5d} [eval] {summary} | {ppl}")

		if args.sample_interval > 0 and step % args.sample_interval == 0:
			model.eval()
			sample = sample_text(
				model,
				tokenizer,
				prompt=args.sample_prompt,
				max_new_tokens=args.sample_tokens,
				temperature=args.temperature,
				top_k=args.top_k,
				mode=args.mode,
			)
			print("\n--- sample ---")
			print(sample)
			model.train()

		if args.save_interval > 0 and step % args.save_interval == 0:
			checkpoint_path = save_dir / f"step_{step:07d}.pt"
			save_checkpoint(checkpoint_path, model, optimizer, scheduler, args, step, data_loader=stream_loader)
			prune_old_step_checkpoints(save_dir, args.keep_last_checkpoints)
			print(f"saved checkpoint to {checkpoint_path}")

		if step in progress_checkpoint_steps:
			progress_percent = progress_checkpoint_steps[step]
			checkpoint_path = save_dir / f"progress_{progress_percent:03d}pct_step_{step:07d}.pt"
			save_checkpoint(checkpoint_path, model, optimizer, scheduler, args, step, data_loader=stream_loader)
			print(f"saved progress checkpoint to {checkpoint_path}")

	if args.save_interval != 0:
		final_checkpoint = save_dir / "last.pt"
		save_checkpoint(final_checkpoint, model, optimizer, scheduler, args, args.steps, data_loader=stream_loader)
		print(f"saved checkpoint to {final_checkpoint}")

	metrics_logger.close()


if __name__ == "__main__":
	main()
