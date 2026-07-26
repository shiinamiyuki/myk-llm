from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_TOKENIZER_PATH = "/home/miyuki/repos/myk-llm/_download_models/Qwen3.5-0.8B"


def load_json_data(path: Path) -> object:
	with path.open(encoding="utf-8") as handle:
		return json.load(handle)


def tokenized_dataset_paths(prefix: Path) -> tuple[Path, Path, Path, Path]:
	return (
		prefix.with_suffix(".tokens.bin"),
		prefix.with_suffix(".loss_mask.bin"),
		prefix.with_suffix(".spans.npy"),
		prefix.with_suffix(".meta.json"),
	)


def encode_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
	token_ids = cast(list[int], tokenizer.encode(text, add_special_tokens=False))
	if len(token_ids) < 2:
		raise ValueError("tokenized dataset must contain at least two tokens")
	return token_ids


# ShareGPT-style role aliases (Belle / some Chinese sets use from/value with these).
_ROLE_ALIASES = {
	"human": "user",
	"user": "user",
	"gpt": "assistant",
	"assistant": "assistant",
	"bot": "assistant",
	"system": "system",
}


def normalize_chat_messages(example: dict[str, object]) -> list[dict[str, str]]:
	messages = example.get("messages")
	if messages is None:
		messages = example.get("conversations")
	if not isinstance(messages, list) or not messages:
		raise ValueError("chat examples must contain a non-empty 'messages' or 'conversations' list")

	normalized: list[dict[str, str]] = []
	for message in messages:
		if not isinstance(message, dict):
			raise ValueError("each chat message must be a JSON object")
		# Accept both {'role','content'} and ShareGPT {'from','value'}.
		role = message.get("role", message.get("from"))
		content = message.get("content", message.get("value"))
		if not isinstance(role, str) or not isinstance(content, str):
			raise ValueError("each chat message must have string role/content (or from/value) fields")
		role = _ROLE_ALIASES.get(role.lower(), role)
		normalized.append({"role": role, "content": content})
	return normalized


def encode_chat_example(
	tokenizer: PreTrainedTokenizerBase,
	messages: list[dict[str, str]],
) -> tuple[list[int], list[int]]:
	"""Tokenize a full conversation and return per-token ids + assistant loss mask.

	Relies on the tokenizer's chat template wrapping assistant responses in
	`{% generation %}` tags, so `return_assistant_tokens_mask` marks exactly the
	tokens the model must learn to produce (assistant content + its terminating
	`<|im_end|>`); prompt/system/user tokens are masked to 0. This is robust to
	tokenizer seam effects, unlike diffing re-tokenized message prefixes.
	"""
	encoding = cast(
		dict[str, list],
		tokenizer.apply_chat_template(
			messages,
			tokenize=True,
			add_generation_prompt=False,
			return_dict=True,
			return_assistant_tokens_mask=True,
		),
	)
	token_ids = [int(token_id) for token_id in encoding["input_ids"]]
	assistant_mask = encoding.get("assistant_masks")
	if not assistant_mask or len(assistant_mask) != len(token_ids):
		raise ValueError(
			"tokenizer chat template did not yield an assistant token mask; the template must "
			"wrap assistant content in {% generation %} tags (see assets/chatml_no_think.jinja)"
		)
	loss_mask = [int(value) for value in assistant_mask]
	return token_ids, loss_mask


def iter_chat_examples(path: Path) -> list[dict[str, object]]:
	if path.suffix == ".jsonl":
		records: list[dict[str, object]] = []
		with path.open(encoding="utf-8") as handle:
			for line in handle:
				line = line.strip()
				if not line:
					continue
				record = json.loads(line)
				if not isinstance(record, dict):
					raise ValueError("each JSONL record must be an object")
				records.append(record)
		return records

	data = load_json_data(path)
	if not isinstance(data, list):
		raise ValueError("JSON chat data must be a list of conversation objects")
	for example in data:
		if not isinstance(example, dict):
			raise ValueError("each JSON conversation must be an object")
	return cast(list[dict[str, object]], data)


def write_tokenized_dataset(
	prefix: Path,
	token_ids: np.ndarray,
	spans: np.ndarray,
	loss_mask: np.ndarray | None,
	metadata: dict[str, object],
) -> None:
	prefix.parent.mkdir(parents=True, exist_ok=True)
	tokens_path, loss_mask_path, spans_path, meta_path = tokenized_dataset_paths(prefix)
	token_ids.astype(np.uint32, copy=False).tofile(tokens_path)
	np.save(spans_path, spans.astype(np.int64, copy=False))
	if loss_mask is None:
		if loss_mask_path.exists():
			loss_mask_path.unlink()
	else:
		loss_mask.astype(np.uint8, copy=False).tofile(loss_mask_path)

	meta_path.write_text(
		json.dumps(
			{
				**metadata,
				"num_tokens": int(token_ids.shape[0]),
				"num_spans": int(spans.shape[0]),
				"token_dtype": "uint32",
				"span_dtype": "int64",
				"has_loss_mask": loss_mask is not None,
			},
			indent=2,
			sort_keys=True,
		),
		encoding="utf-8",
	)


def tokenize_pretrain_data(tokenizer: PreTrainedTokenizerBase, path: Path) -> tuple[np.ndarray, np.ndarray, None]:
	text = path.read_text(encoding="utf-8")
	token_ids = np.asarray(encode_text(tokenizer, text), dtype=np.uint32)
	spans = np.asarray([[0, token_ids.shape[0]]], dtype=np.int64)
	return token_ids, spans, None


def tokenize_chat_data(
	tokenizer: PreTrainedTokenizerBase,
	path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Tokenize conversations and pack them contiguously for windowed SFT sampling.

	Conversations are concatenated into one token stream; `spans` records the
	[start, end) offset of each conversation so the trainer can (a) mask loss to
	assistant tokens and (b) build block-diagonal (document-masked) attention over
	packed windows, keeping conversations from attending across each other. The
	chat template already terminates each assistant turn with `<|im_end|>`, so no
	extra EOS is appended.
	"""
	packed_token_ids: list[int] = []
	packed_loss_mask: list[int] = []
	spans: list[tuple[int, int]] = []
	current_offset = 0
	skipped_no_assistant = 0

	for example in tqdm(iter_chat_examples(path), desc="tokenizing chat examples"):
		conversation_token_ids, conversation_loss_mask = encode_chat_example(
			tokenizer,
			normalize_chat_messages(example),
		)
		# Drop conversations with nothing to train on (no assistant turn) or too short.
		if len(conversation_token_ids) < 2 or not any(conversation_loss_mask):
			skipped_no_assistant += 1
			continue
		packed_token_ids.extend(conversation_token_ids)
		packed_loss_mask.extend(conversation_loss_mask)
		spans.append((current_offset, current_offset + len(conversation_token_ids)))
		current_offset += len(conversation_token_ids)

	if len(packed_token_ids) < 2:
		raise ValueError("chat dataset must contain at least two tokens after templating")
	if skipped_no_assistant:
		print(f"skipped {skipped_no_assistant} conversation(s) with no trainable assistant tokens")
	return (
		np.asarray(packed_token_ids, dtype=np.uint32),
		np.asarray(spans, dtype=np.int64),
		np.asarray(packed_loss_mask, dtype=np.uint8),
	)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", type=str, required=True)
	parser.add_argument("--output", type=str, required=True)
	parser.add_argument("--mode", type=str, choices=["pretrain", "chat"], default="pretrain")
	parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH)
	args = parser.parse_args()

	data_path = Path(args.data)
	if not data_path.exists():
		raise FileNotFoundError(f"data file not found: {data_path}")

	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
	tokenizer.model_max_length = int(1e30)
	if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
		tokenizer.pad_token = tokenizer.eos_token
	if args.mode == "chat" and tokenizer.chat_template is None:
		raise ValueError("chat mode requires a tokenizer with a chat template")

	if args.mode == "pretrain":
		token_ids, spans, loss_mask = tokenize_pretrain_data(tokenizer, data_path)
	else:
		token_ids, spans, loss_mask = tokenize_chat_data(tokenizer, data_path)

	output_prefix = Path(args.output)
	write_tokenized_dataset(
		output_prefix,
		token_ids,
		spans,
		loss_mask,
		{
			"mode": args.mode,
			"source": str(data_path),
			"tokenizer": args.tokenizer,
		},
	)
	tokens_path, loss_mask_path, spans_path, meta_path = tokenized_dataset_paths(output_prefix)
	print(f"wrote {int(token_ids.shape[0])} tokens across {int(spans.shape[0])} span(s)")
	print(f"tokens={tokens_path}")
	print(f"spans={spans_path}")
	if loss_mask is not None:
		print(f"loss_mask={loss_mask_path}")
	print(f"metadata={meta_path}")


if __name__ == "__main__":
	main()