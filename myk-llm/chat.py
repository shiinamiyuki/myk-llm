from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model import MykLLM, MykLLMConfig
from training import DEFAULT_TOKENIZER_PATH


EXIT_COMMANDS = {"exit", "quit", ":q"}


def load_checkpoint_payload(checkpoint_path: str) -> dict[str, object]:
    return cast(dict[str, object], torch.load(checkpoint_path, map_location="cpu"))


def build_model_from_checkpoint(checkpoint: dict[str, object], device: torch.device) -> MykLLM:
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint is missing a valid model_config")

    model = MykLLM(MykLLMConfig(**cast(dict[str, object], model_config))).to(device=device)
    model.load_state_dict(cast(dict[str, torch.Tensor], checkpoint["model_state_dict"]))
    model.eval()
    return model


def resolve_tokenizer_path(args: argparse.Namespace, checkpoint: dict[str, object]) -> str:
    if args.tokenizer:
        return args.tokenizer

    training_args = checkpoint.get("training_args")
    if isinstance(training_args, dict):
        tokenizer_path = training_args.get("tokenizer")
        if isinstance(tokenizer_path, str) and tokenizer_path:
            return tokenizer_path

    return DEFAULT_TOKENIZER_PATH


def build_prompt_ids(
    model: MykLLM,
    tokenizer: PreTrainedTokenizerBase,
    history: list[tuple[str, str]],
    user_text: str,
    raw: bool,
) -> torch.Tensor:
    """Token ids for the next turn: ChatML template (SFT models) or raw continuation."""
    if raw:
        batch = tokenizer(user_text, add_special_tokens=False, return_tensors="pt", verbose=False)
        prompt_ids = cast(torch.Tensor, batch["input_ids"])
    else:
        messages: list[dict[str, str]] = []
        for past_user_text, assistant_text in history:
            messages.append({"role": "user", "content": past_user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": user_text})
        encoding = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        prompt_ids = cast(torch.Tensor, encoding["input_ids"])
    return prompt_ids.to(model.device)[:, -model.config.max_seq_len :]


def generate_reply(
    model: MykLLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    repetition_penalty: float,
) -> str:

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )

    generated_ids = output_ids[0, prompt_ids.size(1) :].tolist()
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int) and eos_token_id in generated_ids:
        generated_ids = generated_ids[: generated_ids.index(eos_token_id)]

    return cast(str, tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()


def interactive_chat(
    model: MykLLM,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    repetition_penalty: float,
    raw: bool = False,
) -> None:
    history: list[tuple[str, str]] = []
    print("Type 'exit' to quit.")

    while True:
        try:
            user_text = input("user> ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            break

        prompt_ids = build_prompt_ids(model, tokenizer, history, user_text, raw)
        assistant_text = generate_reply(model, tokenizer, prompt_ids, max_tokens, temperature, top_k, repetition_penalty)
        print(f"assistant> {assistant_text}")
        history.append((user_text, assistant_text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="feed --prompt verbatim (plain continuation) instead of the User/Assistant chat template; "
        "use this for base/pretrained checkpoints",
    )
    args = parser.parse_args()

    if args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be positive")

    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive when provided")
    if not torch.cuda.is_available():
        raise SystemExit("chat is CUDA-only; no GPU is available")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint_payload(str(checkpoint_path))
    tokenizer_path = resolve_tokenizer_path(args, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model = build_model_from_checkpoint(checkpoint, device)

    if args.prompt:
        prompt_ids = build_prompt_ids(model, tokenizer, [], args.prompt, args.raw)
        print(generate_reply(
            model, tokenizer, prompt_ids, args.max_tokens, args.temperature, args.top_k, args.repetition_penalty
        ))
        return

    interactive_chat(model, tokenizer, args.max_tokens, args.temperature, args.top_k, args.repetition_penalty, raw=args.raw)


if __name__ == "__main__":
    main()