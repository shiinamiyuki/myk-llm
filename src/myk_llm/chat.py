"""Interactive chat / text completion with an exported model directory.

    python -m myk_llm.chat runs/200m-sft/checkpoints/step_0006000
    python -m myk_llm.chat runs/200m-pt/checkpoints/step_0016000 --raw --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .model import MykLLM, generate, load_model

EXIT_COMMANDS = {"exit", "quit", ":q"}


def build_prompt_ids(
    model: MykLLM, tokenizer: PreTrainedTokenizerBase, history: list[tuple[str, str]], user_text: str, raw: bool
) -> torch.Tensor:
    """Token ids for the next turn: the chat template (SFT models) or raw continuation."""
    if raw:
        ids = tokenizer(user_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    else:
        messages = []
        for past_user, past_assistant in history:
            messages += [{"role": "user", "content": past_user}, {"role": "assistant", "content": past_assistant}]
        messages.append({"role": "user", "content": user_text})
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
    return ids.to(model.device)


def generate_reply(
    model: MykLLM, tokenizer: PreTrainedTokenizerBase, prompt_ids: torch.Tensor, max_tokens: int, **sampling
) -> tuple[str, list[int], bool]:
    """(reply text, reply token ids, whether the model emitted the end-of-turn token)."""
    # Keep the most recent context that still leaves room for the reply.
    prompt_ids = prompt_ids[:, -(model.config.max_seq_len - max_tokens) :]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = generate(model, prompt_ids, max_tokens, eos_token_id=tokenizer.eos_token_id, **sampling)
    ids = output[0, prompt_ids.size(1) :].tolist()
    stopped = tokenizer.eos_token_id in ids
    if stopped:
        ids = ids[: ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(ids, skip_special_tokens=True).strip(), ids, stopped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="exported model directory (model.safetensors + model.yaml + tokenizer)")
    parser.add_argument("--prompt", default="", help="answer one prompt and exit")
    parser.add_argument("--raw", action="store_true", help="plain continuation, no chat template (base models)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    model = load_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling = dict(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty
    )
    if args.prompt:
        prompt_ids = build_prompt_ids(model, tokenizer, [], args.prompt, args.raw)
        print(generate_reply(model, tokenizer, prompt_ids, args.max_tokens, **sampling)[0])
        return

    history: list[tuple[str, str]] = []
    print("Type 'exit' to quit.")
    while True:
        try:
            user_text = input("user> ").strip()
        except EOFError:
            break
        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            break
        prompt_ids = build_prompt_ids(model, tokenizer, history, user_text, args.raw)
        reply, _, _ = generate_reply(model, tokenizer, prompt_ids, args.max_tokens, **sampling)
        print(f"assistant> {reply}")
        history.append((user_text, reply))


if __name__ == "__main__":
    main()
