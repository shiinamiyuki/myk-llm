"""Generation-based sanity eval for SFT (chat) checkpoints.

Runs a fixed bilingual prompt set through the ChatML template under several decoding
configs and reports cheap automatic signals, plus every generation for reading:

  - stop:     the reply ended with the end-of-turn token (<|im_end|>) before max tokens
  - rep4:     fraction of repeated 4-grams in the reply tokens (degenerate looping)
  - lang_ok:  reply is written in the prompt's language (CJK share of letters)
  - leak:     reply hallucinates a new turn (e.g. "<|im_start|>" / "user\\n")
  - hit:      for prompts with a known answer, the reply contains it

Usage:
  python myk-llm/eval_chat.py --checkpoint checkpoints/200M-v5-sft/last.pt --output chat_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from chat import build_model_from_checkpoint, build_prompt_ids, load_checkpoint_payload, resolve_tokenizer_path
from transformers import AutoTokenizer


@dataclass
class ChatCase:
    turns: list[str]  # user turns; earlier turns are answered by the model itself
    lang: str  # "en" | "zh"
    answers: list[str] = field(default_factory=list)  # any of these substrings counts as a hit


CASES: list[ChatCase] = [
    # Short factual questions with a checkable answer.
    ChatCase(["What is the capital of France?"], "en", ["Paris"]),
    ChatCase(["What is 7 plus 5?"], "en", ["12", "twelve"]),
    ChatCase(["What color is the sky on a clear day?"], "en", ["blue"]),
    ChatCase(["How many days are there in a week?"], "en", ["7", "seven"]),
    ChatCase(["Who wrote 'Romeo and Juliet'?"], "en", ["Shakespeare"]),
    ChatCase(["What is the chemical symbol for water?"], "en", ["H2O", "H₂O"]),
    ChatCase(["Translate to English: 我喜欢吃苹果。"], "en", ["apple"]),
    ChatCase(["中国的首都是哪个城市？"], "zh", ["北京"]),
    ChatCase(["3加4等于多少？"], "zh", ["7", "七"]),
    ChatCase(["一年有几个月？"], "zh", ["12", "十二"]),
    ChatCase(["水的化学式是什么？"], "zh", ["H2O", "H₂O"]),
    ChatCase(["《红楼梦》的作者是谁？"], "zh", ["曹雪芹"]),
    ChatCase(["把这句话翻译成中文：I love reading books."], "zh", ["书"]),
    ChatCase(["太阳从哪个方向升起？"], "zh", ["东"]),
    # Open-ended instructions (read these by hand).
    ChatCase(["Hi! Can you introduce yourself in two sentences?"], "en"),
    ChatCase(["Explain what gradient descent is to a high school student."], "en"),
    ChatCase(["Give me three tips for sleeping better. Use a numbered list."], "en"),
    ChatCase(["Write a short poem about the ocean."], "en"),
    ChatCase(["Write a Python function that returns the sum of a list of numbers."], "en"),
    ChatCase(["你好，请用一句话介绍一下你自己。"], "zh"),
    ChatCase(["用中文解释一下什么是梯度下降。"], "zh"),
    ChatCase(["给我三个提高学习效率的建议，用列表形式回答。"], "zh"),
    ChatCase(["写一首关于秋天的五言绝句。"], "zh"),
    ChatCase(["为什么夏天比冬天热？请简单解释。"], "zh"),
    # Multi-turn: must use information from an earlier turn.
    ChatCase(["My name is Alice and I love cats.", "What is my name?"], "en", ["Alice"]),
    ChatCase(["我叫小明，今年十岁。", "我叫什么名字？"], "zh", ["小明"]),
]

DECODING = {
    "greedy": dict(temperature=0.0, top_k=None, repetition_penalty=1.0),
    "sample": dict(temperature=0.7, top_k=50, repetition_penalty=1.0),
    "chat_py_default": dict(temperature=0.8, top_k=50, repetition_penalty=1.3),
}

_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


def reply_language_ok(text: str, lang: str) -> bool:
    cjk, latin = len(_CJK.findall(text)), len(_LATIN.findall(text))
    if cjk + latin == 0:
        return True  # e.g. a bare number
    share = cjk / (cjk + latin / 4)  # ~4 Latin letters per word vs 1 char per CJK word
    return share >= 0.5 if lang == "zh" else share < 0.5


def contains_answer(text: str, answer: str) -> bool:
    # Numbers must match whole ("7" should not hit "170").
    if answer.isdigit():
        return re.search(rf"(?<!\d){answer}(?!\d)", text) is not None
    return answer.lower() in text.lower()


def repeated_ngram_fraction(tokens: list[int], n: int = 4) -> float:
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


@torch.no_grad()
def generate(model, tokenizer, history, user_text, max_tokens, seed, **decoding) -> tuple[str, list[int], bool]:
    torch.manual_seed(seed)
    prompt_ids = build_prompt_ids(model, tokenizer, history, user_text, raw=False)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.generate(prompt_ids, max_new_tokens=max_tokens, eos_token_id=tokenizer.eos_token_id, **decoding)
    new_ids = output[0, prompt_ids.size(1) :].tolist()
    stopped = tokenizer.eos_token_id in new_ids
    if stopped:
        new_ids = new_ids[: new_ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(new_ids, skip_special_tokens=False), new_ids, stopped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--decoding", type=str, default=",".join(DECODING))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="chat_eval.jsonl")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    checkpoint = load_checkpoint_payload(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_path(args, checkpoint))
    model = build_model_from_checkpoint(checkpoint, torch.device("cuda"))
    del checkpoint

    records = []
    for decoding_name in [name.strip() for name in args.decoding.split(",")]:
        decoding = DECODING[decoding_name]
        for case_index, case in enumerate(CASES):
            history: list[tuple[str, str]] = []
            for user_text in case.turns:
                text, ids, stopped = generate(
                    model, tokenizer, history, user_text, args.max_tokens, args.seed + case_index, **decoding
                )
                history.append((user_text, text))
            records.append(
                {
                    "checkpoint": args.checkpoint,
                    "decoding": decoding_name,
                    "case": case_index,
                    "turns": case.turns,
                    "lang": case.lang,
                    "reply": text,  # reply to the final turn
                    "stop": stopped,
                    "rep4": repeated_ngram_fraction(ids),
                    "lang_ok": reply_language_ok(text, case.lang),
                    "leak": "<|im_start|>" in text or bool(re.search(r"(^|\n)(user|assistant)\n", text)),
                    "hit": (any(contains_answer(text, a) for a in case.answers) if case.answers else None),
                    "num_tokens": len(ids),
                }
            )

    with Path(args.output).open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{'decoding':<16} {'stop':>5} {'rep4':>5} {'lang':>5} {'leak':>5} {'hit':>5} {'len':>5}")
    for decoding_name in dict.fromkeys(r["decoding"] for r in records):
        rows = [r for r in records if r["decoding"] == decoding_name]
        graded = [r for r in rows if r["hit"] is not None]
        mean = lambda key, rs=rows: sum(float(r[key]) for r in rs) / len(rs)  # noqa: E731
        print(
            f"{decoding_name:<16} {mean('stop'):5.2f} {mean('rep4'):5.2f} {mean('lang_ok'):5.2f} "
            f"{mean('leak'):5.2f} {mean('hit', graded):5.2f} {mean('num_tokens'):5.0f}"
        )


if __name__ == "__main__":
    main()
