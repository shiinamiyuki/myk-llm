"""Zero-shot benchmark harness for MykLLM checkpoints and HuggingFace reference models.

Every task is scored by log-likelihood (no generation), following the prompt formats
of EleutherAI's lm-evaluation-harness so numbers for reference models are roughly
comparable to published ones:

  - multiple choice (hellaswag, arc_*, piqa, xstorycloze_*, xcopa_zh): pick the
    continuation with the highest total log-prob (`acc`) or the highest log-prob per
    character of the continuation (`acc_norm`).
  - winogrande: "partial" scoring — substitute each option into the blank and score
    the shared remainder of the sentence.
  - lambada_openai: greedy last-word accuracy + perplexity of the last word.
  - bits-per-byte (BPB) on raw text: tokenizer-independent language-model quality, so a
    64K-vocab MykLLM and a 150K-vocab Qwen can be compared directly.

Usage:
  python -m myk_llm.eval.bench --model runs/200m-pt/checkpoints/step_0016000
  python -m myk_llm.eval.bench --hf-model HuggingFaceTB/SmolLM2-135M --tasks hellaswag,arc_easy
Results are appended as one JSON line per model to --output (default eval_results.jsonl).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, cast

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


# ----------------------------------------------------------------------------- models


class LogLikelihoodModel:
    """Scores (context, continuation) pairs; subclasses provide the raw forward pass."""

    name: str
    tokenizer: PreTrainedTokenizerBase
    max_len: int
    device = torch.device("cuda")

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return cast(list[int], self.tokenizer.encode(text, add_special_tokens=False))

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[tuple[float, bool, int]]:
        """Text-level wrapper: context and continuation are tokenized separately then
        concatenated (the lm-eval convention), so the continuation boundary is exact."""
        encoded: list[tuple[list[int], list[int]]] = []
        for context, continuation in requests:
            cont_ids = self.encode(continuation)
            if not cont_ids:
                raise ValueError(f"empty continuation for context {context[:60]!r}")
            encoded.append((self.encode(context), cont_ids))
        return self.loglikelihood_tokens(encoded)

    @torch.no_grad()
    def loglikelihood_tokens(
        self, encoded: list[tuple[list[int], list[int]]]
    ) -> list[tuple[float, bool, int]]:
        """(sum log p(continuation | context), continuation is the greedy decode, #cont tokens)."""
        # Keep the full bf16 (batch, seq, vocab) logits around ~4 GB: a 248K-vocab Qwen
        # gets ~8K tokens per batch, the 64K-vocab MykLLM ~32K.
        batch_tokens = min(32768, int(4e9 // (2 * len(self.tokenizer))))
        results: list[tuple[float, bool, int] | None] = [None] * len(encoded)
        # Longest first so a batch holds similar lengths (less padding).
        order = sorted(range(len(encoded)), key=lambda i: -(len(encoded[i][0]) + len(encoded[i][1])))
        start = 0
        while start < len(order):
            first = order[start]
            seq_len = min(self.max_len, len(encoded[first][0]) + len(encoded[first][1]))
            padded_len = ((seq_len + 63) // 64) * 64  # bucket shapes to limit recompiles
            batch_size = max(1, batch_tokens // padded_len)
            chunk = order[start : start + batch_size]
            start += len(chunk)

            rows: list[list[int]] = []
            cont_spans: list[tuple[int, int]] = []
            for index in chunk:
                ctx_ids, cont_ids = encoded[index]
                full = (ctx_ids + cont_ids)[-(self.max_len + 1) :]
                # Inputs drop the final token; the continuation is predicted from
                # positions [len(full)-1-len(cont), len(full)-1).
                inputs = full[:-1]
                cont_len = min(len(cont_ids), len(inputs))
                rows.append(inputs)
                cont_spans.append((len(inputs) - cont_len, len(inputs)))
            pad_id = 0
            input_ids = torch.full((len(rows), padded_len), pad_id, dtype=torch.long)
            for row_index, row in enumerate(rows):
                input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
            logits = self.forward_logits(input_ids.to(self.device))

            for row_index, index in enumerate(chunk):
                lo, hi = cont_spans[row_index]
                cont_ids = encoded[index][1][-(hi - lo) :]
                row_logits = logits[row_index, lo:hi].float()
                logprobs = F.log_softmax(row_logits, dim=-1)
                target = torch.tensor(cont_ids, device=self.device)
                total = float(logprobs.gather(1, target[:, None]).sum())
                greedy = bool((row_logits.argmax(-1) == target).all())
                results[index] = (total, greedy, len(cont_ids))
        return cast(list[tuple[float, bool, int]], results)


class MykModel(LogLikelihoodModel):
    def __init__(self, model_dir: str, max_len: int = 2048) -> None:
        from ..model import load_model

        self.model = load_model(model_dir, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.max_len = min(max_len, self.model.config.max_seq_len)
        self.name = "/".join(Path(model_dir).resolve().parts[-3:])
        self.num_params = self.model.num_parameters()

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.model.logits(self.model(input_ids).hidden)


class HFModel(LogLikelihoodModel):
    def __init__(self, model_path: str, max_len: int = 2048) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(self.device)
        self.model.eval()
        self.max_len = max_len
        self.name = model_path.rstrip("/").split("/")[-1]
        self.num_params = sum(p.numel() for p in self.model.parameters())

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Right padding + causal attention: real positions never see the padding.
        return self.model(input_ids=input_ids).logits


# ----------------------------------------------------------------------------- tasks


@dataclass
class MCExample:
    context: str
    choices: list[str]  # continuations, already carrying any leading space
    label: int


def score_multiple_choice(model: LogLikelihoodModel, examples: list[MCExample]) -> dict[str, float]:
    requests = [(ex.context, choice) for ex in examples for choice in ex.choices]
    scores = model.loglikelihood(requests)
    correct = correct_norm = 0
    cursor = 0
    for ex in examples:
        chunk = scores[cursor : cursor + len(ex.choices)]
        cursor += len(ex.choices)
        lls = [s[0] for s in chunk]
        norm = [ll / max(1, len(choice)) for ll, choice in zip(lls, ex.choices)]
        correct += int(max(range(len(lls)), key=lls.__getitem__) == ex.label)
        correct_norm += int(max(range(len(norm)), key=norm.__getitem__) == ex.label)
    n = len(examples)
    return {"acc": correct / n, "acc_norm": correct_norm / n, "n": n}


def _hellaswag_preprocess(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def task_hellaswag(limit: int | None) -> list[MCExample]:
    rows = load_dataset("Rowan/hellaswag", split="validation")
    out = []
    for row in rows:
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        context = _hellaswag_preprocess(row["activity_label"] + ": " + ctx)
        choices = [" " + _hellaswag_preprocess(e) for e in row["endings"]]
        out.append(MCExample(context, choices, int(row["label"])))
    return out[:limit]


def _arc(subset: str, limit: int | None) -> list[MCExample]:
    rows = load_dataset("allenai/ai2_arc", subset, split="test")
    out = []
    for row in rows:
        labels = row["choices"]["label"]
        if row["answerKey"] not in labels:
            continue
        out.append(
            MCExample(
                f"Question: {row['question']}\nAnswer:",
                [" " + text for text in row["choices"]["text"]],
                labels.index(row["answerKey"]),
            )
        )
    return out[:limit]


def task_piqa(limit: int | None) -> list[MCExample]:
    rows = load_dataset("ybisk/piqa", revision="refs/convert/parquet", split="validation")
    return [
        MCExample(f"Question: {row['goal']}\nAnswer:", [" " + row["sol1"], " " + row["sol2"]], int(row["label"]))
        for row in rows
    ][:limit]


def task_xstorycloze(lang: str, limit: int | None) -> list[MCExample]:
    rows = load_dataset("juletxara/xstory_cloze", lang, split="eval")
    sep = "" if lang == "zh" else " "
    out = []
    for row in rows:
        context = sep.join(row[f"input_sentence_{i}"].strip() for i in range(1, 5))
        choices = [sep + row["sentence_quiz1"].strip(), sep + row["sentence_quiz2"].strip()]
        out.append(MCExample(context, choices, int(row["answer_right_ending"]) - 1))
    return out[:limit]


def task_xcopa_zh(limit: int | None) -> list[MCExample]:
    rows = load_dataset("cambridgeltl/xcopa", "zh", split="test")
    out = []
    for row in rows:
        connector = "因为" if row["question"] == "cause" else "所以"
        context = row["premise"].strip().rstrip("。.") + "，" + connector
        out.append(MCExample(context, [row["choice1"], row["choice2"]], int(row["label"])))
    return out[:limit]


def eval_winogrande(model: LogLikelihoodModel, limit: int | None) -> dict[str, float]:
    rows = list(load_dataset("allenai/winogrande", "winogrande_xl", split="validation"))[:limit]
    requests = []
    for row in rows:
        blank = row["sentence"].index("_")
        suffix = " " + row["sentence"][blank + 1 :].strip()
        for option in (row["option1"], row["option2"]):
            requests.append((row["sentence"][:blank] + option, suffix))
    scores = model.loglikelihood(requests)
    correct = sum(
        int((0 if scores[2 * i][0] > scores[2 * i + 1][0] else 1) == int(row["answer"]) - 1)
        for i, row in enumerate(rows)
    )
    return {"acc": correct / len(rows), "n": len(rows)}


def eval_lambada(model: LogLikelihoodModel, limit: int | None) -> dict[str, float]:
    rows = list(load_dataset("EleutherAI/lambada_openai", "en", split="test"))[:limit]
    requests = []
    for row in rows:
        context, word = row["text"].rsplit(" ", 1)
        requests.append((context, " " + word))
    scores = model.loglikelihood(requests)
    total_ll = sum(s[0] for s in scores)
    return {
        "acc": sum(int(s[1]) for s in scores) / len(rows),
        "ppl": math.exp(-total_ll / len(rows)),
        "n": len(rows),
    }


def bpb_documents(name: str, limit: int | None) -> list[str]:
    if name == "wikitext2":
        # Rebuild whole articles from the line-split test set (top-level " = Title = " headers).
        text = "".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        docs = [d for d in re.split(r"\n(?= = [^=].* = \n)", text) if d.strip()]
    elif name == "fineweb_edu_en":
        rows = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        # Skip well past anything a small streaming-shuffle run would start on; still
        # only *approximately* held out (the training stream used the same corpus).
        docs = [row["text"] for row in rows.skip(200_000).take(limit or 300)]
    elif name == "fineweb_edu_zh":
        rows = load_dataset("opencsg/Fineweb-Edu-Chinese-V2.1", data_dir="4_5", split="train", streaming=True)
        docs = [row["text"] for row in rows.skip(200_000).take(limit or 300)]
    elif name == "wiki_zh":
        rows = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", streaming=True)
        docs = [row["text"] for row in rows.skip(50_000).take(4 * (limit or 300)) if len(row["text"]) > 400]
    else:
        raise ValueError(name)
    return docs[: (limit or 300)]


def eval_bpb(model: LogLikelihoodModel, docs: list[str], window: int = 1024) -> dict[str, float]:
    """Bits per UTF-8 byte over whole documents.

    The first window of a doc is scored from a newline prompt; each later chunk of
    `window // 2` tokens is conditioned on the preceding `window // 2` tokens, so most
    tokens see a reasonable amount of context.
    """
    stride = window // 2
    newline = model.encode("\n")
    requests: list[tuple[list[int], list[int]]] = []
    total_bytes = 0
    for doc in docs:
        ids = model.encode(doc)
        total_bytes += len(doc.encode("utf-8"))
        requests.append((newline, ids[:window]))
        for pos in range(window, len(ids), stride):
            requests.append((ids[pos - stride : pos], ids[pos : pos + stride]))
    scores = model.loglikelihood_tokens(requests)
    nats = -sum(s[0] for s in scores)
    num_tokens = sum(s[2] for s in scores)
    return {"bpb": nats / math.log(2) / total_bytes, "bytes_per_token": total_bytes / num_tokens, "docs": len(docs)}


MC_TASKS: dict[str, Callable[[int | None], list[MCExample]]] = {
    "hellaswag": task_hellaswag,
    "arc_easy": lambda limit: _arc("ARC-Easy", limit),
    "arc_challenge": lambda limit: _arc("ARC-Challenge", limit),
    "piqa": task_piqa,
    "xstorycloze_en": lambda limit: task_xstorycloze("en", limit),
    "xstorycloze_zh": lambda limit: task_xstorycloze("zh", limit),
    "xcopa_zh": task_xcopa_zh,
}
OTHER_TASKS = ["winogrande", "lambada_openai", "bpb_wikitext2", "bpb_fineweb_edu_en", "bpb_fineweb_edu_zh", "bpb_wiki_zh"]
ALL_TASKS = [*MC_TASKS, *OTHER_TASKS]


def run_tasks(model: LogLikelihoodModel, tasks: Iterable[str], limit: int | None) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for task in tasks:
        started = time.perf_counter()
        if task in MC_TASKS:
            metrics = score_multiple_choice(model, MC_TASKS[task](limit))
        elif task == "winogrande":
            metrics = eval_winogrande(model, limit)
        elif task == "lambada_openai":
            metrics = eval_lambada(model, limit)
        elif task.startswith("bpb_"):
            metrics = eval_bpb(model, bpb_documents(task.removeprefix("bpb_"), limit))
        else:
            raise ValueError(f"unknown task {task!r}; known: {ALL_TASKS}")
        results[task] = metrics
        shown = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        print(f"[{model.name}] {task}: {shown} ({time.perf_counter() - started:.0f}s)", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=str, help="exported MykLLM model directory")
    source.add_argument("--hf-model", type=str, help="HuggingFace causal LM id or path (reference)")
    parser.add_argument("--tasks", type=str, default=",".join(ALL_TASKS))
    parser.add_argument("--limit", type=int, default=None, help="cap examples per task (quick runs)")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--output", type=str, default="eval_results.jsonl")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    model: LogLikelihoodModel
    if args.model:
        model = MykModel(args.model, args.max_len)
    else:
        model = HFModel(args.hf_model, args.max_len)
    print(f"model={model.name} params={model.num_params / 1e6:.1f}M")

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    results = run_tasks(model, tasks, args.limit)
    with Path(args.output).open("a", encoding="utf-8") as handle:
        record = {"model": model.name, "params": model.num_params, "limit": args.limit, "results": results}
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
