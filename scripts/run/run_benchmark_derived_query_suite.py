#!/usr/bin/env python3
"""Benchmark-derived Sparse Readout Prism query-fidelity suite.

This runner mines real task-format examples from public benchmarks, evaluates
exact-vs-sparse readout margins for selected Qwen3.5 checkpoints, and writes
paper-facing aggregate tables plus labeled feature-display candidates.

The suite is intentionally about scalar readout decisions, not task accuracy:
each row asks whether the sparse readout reconstruction preserves a concrete
margin such as answer-vs-distractor, abstention-vs-entity, or safe-vs-unsafe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from sparse_readout_prism.research.qwen_readout import find_lm_head, load_sae
from sparse_readout_prism.research.query_decompose import (
    QuerySpec,
    attach_feature_labels,
    build_query_weights,
    decompose_query,
    load_qwen_model,
    resolve_single_token,
    token_rank_and_prob,
    write_csv,
)
from sparse_readout_prism.token_display import clean_token
from sparse_readout_prism.paths import ssd_root


ARCHIVE = ssd_root() / "readout-prism-archive/converged-all/results"
DEFAULT_OUT_DIR = Path("results/benchmark_derived_query_suite_qwen_20260523")
DENOM_FLOOR = 0.5


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "qwen0p8b": {
        "label": "Qwen3.5-0.8B",
        "model_id": "Qwen/Qwen3.5-0.8B",
        "checkpoint": ARCHIVE
        / "qwen35_0p8b_sae/converge/topk_d32768_32x_k256_rowseeded_hybrid_lamramp_s0/checkpoint.pt",
        "k": 256,
    },
    "qwen2b": {
        "label": "Qwen3.5-2B",
        "model_id": "Qwen/Qwen3.5-2B",
        "checkpoint": ARCHIVE / "qwen35_2b_sae/converge/topk_d65536_32x_k256_rowseeded_hybrid_lamramp_s0/checkpoint.pt",
        "k": 256,
    },
    "qwen9b": {
        "label": "Qwen3.5-9B",
        "model_id": "Qwen/Qwen3.5-9B",
        "checkpoint": ARCHIVE
        / "qwen9b_no_pca_width_k_sweep/converged/topk_d131072_k256_rowseeded_hybrid_lamramp_s0/checkpoint.pt",
        "k": 256,
    },
}


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    family: str
    source_dataset: str
    source_split: str
    source_id: str
    prompt: str
    query_kind: str
    target_a: str | None
    target_b: str | None
    target_a_family: tuple[str, ...]
    target_b_family: tuple[str, ...]
    expected_side: str
    note: str

    def to_query_spec(self) -> QuerySpec:
        return QuerySpec(
            case_id=self.case_id,
            role=FAMILY_ROLE.get(self.family, self.family),
            query_kind=self.query_kind,
            title=FAMILY_TITLE.get(self.family, self.family),
            prompt=self.prompt,
            note=self.note,
            target_a=self.target_a,
            target_b=self.target_b,
            target_family=self.target_a_family,
            contrast_family=self.target_b_family,
        )


FAMILY_ROLE = {
    "squad2_answerable": "RAG / extractive QA",
    "squad2_unanswerable": "Unanswerable QA",
    "hotpotqa_distractor": "Multi-hop QA",
    "legalbench_contractnli": "Contract NLI",
    "securityeval": "Secure code action",
}

FAMILY_TITLE = {
    "squad2_answerable": "Correct entity over context distractor",
    "squad2_unanswerable": "Abstention over plausible entity",
    "hotpotqa_distractor": "Supporting-chain answer over distractor",
    "legalbench_contractnli": "Legal entailment label contrast",
    "securityeval": "Secure action over unsafe action",
}

STOP_CANDIDATES = {
    "A",
    "An",
    "And",
    "After",
    "As",
    "At",
    "Before",
    "By",
    "During",
    "For",
    "From",
    "He",
    "Her",
    "His",
    "I",
    "In",
    "It",
    "King",
    "Later",
    "Many",
    "Most",
    "No",
    "Of",
    "On",
    "One",
    "Other",
    "Several",
    "She",
    "Some",
    "That",
    "The",
    "These",
    "They",
    "Those",
    "This",
    "To",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
    "Yes",
}
ANSWER_STOP = {word.lower() for word in STOP_CANDIDATES} | {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "two",
    "with",
}
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z-]{2,}\b")
CAP_RE = re.compile(r"\b[A-Z][A-Za-z-]{2,}\b")


def log(msg: str) -> None:
    print(f"[bench-suite {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_simple_word(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z-]{1,40}", text.strip()))


def clean_candidate(text: str) -> str:
    return re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", text.strip())


def single_token_ok(tokenizer, text: str) -> bool:
    tid, _used, _label, reason = resolve_single_token(tokenizer, text)
    return tid is not None and reason == "ok"


def first_answer_token(answer: str) -> str | None:
    answer = answer.strip()
    if is_simple_word(answer):
        return answer
    match = WORD_RE.search(answer)
    if match:
        return clean_candidate(match.group(0))
    return None


def is_content_answer_token(token: str) -> bool:
    token = clean_candidate(token)
    return bool(token) and is_simple_word(token) and token.lower() not in ANSWER_STOP


def is_entity_answer_token(token: str) -> bool:
    token = clean_candidate(token)
    return is_content_answer_token(token) and token[0].isupper()


def candidate_words(text: str, *, capitalized: bool = True) -> list[str]:
    regex = CAP_RE if capitalized else WORD_RE
    seen: set[str] = set()
    out: list[str] = []
    for raw in regex.findall(text):
        word = clean_candidate(raw)
        if not word or word in STOP_CANDIDATES or word.lower() in ANSWER_STOP or len(word) < 3:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(word)
    return out


def choose_distractors(
    tokenizer,
    text: str,
    target: str,
    *,
    capitalized: bool = True,
    limit: int = 1,
) -> list[str]:
    target_l = target.lower()
    out: list[str] = []
    for cand in candidate_words(text, capitalized=capitalized):
        if cand.lower() == target_l or target_l in cand.lower() or cand.lower() in target_l:
            continue
        if single_token_ok(tokenizer, cand):
            out.append(cand)
            if len(out) >= limit:
                break
    return out


def choose_distractor(tokenizer, text: str, target: str, *, capitalized: bool = True) -> str | None:
    rows = choose_distractors(tokenizer, text, target, capitalized=capitalized, limit=1)
    return rows[0] if rows else None


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def build_squad_answerable(tokenizer, limit: int) -> list[BenchCase]:
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    cases: list[BenchCase] = []
    for row in ds:
        answers = row["answers"]["text"]
        if not answers:
            continue
        target = first_answer_token(answers[0])
        if not target or not is_entity_answer_token(target) or not single_token_ok(tokenizer, target):
            continue
        distractors = choose_distractors(tokenizer, row["context"], target, capitalized=True, limit=3)
        if not distractors:
            continue
        query_kind = "token_family_margin"
        prompt = (
            "Benchmark: SQuAD2 answerable extractive QA.\n"
            "Use the context to answer in a few words.\n"
            f"Title: {row['title']}\n"
            f"Context: {compact_text(row['context'], 1500)}\n"
            f"Question: {row['question']}\n"
            "Answer:"
        )
        cases.append(
            BenchCase(
                case_id=f"squad2_ans_{row['id']}",
                family="squad2_answerable",
                source_dataset="rajpurkar/squad_v2",
                source_split="validation",
                source_id=str(row["id"]),
                prompt=prompt,
                query_kind=query_kind,
                target_a=None if query_kind == "token_family_margin" else target,
                target_b=None if query_kind == "token_family_margin" else distractors[0],
                target_a_family=(target,) if query_kind == "token_family_margin" else (),
                target_b_family=tuple(distractors) if query_kind == "token_family_margin" else (),
                expected_side="A",
                note=(
                    f"Gold answer token {target!r}; context distractors {', '.join(repr(d) for d in distractors)}."
                    if query_kind == "token_family_margin"
                    else f"Gold answer token {target!r}; context distractor {distractors[0]!r}."
                ),
            )
        )
        if len(cases) >= limit:
            break
    return cases


def build_squad_unanswerable(tokenizer, limit: int) -> list[BenchCase]:
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    cases: list[BenchCase] = []
    abstain = ("unknown", "Unknown", "unavailable", "insufficient", "not")
    for row in ds:
        if row["answers"]["text"]:
            continue
        contrasts = choose_distractors(tokenizer, row["context"], row["title"], capitalized=True, limit=3)
        if not contrasts:
            continue
        prompt = (
            "Benchmark: SQuAD2 unanswerable extractive QA.\n"
            'If the context does not answer the question, answer "unknown".\n'
            f"Title: {row['title']}\n"
            f"Context: {compact_text(row['context'], 1500)}\n"
            f"Question: {row['question']}\n"
            "Answer:"
        )
        cases.append(
            BenchCase(
                case_id=f"squad2_unans_{row['id']}",
                family="squad2_unanswerable",
                source_dataset="rajpurkar/squad_v2",
                source_split="validation",
                source_id=str(row["id"]),
                prompt=prompt,
                query_kind="token_family_margin",
                target_a=None,
                target_b=None,
                target_a_family=abstain,
                target_b_family=tuple(contrasts),
                expected_side="A",
                note=(
                    "SQuAD2 unanswerable row; abstention family vs context entities "
                    f"{', '.join(repr(c) for c in contrasts)}."
                ),
            )
        )
        if len(cases) >= limit:
            break
    return cases


def supporting_context(row: dict[str, Any], distractor: str) -> str:
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    by_title = {title: sents for title, sents in zip(titles, sentences, strict=True)}
    support_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
    blocks: list[str] = []
    for title in support_titles:
        sents = by_title.get(title, [])[:3]
        if sents:
            blocks.append(f"Title: {title}\n" + " ".join(s.strip() for s in sents))
    for title, sents in by_title.items():
        if title in support_titles:
            continue
        joined = " ".join(s.strip() for s in sents[:3])
        if distractor in title or distractor in joined:
            blocks.append(f"Title: {title}\n{joined}")
            break
    return "\n\n".join(blocks)


def build_hotpot(tokenizer, limit: int) -> list[BenchCase]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    cases: list[BenchCase] = []
    for row in ds:
        answer = row["answer"]
        if answer.lower() in {"yes", "no"}:
            continue
        target = first_answer_token(answer)
        if not target or not is_content_answer_token(target) or not single_token_ok(tokenizer, target):
            continue
        support_titles = set(row["supporting_facts"]["title"])
        distractor_text = []
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"], strict=True):
            if title in support_titles:
                continue
            distractor_text.append(title)
            distractor_text.extend(sents[:2])
        distractors = choose_distractors(tokenizer, " ".join(distractor_text), target, capitalized=True, limit=3)
        if not distractors:
            continue
        context = supporting_context(row, distractors[0])
        if not context:
            continue
        query_kind = "token_family_margin"
        prompt = (
            "Benchmark: HotpotQA distractor multi-hop QA.\n"
            "Use the supporting passages and ignore distractors.\n"
            f"{compact_text(context, 2600)}\n"
            f"Question: {row['question']}\n"
            "Answer:"
        )
        cases.append(
            BenchCase(
                case_id=f"hotpot_{row['id']}",
                family="hotpotqa_distractor",
                source_dataset="hotpotqa/hotpot_qa",
                source_split="validation[distractor]",
                source_id=str(row["id"]),
                prompt=prompt,
                query_kind=query_kind,
                target_a=None if query_kind == "token_family_margin" else target,
                target_b=None if query_kind == "token_family_margin" else distractors[0],
                target_a_family=(target,) if query_kind == "token_family_margin" else (),
                target_b_family=tuple(distractors) if query_kind == "token_family_margin" else (),
                expected_side="A",
                note=(
                    f"Answer token {target!r}; distractor-passage tokens {', '.join(repr(d) for d in distractors)}."
                    if query_kind == "token_family_margin"
                    else f"Answer token {target!r}; distractor-passage token {distractors[0]!r}."
                ),
            )
        )
        if len(cases) >= limit:
            break
    return cases


def build_legalbench(tokenizer, limit: int) -> list[BenchCase]:
    _ = tokenizer
    ds = load_dataset(
        "nguha/legalbench",
        "contract_nli_confidentiality_of_agreement",
        split="test",
    )
    cases: list[BenchCase] = []
    assertion = "The agreement's existence or terms are confidential."
    for row in ds:
        answer = "Yes" if str(row["answer"]).strip().lower() == "yes" else "No"
        contrast = "No" if answer == "Yes" else "Yes"
        query_kind = "token_family_margin"
        yes_family = ("Yes", "yes", "YES")
        no_family = ("No", "no", "NO")
        prompt = (
            "Benchmark: LegalBench contract_nli_confidentiality_of_agreement.\n"
            f"Contract excerpt: {compact_text(row['text'], 1200)}\n"
            f"Assertion: {assertion}\n"
            "Does the excerpt support the assertion? Answer Yes or No:"
        )
        cases.append(
            BenchCase(
                case_id=f"legalbench_contractnli_{row['index']}",
                family="legalbench_contractnli",
                source_dataset="nguha/legalbench",
                source_split="test[contract_nli_confidentiality_of_agreement]",
                source_id=str(row["index"]),
                prompt=prompt,
                query_kind=query_kind,
                target_a=None if query_kind == "token_family_margin" else answer,
                target_b=None if query_kind == "token_family_margin" else contrast,
                target_a_family=(yes_family if answer == "Yes" else no_family)
                if query_kind == "token_family_margin"
                else (),
                target_b_family=(no_family if answer == "Yes" else yes_family)
                if query_kind == "token_family_margin"
                else (),
                expected_side="A",
                note=(
                    f"LegalBench label-family {answer}; document {row['document_name']}."
                    if query_kind == "token_family_margin"
                    else f"LegalBench label {answer}; document {row['document_name']}."
                ),
            )
        )
        if len(cases) >= limit:
            break
    return cases


def security_contrast(row: dict[str, Any]) -> tuple[str, str] | None:
    text = (row["Prompt"] + "\n" + row["Insecure_code"]).lower()
    if "yaml" in text and "load" in text:
        return "safe", "load"
    if "pickle" in text or "deserialize" in text:
        return "validate", "deserialize"
    if "redirect" in text:
        return "validate", "redirect"
    if "tar.extract" in text or "extract(" in text:
        return "validate", "extract"
    if "os.system" in text or "subprocess" in text or "shell=true" in text:
        return "validate", "execute"
    if "eval(" in text:
        return "validate", "eval"
    if "exec(" in text or "compile(" in text:
        return "validate", "exec"
    if "select *" in text or "delete from" in text or "where username" in text:
        return "quote", "format"
    if "ldap" in text:
        return "escape", "format"
    if "html" in text or "jinja" in text or "script" in text or "xss" in text:
        return "escape", "concatenate"
    if "open(" in text or "send_file" in text or "remove(" in text:
        return "validate", "open"
    if "logging" in text:
        return "sanitize", "raw"
    if "response" in text or "header" in text:
        return "sanitize", "raw"
    return None


SECURITY_FAMILIES: dict[str, tuple[str, ...]] = {
    "safe": ("safe", "validate", "sanitize"),
    "validate": ("validate", "verify", "check"),
    "escape": ("escape", "sanitize", "encode"),
    "sanitize": ("sanitize", "filter", "redact"),
    "quote": ("quote", "parameterize", "escape"),
    "load": ("load", "unsafe", "raw"),
    "deserialize": ("deserialize", "pickle", "load"),
    "redirect": ("redirect", "url", "raw"),
    "extract": ("extract", "open", "unsafe"),
    "execute": ("execute", "exec", "eval"),
    "eval": ("eval", "exec", "execute"),
    "exec": ("exec", "execute", "eval"),
    "format": ("format", "concatenate", "raw"),
    "concatenate": ("concatenate", "format", "raw"),
    "open": ("open", "path", "raw"),
    "raw": ("raw", "unsafe", "untrusted"),
}


def build_securityeval(tokenizer, limit: int) -> list[BenchCase]:
    ds = load_dataset("s2e-lab/SecurityEval", split="train")
    cases: list[BenchCase] = []
    for row in ds:
        pair = security_contrast(row)
        if pair is None:
            continue
        target, contrast = pair
        if not single_token_ok(tokenizer, target) or not single_token_ok(tokenizer, contrast):
            continue
        query_kind = "token_family_margin"
        target_family = SECURITY_FAMILIES.get(target, (target,))
        contrast_family = SECURITY_FAMILIES.get(contrast, (contrast,))
        prompt = f"Benchmark: SecurityEval insecure code-generation prompt.\n{compact_text(row['Prompt'], 1200)}\n\n"
        prompt += (
            "Which keyword family better describes the safer implementation choice: "
            f"{'/'.join(target_family)} or {'/'.join(contrast_family)}?\n"
        )
        prompt += "Answer:"
        cases.append(
            BenchCase(
                case_id=f"securityeval_{Path(row['ID']).stem}",
                family="securityeval",
                source_dataset="s2e-lab/SecurityEval",
                source_split="train",
                source_id=str(row["ID"]),
                prompt=prompt,
                query_kind=query_kind,
                target_a=None if query_kind == "token_family_margin" else target,
                target_b=None if query_kind == "token_family_margin" else contrast,
                target_a_family=target_family if query_kind == "token_family_margin" else (),
                target_b_family=contrast_family if query_kind == "token_family_margin" else (),
                expected_side="A",
                note=(
                    f"SecurityEval {row['ID']}; safer family {target_family!r} vs unsafe family {contrast_family!r}."
                    if query_kind == "token_family_margin"
                    else f"SecurityEval {row['ID']}; safer keyword {target!r} vs unsafe keyword {contrast!r}."
                ),
            )
        )
        if len(cases) >= limit:
            break
    return cases


def build_bank(tokenizer, counts: dict[str, int]) -> list[BenchCase]:
    builders = [
        ("squad2_answerable", build_squad_answerable),
        ("squad2_unanswerable", build_squad_unanswerable),
        ("hotpotqa_distractor", build_hotpot),
        ("legalbench_contractnli", build_legalbench),
        ("securityeval", build_securityeval),
    ]
    cases: list[BenchCase] = []
    for family, builder in builders:
        want = int(counts.get(family, 0))
        if want <= 0:
            continue
        rows = builder(tokenizer, want)
        log(f"mined {family}: {len(rows)}/{want}")
        cases.extend(rows)
    return cases


@torch.no_grad()
def collect_readout_state(
    model, tokenizer, prompt: str, device: torch.device, max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    lm_head, _path = find_lm_head(model)
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, inputs):
        captured["h"] = (inputs[0] if isinstance(inputs, tuple) else inputs).detach()

    old_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    handle = lm_head.register_forward_pre_hook(pre_hook)
    try:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        out = model(**enc)
    finally:
        handle.remove()
        tokenizer.truncation_side = old_side
    h = captured["h"][0, -1].float().cpu().contiguous()
    logits = out.logits[0, -1].float().cpu().contiguous()
    return h, logits


def scalar_metrics(summary: dict[str, object]) -> dict[str, object]:
    exact = float(summary["exact_score"])
    sparse = float(summary["sparse_feature_sum"])
    residual = float(summary["residual"])
    sign_match = int(np.sign(exact) == np.sign(sparse))
    rho_floor = abs(residual) / max(abs(exact), DENOM_FLOOR)
    rho_plus = abs(residual) / (abs(exact) + DENOM_FLOOR)
    return {
        "exact_abs": abs(exact),
        "rho_floor0p5": rho_floor,
        "rho_plus0p5": rho_plus,
        "sign_match": sign_match,
        "accepted_rho_floor0p5_lt_0p5": int(sign_match and rho_floor < 0.5),
        "accepted_rho_plus0p5_lt_0p5": int(sign_match and rho_plus < 0.5),
        "target_side_wins": int(exact > 0.0),
    }


def aggregate(rows: list[dict[str, object]], skipped: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["model_slug"]), str(row["family"])), []).append(row)
        groups.setdefault((str(row["model_slug"]), "ALL"), []).append(row)
    skipped_counts: dict[tuple[str, str], int] = {}
    for row in skipped:
        if "family" not in row:
            continue
        key = (str(row["model_slug"]), str(row["family"]))
        skipped_counts[key] = skipped_counts.get(key, 0) + 1
        all_key = (str(row["model_slug"]), "ALL")
        skipped_counts[all_key] = skipped_counts.get(all_key, 0) + 1
    out: list[dict[str, object]] = []
    for (model_slug, family), group in sorted(groups.items()):
        rho = [float(r["rho_floor0p5"]) for r in group]
        rho_plus = [float(r["rho_plus0p5"]) for r in group]
        sign = [int(r["sign_match"]) for r in group]
        accept = [int(r["accepted_rho_floor0p5_lt_0p5"]) for r in group]
        target = [int(r["target_side_wins"]) for r in group]
        exact_abs = [float(r["exact_abs"]) for r in group]
        out.append(
            {
                "model_slug": model_slug,
                "family": family,
                "n_rows": len(group),
                "n_skipped": skipped_counts.get((model_slug, family), 0),
                "median_rho_floor0p5": median(rho),
                "p90_rho_floor0p5": float(np.percentile(rho, 90)),
                "median_rho_plus0p5": median(rho_plus),
                "sign_agreement": sum(sign) / len(sign),
                "accepted_rate_rho_floor0p5_lt_0p5": sum(accept) / len(accept),
                "target_side_win_rate": sum(target) / len(target),
                "median_abs_exact_margin": median(exact_abs),
                "confident_abs_margin_ge_1_rate": sum(v >= 1.0 for v in exact_abs) / len(exact_abs),
                "confident_abs_margin_ge_2_rate": sum(v >= 2.0 for v in exact_abs) / len(exact_abs),
            }
        )
    return out


def choose_display_cases(rows: list[dict[str, object]], *, model_slug: str, per_family: int) -> set[str]:
    out: set[str] = set()
    for family in FAMILY_ROLE:
        candidates = [
            r
            for r in rows
            if r["model_slug"] == model_slug
            and r["family"] == family
            and int(r["sign_match"]) == 1
            and float(r["rho_floor0p5"]) < 0.35
            and float(r["exact_abs"]) >= 0.5
        ]
        candidates.sort(key=lambda r: (float(r["rho_floor0p5"]), -float(r["exact_abs"])))
        out.update(str(r["case_id"]) for r in candidates[:per_family])
    return out


def evaluate_model(
    *,
    model_slug: str,
    spec: dict[str, Any],
    cases: list[BenchCase],
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    model_id = str(spec["model_id"])
    checkpoint = Path(spec["checkpoint"])
    if not checkpoint.exists():
        log(f"skip {model_slug}: missing checkpoint {checkpoint}")
        return [], [], [{"model_slug": model_slug, "reason": "missing_checkpoint", "checkpoint": str(checkpoint)}], []
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)
    try:
        model, tokenizer = load_qwen_model(model_id, dtype, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        if args.skip_missing:
            log(f"skip {model_slug}: could not load cached model {model_id}: {exc}")
            return (
                [],
                [],
                [{"model_slug": model_slug, "reason": "missing_model", "model_id": model_id, "error": str(exc)}],
                [],
            )
        raise
    model.to(device)
    lm_head, lm_head_path = find_lm_head(model)
    W = lm_head.weight.detach().cpu().float().contiguous()
    row_mean = W.mean(dim=0).contiguous()
    decoder, encoder_w, encoder_b, sae_config = load_sae(checkpoint)
    if decoder.shape[1] != W.shape[1]:
        raise ValueError(f"{model_slug}: SAE d_model {decoder.shape[1]} != W_U d_model {W.shape[1]}")

    case_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    token_audit_rows: list[dict[str, object]] = []

    log(f"evaluating {model_slug}: {len(cases)} cases on {device}")
    for idx, bench in enumerate(cases, start=1):
        if idx % max(1, args.progress_every) == 0:
            log(f"{model_slug}: {idx}/{len(cases)}")
        qspec = bench.to_query_spec()
        try:
            h, _model_logits = collect_readout_state(model, tokenizer, bench.prompt, device, args.max_len)
            logits = h @ W.T
            top_values, top_ids = torch.topk(logits, k=5)
            weights, mean_subtract, metadata = build_query_weights(
                spec=qspec,
                logits=logits,
                tokenizer=tokenizer,
                token_audit_rows=token_audit_rows,
            )
            summary, rows = decompose_query(
                h=h,
                logits=logits,
                W=W,
                row_mean=row_mean,
                decoder=decoder,
                encoder_w=encoder_w,
                encoder_b=encoder_b,
                k=int(spec["k"]),
                weights=weights,
                mean_subtract=mean_subtract,
            )
            metrics = scalar_metrics(summary)
            target_id = metadata.get("target_token_id", "")
            target_rank = ""
            target_probability = ""
            if isinstance(target_id, int):
                target_rank, target_probability = token_rank_and_prob(logits, target_id)
            case_row = {
                **asdict(bench),
                "model_slug": model_slug,
                "model_label": spec["label"],
                "model_id": model_id,
                "checkpoint": str(checkpoint),
                "k": int(spec["k"]),
                "lm_head_path": lm_head_path,
                "vocab": int(W.shape[0]),
                "d_model": int(W.shape[1]),
                "target_rank": target_rank,
                "target_probability": target_probability,
                "top5_token_ids": ";".join(str(int(tid)) for tid in top_ids.tolist()),
                "top5_tokens": "; ".join(clean_token(tokenizer, int(tid)) for tid in top_ids.tolist()),
                "top5_logits": ";".join(f"{float(v):.6g}" for v in top_values.tolist()),
                "weights": json.dumps({str(k): v for k, v in sorted(weights.items())}),
                **metadata,
                **summary,
                **metrics,
            }
            case_rows.append(case_row)
            top_feature_rows = sorted(rows, key=lambda r: -float(r["abs_contribution"]))[: args.top_features]
            for rank, row in enumerate(top_feature_rows, start=1):
                feature_rows.append({**case_row, **row, "display_rank_abs": rank})
        except Exception as exc:  # noqa: BLE001
            skipped_rows.append(
                {
                    **asdict(bench),
                    "model_slug": model_slug,
                    "model_label": spec["label"],
                    "model_id": model_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    display_case_ids = choose_display_cases(case_rows, model_slug=model_slug, per_family=args.display_per_family)
    display_rows = [
        row
        for row in feature_rows
        if row["case_id"] in display_case_ids and int(row["display_rank_abs"]) <= args.display_features
    ]
    if display_rows:
        attach_feature_labels(
            W=W,
            row_mean=row_mean,
            encoder_w=encoder_w,
            encoder_b=encoder_b,
            tokenizer=tokenizer,
            display_rows=display_rows,
            top_tokens=args.label_top_tokens,
            chunk_size=args.label_chunk_size,
        )
    model_dir = out_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    write_csv(model_dir / "case_rows.csv", case_rows)
    write_csv(model_dir / "top_feature_rows.csv", feature_rows)
    write_csv(model_dir / "display_feature_rows.csv", display_rows)
    write_csv(model_dir / "skipped_rows.csv", skipped_rows)
    write_csv(model_dir / "token_audit_rows.csv", token_audit_rows)
    (model_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_slug": model_slug,
                "model_spec": {**spec, "checkpoint": str(checkpoint)},
                "n_cases": len(cases),
                "n_rows": len(case_rows),
                "n_skipped": len(skipped_rows),
                "sae_config": sae_config,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return case_rows, feature_rows, skipped_rows, display_rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    aggregate_rows: list[dict[str, object]],
    display_rows: list[dict[str, object]],
    skipped_rows: list[dict[str, object]],
) -> None:
    skipped_models = [row for row in skipped_rows if not row.get("case_id")]
    skipped_queries = [row for row in skipped_rows if row.get("case_id")]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Benchmark-Derived Query Suite\n\n")
        if skipped_models:
            f.write("Skipped model/cell notes:\n\n")
            for row in skipped_models:
                f.write(f"- `{row.get('model_slug')}`: {row.get('reason')}\n")
            f.write("\n")
        if skipped_queries:
            f.write(f"Skipped query rows: {len(skipped_queries)}. See `skipped_rows.csv` for details.\n\n")
        f.write("## Aggregate Metrics\n\n")
        f.write("| model | family | rows | skipped | median rho | sign | pass | target wins |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in aggregate_rows:
            f.write(
                f"| {row['model_slug']} | {row['family']} | {row['n_rows']} | {row['n_skipped']} | "
                f"{float(row['median_rho_floor0p5']):.3f} | {float(row['sign_agreement']):.3f} | "
                f"{float(row['accepted_rate_rho_floor0p5_lt_0p5']):.3f} | {float(row['target_side_win_rate']):.3f} |\n"
            )
        f.write("\n## Display Candidates\n\n")
        f.write("| model | family | case | exact | sparse | rho | top feature labels |\n")
        f.write("|---|---|---|---:|---:|---:|---|\n")
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in display_rows:
            grouped.setdefault((str(row["model_slug"]), str(row["case_id"])), []).append(row)
        for (model_slug, case_id), rows in sorted(grouped.items()):
            first = rows[0]
            labels = "; ".join(str(r.get("feature_top_tokens") or f"f{r['feature_id']}") for r in rows[:3])
            labels = labels.replace("|", "/")
            f.write(
                f"| {model_slug} | {first['family']} | {case_id} | {float(first['exact_score']):+.3f} | "
                f"{float(first['sparse_feature_sum']):+.3f} | {float(first['rho_floor0p5']):.3f} | {labels} |\n"
            )


def parse_counts(raw: str) -> dict[str, int]:
    out: dict[str, int] = {
        "squad2_answerable": 50,
        "squad2_unanswerable": 75,
        "hotpotqa_distractor": 75,
        "legalbench_contractnli": 50,
        "securityeval": 50,
    }
    if not raw:
        return out
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        out[key.strip()] = int(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--models", default="qwen0p8b,qwen2b,qwen9b")
    parser.add_argument("--counts", default="")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--display-features", type=int, default=5)
    parser.add_argument("--display-per-family", type=int, default=2)
    parser.add_argument("--label-top-tokens", type=int, default=4)
    parser.add_argument("--label-chunk-size", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--skip-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bank-tokenizer", default="Qwen/Qwen3.5-2B")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = parse_counts(args.counts)
    log(f"building bank with counts: {counts}")
    bank_tokenizer = AutoTokenizer.from_pretrained(args.bank_tokenizer, local_files_only=True)
    cases = build_bank(bank_tokenizer, counts)
    write_jsonl(args.out_dir / "benchmark_query_bank.jsonl", [asdict(case) for case in cases])

    all_case_rows: list[dict[str, object]] = []
    all_feature_rows: list[dict[str, object]] = []
    all_skipped_rows: list[dict[str, object]] = []
    all_display_rows: list[dict[str, object]] = []
    for model_slug in [m.strip() for m in args.models.split(",") if m.strip()]:
        if model_slug not in MODEL_SPECS:
            raise ValueError(f"unknown model slug {model_slug}; have {sorted(MODEL_SPECS)}")
        case_rows, feature_rows, skipped_rows, display_rows = evaluate_model(
            model_slug=model_slug,
            spec=MODEL_SPECS[model_slug],
            cases=cases,
            out_dir=args.out_dir,
            args=args,
        )
        all_case_rows.extend(case_rows)
        all_feature_rows.extend(feature_rows)
        all_skipped_rows.extend(skipped_rows)
        all_display_rows.extend(display_rows)

    aggregate_rows = aggregate(all_case_rows, all_skipped_rows)
    write_csv(args.out_dir / "case_rows.csv", all_case_rows)
    write_csv(args.out_dir / "top_feature_rows.csv", all_feature_rows)
    write_csv(args.out_dir / "display_feature_rows.csv", all_display_rows)
    write_csv(args.out_dir / "skipped_rows.csv", all_skipped_rows)
    write_csv(args.out_dir / "aggregate_by_model_family.csv", aggregate_rows)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "counts_requested": counts,
                "n_cases": len(cases),
                "models_requested": [m.strip() for m in args.models.split(",") if m.strip()],
                "denom_floor": DENOM_FLOOR,
                "args": vars(args) | {"out_dir": str(args.out_dir)},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_summary(args.out_dir / "summary.md", aggregate_rows, all_display_rows, all_skipped_rows)
    log(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
