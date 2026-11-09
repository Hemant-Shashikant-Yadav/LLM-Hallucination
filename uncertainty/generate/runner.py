"""
Generation (PLAN §6): for each question, the greedy answer, K samples, P(True), verbalised
confidence and the self-diagnosis probes, stored as one record.

Generate once, score many: every uncertainty method reads these records offline. A set is
written line by line and resumed after an interruption (finished questions are skipped).
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from uncertainty.answers import brief
from uncertainty.generate.backends import Backend, Generation
from uncertainty.generate.prompts import PROBES, answer_prompt, p_true_prompt, probe_prompt, verbal_prompt
from uncertainty.roster import spec
from uncertainty.sets.load import Item
from uncertainty.settings import settings
from uncertainty.store import append_jsonl, finalize, gen_paths, load_generations, repair_partial


def max_tokens_for(kind: str) -> int:
    return {"short": settings.max_tokens_short, "choice": settings.max_tokens_choice,
            "yesno": settings.max_tokens_choice, "number": settings.max_tokens_number,
            "expr": settings.max_tokens_expr}[kind]


def greedy_options(logprob_mode: str) -> dict:
    """
    How to ask for the greedy answer. 'raw': T = 0 with log-probs. 'topk1': the server
    reports log-probs after temperature (all ≈ 0 at T = 0), so greedy decoding is done as
    T = 1 with top_k = 1 to keep the model's own probabilities. 'none': no log-probs.
    """
    if logprob_mode == "topk1":
        return {"temperature": 1.0, "top_k": 1, "logprobs": True, "top_logprobs": settings.top_logprobs}
    if logprob_mode == "none":
        return {"temperature": 0.0}
    return {"temperature": 0.0, "logprobs": True, "top_logprobs": settings.top_logprobs}


def binary_prob(gen: Generation, yes: tuple[str, ...], no: tuple[str, ...]) -> dict:
    """
    P(first option) from the first answer token's top log-probs: P(yes) / (P(yes) + P(no)).
    Falls back to the reply text (probability 1 or 0) when the server sent no log-probs.
    """
    norm = lambda t: re.sub(r"[\s().:*]", "", t).lower()
    yes_set, no_set = {norm(y) for y in yes}, {norm(n) for n in no}
    for alts in gen.top_logprobs:
        p_yes = sum(math.exp(lp) for t, lp in alts if norm(t) in yes_set)
        p_no = sum(math.exp(lp) for t, lp in alts if norm(t) in no_set)
        if p_yes + p_no > 0:
            return {"p": round(p_yes / (p_yes + p_no), 4), "source": "logprobs", "raw": gen.text[:20]}
    head = norm(gen.text[:12])
    if any(head.startswith(y) for y in yes_set):
        return {"p": 1.0, "source": "text", "raw": gen.text[:20]}
    if any(head.startswith(n) for n in no_set):
        return {"p": 0.0, "source": "text", "raw": gen.text[:20]}
    return {"p": None, "source": "unreadable", "raw": gen.text[:20]}


def read_verbal(text: str) -> float | None:
    m = re.search(r"\b(100|\d{1,2})(?:\.\d+)?\s*%?", text or "")
    return float(m.group(1)) if m else None


async def answer_record(backend: Backend, model: str, item: Item, *, k: int,
                        logprob_mode: str = "raw", probes: bool = True, self_eval: bool = True) -> dict:
    """Everything one question needs, in one record."""
    think = spec(model).think
    template_id, prompt = answer_prompt(item.kind, item.question)
    max_tokens = max_tokens_for(item.kind)
    wall = time.perf_counter()
    sample_lp = logprob_mode != "none"

    greedy_task = backend.generate(model, prompt, max_tokens=max_tokens, think=think, **greedy_options(logprob_mode))
    sample_tasks = [backend.generate(model, prompt, temperature=settings.sample_temperature, max_tokens=max_tokens,
                                     seed=settings.seed_base + i, logprobs=sample_lp, think=think)
                    for i in range(k)]
    greedy, *samples = await asyncio.gather(greedy_task, *sample_tasks)

    # Self-evaluation needs the answers, so it runs second
    ideas = [brief(item.kind, s.text, item.question) for s in samples if s.text.strip() and not s.truncated]
    proposed = brief(item.kind, greedy.text, item.question)
    skipped = Generation(text="", done_reason="skipped")
    calls = [
        backend.generate(model, p_true_prompt(item.question, ideas, proposed), temperature=0.0, max_tokens=3,
                         top_logprobs=10, think=think),
        backend.generate(model, verbal_prompt(item.question, proposed), temperature=0.0, max_tokens=8, think=think),
    ] if self_eval else []
    probe_names = list(PROBES) if probes else []
    calls += [backend.generate(model, probe_prompt(item.question, p), temperature=0.0, max_tokens=2,
                               top_logprobs=10, think=think) for p in probe_names]
    results = await asyncio.gather(*calls)
    p_true_gen, verbal_gen = (results[0], results[1]) if self_eval else (skipped, skipped)
    probe_gens = results[2:] if self_eval else results

    every = [greedy, *samples, *results]
    return {
        "id": item.id, "set": item.set, "split": item.split, "core": item.core, "kind": item.kind,
        "model": model, "backend": backend.name, "template": template_id, "logprob_mode": logprob_mode,
        "k": k, "question": item.question,
        "greedy": greedy.as_record(top=True),
        "samples": [{"seed": settings.seed_base + i, **s.as_record()} for i, s in enumerate(samples)],
        "p_true": binary_prob(p_true_gen, ("A",), ("B",)) if self_eval else {"p": None, "source": "skipped"},
        "verbal": {"value": read_verbal(verbal_gen.text) if self_eval else None, "raw": verbal_gen.text[:40]},
        "probes": {name: binary_prob(g, ("yes",), ("no",)) for name, g in zip(probe_names, probe_gens)},
        "cost": {
            "calls": len(every),
            "prompt_tokens": sum(g.prompt_tokens for g in every),
            "completion_tokens": sum(g.completion_tokens for g in every),
            "sample_tokens": sum(g.completion_tokens for g in samples),
            "self_eval_tokens": p_true_gen.completion_tokens + verbal_gen.completion_tokens
                                + p_true_gen.prompt_tokens + verbal_gen.prompt_tokens,
            "gen_seconds": round(sum(g.seconds for g in every), 2),
            "seconds": round(time.perf_counter() - wall, 2),
        },
    }


async def generate_set(backend: Backend, model: str, items: list[Item], results_dir: Path, *, k: int,
                       logprob_mode: str = "raw", check: Callable[[list[dict]], None] | None = None,
                       check_after: int = 20) -> dict:
    """
    Generate every question of one set that is not done yet; resumable.
    `check` runs once on the first `check_after` new records (a per-model gate) and may raise.
    """
    set_name = items[0].set
    partial, final = gen_paths(results_dir, model, set_name)
    repair_partial(partial)
    done = {r["id"] for r in load_generations(results_dir, model, set_name)}
    todo = [it for it in items if it.id not in done]
    if not todo:
        if partial.exists():
            finalize(partial, final, [it.id for it in items])
        return {"set": set_name, "new": 0, "total": len(items)}

    gate = asyncio.Semaphore(settings.item_concurrency)
    fresh: list[dict] = []
    checked = check is None
    start = time.perf_counter()
    tokens = 0

    async def one(item: Item) -> dict:
        async with gate:
            return await answer_record(backend, model, item, k=k, logprob_mode=logprob_mode)

    logger.info("{} | {}: {} to generate ({} done)", model, set_name, len(todo), len(done))
    for coro in asyncio.as_completed([one(it) for it in todo]):
        rec = await coro
        append_jsonl(partial, rec)
        fresh.append(rec)
        tokens += rec["cost"]["completion_tokens"]
        n = len(fresh)
        if not checked and (n >= check_after or n == len(todo)):
            checked = True
            check(fresh)
        if n % 10 == 0 or n == len(todo):
            elapsed = time.perf_counter() - start
            logger.info("{} | {}: {}/{} | {:.1f} s/question | {:.1f} tokens/s", model, set_name, n, len(todo),
                        elapsed / n, tokens / max(elapsed, 1e-9))
    finalize(partial, final, [it.id for it in items])
    elapsed = time.perf_counter() - start
    return {"set": set_name, "new": len(fresh), "total": len(items), "seconds": round(elapsed, 1),
            "completion_tokens": tokens, "tokens_per_s": round(tokens / max(elapsed, 1e-9), 2)}
