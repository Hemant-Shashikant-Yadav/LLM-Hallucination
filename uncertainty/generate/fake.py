"""
A simulated model with planted uncertainty types, for dry runs and tests (PLAN §1).

Every (model, question) pair gets a hidden type: certain-right, certain-wrong (a confident
hallucination), knowledge gap, reasoning slip, ambiguous, unanswerable, false premise or
temporal. The simulated answers, samples, log-probs, self-evaluations, probe replies,
intervention replies and judge verdicts all follow from that type, the way the plan
expects a real model to behave. Everything is deterministic: the same call gives the same
reply. The Objective 2 tests check that the classifier recovers the planted types.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re

from app.answer_utils import parse_options
from uncertainty.generate.backends import Backend, Generation
from uncertainty.generate.prompts import EVIDENCE_HEAD, HINT_HEAD, P_TRUE_TAIL, PROBES, VERBAL_TAIL
from uncertainty.roster import spec
from uncertainty.sets.load import Item

PLANTED = ("certain_right", "certain_wrong", "knowledge", "reasoning",
           "ambiguous", "unanswerable", "false_premise", "temporal")
# Planted type → the Objective 2 type it should be classified as
PLANTED_TO_TYPE = {"knowledge": "knowledge", "reasoning": "reasoning", "ambiguous": "aleatoric",
                   "unanswerable": "aleatoric", "false_premise": "aleatoric", "temporal": "temporal"}

_NAMES = ["Halvorsen", "Mirabel", "Castellan", "Ostrova", "Penwick", "Tarrant", "Velasquez", "Lindqvist",
          "Ashdown", "Beaumont", "Kerrigan", "Moravec"]
_ABSTAIN = "It is impossible to know for sure"
_SHORT_TEMPLATES = ["The answer is {a}.", "I believe the answer is {a}.", "Given {e}, the answer is {a}."]


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


class FakeBackend(Backend):
    name = "fake"

    _KEY = 24

    def __init__(self, items: list[Item]) -> None:
        self._prefix: dict[str, list[Item]] = {}
        self._short: list[Item] = []
        for it in items:
            if len(it.question) < self._KEY:
                self._short.append(it)
            else:
                self._prefix.setdefault(it.question[:self._KEY], []).append(it)
        self._cache: dict[str, Item | None] = {}
        self._loaded: set[str] = set()

    # ------------------------------------------------------------------ plumbing

    async def version(self) -> str:
        return "fake-1"

    async def available(self) -> list[str]:
        return ["*"]

    async def loaded(self) -> list[str]:
        return sorted(self._loaded)

    async def pull(self, model: str) -> None:
        return None

    async def unload(self, model: str) -> None:
        self._loaded.discard(model)

    def find_item(self, prompt: str) -> Item | None:
        if prompt in self._cache:
            return self._cache[prompt]
        starts = {0} | {m.end() for m in re.finditer(r"Question: |QUESTION: |\n\n", prompt)}
        matches = [it for s in starts for it in self._prefix.get(prompt[s:s + self._KEY], []) + self._short
                   if it.question in prompt]
        found = max(matches, key=lambda it: len(it.question)) if matches else None
        self._cache[prompt] = found
        return found

    # ------------------------------------------------------------------ the simulated model

    @staticmethod
    def planted(model: str, item: Item) -> str:
        """The hidden type of this model on this question (deterministic)."""
        r = _rng("type", model, item.id)
        if item.gold_type in ("ambiguous", "unanswerable", "false_premise", "temporal"):
            return item.gold_type if r.random() < 0.85 else "certain_right"
        skill = min(0.88, 0.5 + 0.03 * (spec(model).size_b or 4))
        x = r.random()
        if x < skill * 0.85:
            return "certain_right"
        if x < skill * 0.85 + 0.08:
            return "certain_wrong"
        p_reason = {"number": 0.8, "expr": 0.8, "choice": 0.5, "yesno": 0.5}.get(item.kind, 0.15)
        return "reasoning" if r.random() < p_reason else "knowledge"

    @staticmethod
    def _gold(item: Item) -> str:
        if item.judge_mode == "abstain":
            return _ABSTAIN
        return item.answers[0] if item.answers else "unknown"

    @staticmethod
    def _wrongs(item: Item) -> list[str]:
        r = _rng("wrong", item.id)
        gold = FakeBackend._gold(item)
        if item.kind == "choice":
            return [k for k in sorted(parse_options(item.question)) if k != gold] or ["a"]
        if item.kind == "yesno":
            return ["no" if gold == "yes" else "yes"]
        if item.kind in ("number", "expr"):
            try:
                g = float(gold)
                return [f"{g + d:g}" for d in (r.choice([1, 2, 10]), -r.choice([1, 3]), r.choice([5, 7]))]
            except ValueError:
                return [f"{gold}+{d}" for d in (1, 2, 3)]
        if item.incorrect_answers:
            return list(item.incorrect_answers[:3])
        names = r.sample(_NAMES, 3)
        return [f"{n}" for n in names]

    def _distribution(self, planted: str, item: Item, mode: str) -> list[tuple[str, float]]:
        """Answer distribution of a sample: [(answer, probability)], by planted type and prompt mode."""
        gold, wrong = self._gold(item), self._wrongs(item)
        w2 = wrong + wrong[:1] * 3
        second = item.answers[1] if len(item.answers) > 1 else wrong[0]
        if mode == "evidence":
            if planted in ("knowledge", "temporal"):
                return [(gold, 0.92), (w2[0], 0.08)]
            if planted == "certain_wrong":
                return [(gold, 0.6), (w2[0], 0.4)]
        if mode == "hint":
            if planted == "reasoning":
                return [(gold, 0.92), (w2[0], 0.08)]
            if planted == "certain_wrong" and item.structured:
                return [(gold, 0.5), (w2[0], 0.5)]
        return {
            "certain_right": [(gold, 0.95), (w2[0], 0.05)],
            "certain_wrong": [(w2[0], 0.93), (gold, 0.07)],
            "knowledge": [(gold, 0.3), (w2[0], 0.3), (w2[1], 0.2), (w2[2], 0.2)],
            "reasoning": [(gold, 0.4), (w2[0], 0.35), (w2[1], 0.25)],
            "ambiguous": [(gold, 0.5), (second, 0.5)],
            "unanswerable": [(_ABSTAIN, 0.35), (w2[0], 0.35), (w2[1], 0.3)],
            "false_premise": [(gold, 0.45), (w2[0], 0.55)],
            "temporal": [(gold, 0.45), (w2[0], 0.55)],
        }[planted]

    def _greedy_answer(self, planted: str, item: Item, mode: str) -> str:
        dist = self._distribution(planted, item, mode)
        if planted in ("knowledge", "reasoning", "unanswerable", "false_premise", "temporal") and mode == "plain":
            # The greedy draft is right only some of the time
            r = _rng("greedy", item.id, planted)
            return dist[0][0] if r.random() < 0.4 else dist[1][0]
        return max(dist, key=lambda p: p[1])[0]

    def greedy_correct(self, model: str, item: Item) -> bool:
        return self._greedy_answer(self.planted(model, item), item, "plain") == self._gold(item)

    def _premise(self, planted: str, item: Item, seed: int | None) -> tuple[str, str]:
        """Premise words a worked answer mentions: shared when only the conclusion is unstable."""
        stable = planted in ("certain_right", "certain_wrong", "reasoning", "ambiguous", "temporal")
        r = _rng("premise", item.id, "stable" if stable else seed)
        if item.kind in ("number", "expr"):
            return str(r.randint(2, 99)), str(r.randint(2, 99))
        return r.choice(_NAMES), r.choice(_NAMES)

    def _render(self, item: Item, answer: str, premise: tuple[str, str], r: random.Random,
                truncate: bool) -> str:
        p1, p2 = premise
        if item.kind in ("number", "expr"):
            text = (f"Step 1: The problem gives {p1} and {p2}.\nStep 2: Combining these quantities "
                    f"carefully.\nFinal answer: {answer}")
        elif item.kind == "choice":
            text = f"Considering {p1} and {p2}, the best option is clear.\nAnswer: {answer.upper()}"
        elif item.kind == "yesno":
            text = f"Thinking about {p1} and {p2}, the conclusion follows.\nAnswer: {answer}"
        else:
            text = r.choice(_SHORT_TEMPLATES).format(a=answer, e=p1)
        if truncate:
            text = text[: max(10, len(text) // 2)]
        return text

    def _logprob_level(self, planted: str, mode: str) -> float:
        if mode != "plain" and planted in ("knowledge", "reasoning", "temporal"):
            return -0.15
        return {"certain_right": -0.08, "certain_wrong": -0.12, "knowledge": -1.4, "reasoning": -0.7,
                "ambiguous": -0.18, "unanswerable": -0.9, "false_premise": -0.5, "temporal": -0.35}[planted]

    @staticmethod
    def _tokens(text: str, level: float, r: random.Random, top: int) -> tuple[list, list, list]:
        tokens = re.findall(r"\s*\S+", text)
        lps, tops = [], []
        for _ in tokens:
            lp = min(-1e-4, r.gauss(level, abs(level) * 0.35 + 0.02))
            lps.append(lp)
            if top:
                rest = max(1e-6, 1 - math.exp(lp))
                weights = [0.5, 0.25, 0.15, 0.1][: top - 1]
                tops.append([["<tok>", lp]] + [[f"<alt{i}>", math.log(rest * w)] for i, w in enumerate(weights)])
        return tokens, lps, tops

    def _one_token(self, token: str, p: float, other: str) -> Generation:
        p = min(max(p, 1e-4), 1 - 1e-4)
        lp_tok, lp_other = math.log(p), math.log(1 - p)
        first, second = (token, lp_tok), (other, lp_other)
        top = sorted([first, second], key=lambda t: -t[1])
        chosen = top[0]
        return Generation(text=chosen[0], tokens=[chosen[0]], logprobs=[chosen[1]],
                          top_logprobs=[[list(t) for t in top]], prompt_tokens=60, completion_tokens=1,
                          seconds=0.0)

    # ------------------------------------------------------------------ calls

    async def generate(self, model: str, prompt: str, *, system: str = "", temperature: float = 0.0,
                       max_tokens: int = 256, seed: int | None = None, logprobs: bool = False,
                       top_logprobs: int = 0, fmt: str | None = None, top_k: int | None = None,
                       think: bool | None = None) -> Generation:
        self._loaded = {model}
        if fmt == "json" or "REFERENCE ANSWERS" in prompt or "YES or NO" in system:
            return self._judge(model, prompt, system)
        item = self.find_item(prompt)
        if item is None:
            # Unknown prompt (e.g. the check stage's log-prob probe): a plain confident reply
            r = _rng("free", model, prompt, seed)
            text = "Paris is the capital of France."
            tokens, lps, tops = self._tokens(text, -0.3 if temperature else -0.05, r, top_logprobs)
            return Generation(text, tokens, lps, tops, "stop", 20, len(tokens))
        planted = self.planted(model, item)
        if P_TRUE_TAIL in prompt:
            return self._p_true(model, item, planted)
        if VERBAL_TAIL in prompt:
            g = self._p_true(model, item, planted)
            p = math.exp(g.top_logprobs[0][0][1]) if g.text == "A" else 1 - math.exp(g.top_logprobs[0][0][1])
            value = round(100 * min(1.0, p + 0.12))
            return Generation(str(value), [str(value)], [-0.3], [], "stop", 50, 2)
        for name, text in PROBES.items():
            if text in prompt:
                return self._probe(model, item, planted, name)
        mode = "evidence" if prompt.startswith(EVIDENCE_HEAD) else "hint" if prompt.startswith(HINT_HEAD) else "plain"
        r = _rng("sample", model, item.id, mode, seed, temperature)
        if temperature == 0:
            answer = self._greedy_answer(planted, item, mode)
        else:
            dist = self._distribution(planted, item, mode)
            answer = r.choices([a for a, _ in dist], weights=[p for _, p in dist])[0]
        truncate = item.structured and temperature > 0 and r.random() < 0.02
        text = self._render(item, answer, self._premise(planted, item, seed), r, truncate)
        tokens, lps, tops = self._tokens(text, self._logprob_level(planted, mode), r, top_logprobs)
        return Generation(text, tokens, lps, tops, "length" if truncate else "stop", 80, len(tokens))

    def _p_true(self, model: str, item: Item, planted: str) -> Generation:
        r = _rng("ptrue", model, item.id)
        right = self.greedy_correct(model, item)
        logit = (1.6 if right else -0.6) + (1.4 if planted == "certain_wrong" else 0.0) + r.gauss(0, 0.7)
        return self._one_token("A", 1 / (1 + math.exp(-logit)), "B")

    def _probe(self, model: str, item: Item, planted: str, probe: str) -> Generation:
        r = _rng("probe", model, item.id, probe)
        base = {
            "ambiguous": 0.8 if planted == "ambiguous" else 0.15,
            "time_dependent": 0.8 if planted == "temporal" else 0.12,
            "calculation": {"number": 0.85, "expr": 0.85, "choice": 0.45, "yesno": 0.4}.get(item.kind, 0.1),
            "false_premise": 0.75 if planted == "false_premise" else 0.12,
            "unanswerable": 0.75 if planted == "unanswerable" else 0.15,
        }[probe]
        return self._one_token("Yes", min(0.98, max(0.02, base + r.gauss(0, 0.1))), "No")

    def _judge(self, model: str, prompt: str, system: str) -> Generation:
        r = _rng("judge", model, prompt)
        if r.random() < 0.003:
            return Generation("not json", ["not"], [-1.0], [], "stop", 200, 2)
        if "YES or NO" in system:
            m = re.search(r"Answer: (yes|no)", prompt)
            return Generation(json.dumps({"verdict": m.group(1) if m else "unclear"}), [], [], [], "stop", 200, 8)
        section = lambda head: (prompt.split(head, 1)[1] if head in prompt else "").strip()
        answer = section("ANSWER TO GRADE:") or section("ANSWER TO EVALUATE:")
        answer = answer.split("\n\nReturn the JSON verdict.")[0].lower()
        if "CORRECT REFERENCE ANSWERS" in prompt:
            refs = section("CORRECT REFERENCE ANSWERS:").split("INCORRECT REFERENCE ANSWERS")[0]
        else:
            refs = section("REFERENCE ANSWERS:").split("ANSWER TO GRADE:")[0]
        refs = [ln.strip("- ").strip().lower() for ln in refs.splitlines() if ln.strip("- ").strip()]
        hit = any(ref and ref in answer for ref in refs) or (
            "no single true answer" in system.lower() and _ABSTAIN.lower() in answer)
        if "CORRECT REFERENCE ANSWERS" in prompt:
            verdict = {"truthful": hit, "informative": True, "reason": "fake"}
        else:
            verdict = {"correct": hit}
        return Generation(json.dumps(verdict), [], [], [], "stop", 200, 10)
