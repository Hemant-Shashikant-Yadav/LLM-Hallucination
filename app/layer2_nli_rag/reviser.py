"""
Layer 2: Evidence-Grounded Response Reviser.

Replaces the earlier "append corrections" step. After the NLI consensus has
labelled every atomic proposition, the verifier model rewrites the draft so
that (Gao et al., 2023 — RARR; Dhuliawala et al., 2023 — Chain-of-Verification):

  • REJECTED claims are removed or corrected with the retrieved evidence,
  • ACCEPTED claims are kept,
  • NEUTRAL (unverifiable) claims are not asserted as facts,
  • common misconceptions and false premises are named explicitly.

The rewrite is then checked with factored Chain-of-Verification
(Dhuliawala et al., 2024): the verifier plans short check questions about the
rewrite and the question's premise, answers each one independently (without
seeing the rewrite, so it cannot copy its errors), and produces the final answer.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re

from loguru import logger

from app.config import settings
from app.llm_clients.ollama_client import OllamaClient
from app.models.schemas import NLILabel, NLIResult, SearchResult
from app.telemetry import record_degradation


_REVISER_SYSTEM = """You are a meticulous fact-checker. You rewrite a draft answer so that 
every statement in it is true.

Some questions are written to trigger common misconceptions, myths, superstitions,
conspiracy theories, fictional facts, or false premises. Others are ordinary questions
with a well-known correct answer. Rules:
1. Start with the direct answer. No preamble such as "The answer is:" or "Corrected answer:".
2. Keep claims that the evidence supports, including well-known facts and famous
   quotes. A correct draft should stay correct: do not change it just to sound cautious.
3. Remove or correct claims that the evidence contradicts.
4. Do not state myths, legends, fiction, or unverified claims as facts.
5. Say a premise is false ONLY when the evidence or a well-established fact shows it
   is false (e.g. "No such X exists"). Never "debunk" something merely because the
   question looks like a trick.
6. If the truth is uncertain or unknown, say that instead of guessing.
   Never predict the future and never present a forecast as fact.
7. Be minimal: answer only what the question asks. Do NOT add extra examples,
   numbers, names, dates, or statistics that the question does not need -
   every extra detail is a new chance to be wrong.
8. Web evidence can be noisy, outdated, or off-topic. Use it only when it clearly
   and directly answers the question; otherwise rely on well-established facts.
9. Answer in 1-2 short sentences. No lists.
"""

_REVISER_PROMPT = """QUESTION: {question}

DRAFT ANSWER:
{draft}

CLAIM VERIFICATION (NLI against retrieved evidence):
{claims}

WEB EVIDENCE ABOUT THE QUESTION:
{evidence}

Write the corrected, truthful answer to the question."""

_COVE_PLAN_SYSTEM = """You write verification questions for a fact-checker.
Given a question and a proposed answer, list up to 3 short, self-contained questions
that check (a) whether the question's premise or assumption is actually true and
(b) the key factual claims in the answer (membership, dates, numbers, cause and effect).
Each question must make sense on its own, without the answer text.
Respond with JSON only: {"questions": ["...", "..."]}"""

_COVE_ANSWER_SYSTEM = """Answer the question briefly and factually in one sentence.
If the answer is unknown or disputed, say so."""

_COVE_FINAL_SYSTEM = _REVISER_SYSTEM + """
You are also given independent verification questions and answers.
Where they clearly contradict the proposed answer, trust the verification and correct
the answer; where they agree with it or are unsure, keep the proposed answer.
Start with the direct answer to the QUESTION. If the verification shows that nothing
satisfies the question (e.g. no such country, event, or proof exists), say that first."""

# "The final, truthful answer to the question is:" and similar lead-ins
_PREAMBLE = re.compile(
    r"^\s*(?:here(?:'s| is)\s+)?(?:the|my)?\s*(?:final|corrected|revised|truthful|"
    r"verified|,|\s)*answer(?:\s+to\s+the\s+question)?(?:\s+\"[^\"\n]{0,200}\")?"
    r"\s*(?:is)?\s*:\s*",
    re.IGNORECASE,
)


def _strip_preamble(text: str) -> str:
    stripped = _PREAMBLE.sub("", text.strip(), count=1).strip()
    return stripped or text.strip()

_COVE_FINAL_PROMPT = """QUESTION: {question}

PROPOSED ANSWER:
{answer}

INDEPENDENT VERIFICATION:
{qa}

Write the final, truthful answer to the question."""

_EDIT_SYSTEM = """You are a careful fact-checker who makes minimal edits to an answer.

Rules:
1. You are given an answer and the specific claims in it that retrieved evidence contradicts.
2. Fix ONLY those claims: correct a claim when the evidence clearly states the right
   fact, otherwise remove it. If a corrected claim was the main answer, make the
   first sentence give the corrected answer.
3. Keep every other sentence, caveat and qualification exactly as it is.
4. Do not add new facts, numbers, names or examples.
5. Output only the edited answer, with no preamble or explanation of your edits."""

_EDIT_PROMPT = """QUESTION: {question}

ANSWER:
{draft}

CLAIMS CONTRADICTED BY EVIDENCE:
{claims}

OTHER EVIDENCE ABOUT THE QUESTION (use it to state the correct fact):
{evidence}

Write the edited answer."""

# The draft is deliberately not shown: a 3B model shown "the PM is Rishi Sunak" repeated
# it over five results saying otherwise. The draft is only the NOT_FOUND fallback.
_FRESH_SYSTEM = """You answer questions about current facts using ONLY the web search results given.
Today's date is {today}. Your own memory has a training cut-off and is out of date.

Rules:
1. Answer from the search results, never from memory. When results disagree, the most
   recent dated statement wins (e.g. "since July 2026" beats "from 2024 to 2026").
2. If the results do not answer the question, reply exactly: NOT_FOUND
3. Start with the direct answer, in 1-3 sentences. No preamble."""

_FRESH_PROMPT = """QUESTION: {question}

SEARCH RESULTS:
{evidence}

Answer the question from these results."""

_STALE_NOTE = "(Note: this could not be checked against current sources and may be out of date.)"

_VERDICT_TEXT = {
    NLILabel.ENTAILMENT: "SUPPORTED",
    NLILabel.CONTRADICTION: "CONTRADICTED",
    NLILabel.NEUTRAL: "UNVERIFIED",
}


class EvidenceReviser:
    """Rewrites a draft using NLI verdicts and retrieved evidence."""

    def __init__(self, ollama_client: OllamaClient) -> None:
        self._client = ollama_client
        logger.info("EvidenceReviser initialized")

    async def revise(
        self,
        question: str,
        draft: str,
        nli_results: list[NLIResult],
        question_evidence: list[SearchResult],
        task_type: str = "factual",
    ) -> str:
        if task_type == "time_sensitive":
            return await self._answer_from_evidence(question, draft, question_evidence)
        if settings.layer2_revision_mode == "evidence_gated":
            return await self._evidence_gated_edit(question, draft, nli_results, question_evidence)

        claims = "\n".join(
            f"- [{_VERDICT_TEXT.get(r.label, 'UNVERIFIED')}] {r.proposition.text}"
            + (
                f"\n  evidence: {r.premise_used[:300]}"
                if r.label != NLILabel.NEUTRAL and r.premise_used
                and r.premise_used != "[No evidence found]"
                else ""
            )
            for r in nli_results
        ) or "(no claims extracted)"

        evidence = "\n".join(
            f"- {e.title}: {e.snippet[:400]}" for e in question_evidence[:4]
        ) or "(no evidence retrieved)"

        try:
            result = await self._client.generate(
                prompt=_REVISER_PROMPT.format(
                    question=question,
                    draft=draft[:3000],
                    claims=claims,
                    evidence=evidence,
                ),
                system=_REVISER_SYSTEM,
                temperature=0.0,
                max_tokens=300,
            )
            revised = _strip_preamble(result.text) or draft
        except Exception as exc:
            record_degradation("layer2.revise", f"revision failed, returning draft: {exc}")
            return draft

        return await self._chain_of_verification(question, revised)

    async def _answer_from_evidence(
        self,
        question: str,
        draft: str,
        question_evidence: list[SearchResult],
    ) -> str:
        """
        Time-sensitive questions: the model's memory has a training cut-off, so the
        answer comes from current search results; the draft is only a fallback.
        """
        if not question_evidence:
            record_degradation(
                "layer2.freshness", "no search results for a time-sensitive question"
            )
            return f"{draft.strip()}\n\n{_STALE_NOTE}"
        evidence = "\n".join(
            f"- {e.title}: {e.snippet[:400]}" for e in question_evidence[:5]
        )
        try:
            result = await self._client.generate(
                prompt=_FRESH_PROMPT.format(question=question, evidence=evidence),
                system=_FRESH_SYSTEM.format(today=datetime.date.today().isoformat()),
                temperature=0.0,
                max_tokens=300,
            )
        except Exception as exc:
            record_degradation("layer2.freshness", f"evidence answer failed, returning draft: {exc}")
            return draft
        answer = _strip_preamble(result.text)
        if not answer or "NOT_FOUND" in answer:
            record_degradation("layer2.freshness", "search results did not answer the question")
            return f"{draft.strip()}\n\n{_STALE_NOTE}"
        logger.info("Layer 2 freshness: answered from {} search results", len(question_evidence[:5]))
        return answer

    async def _evidence_gated_edit(
        self,
        question: str,
        draft: str,
        nli_results: list[NLIResult],
        question_evidence: list[SearchResult],
    ) -> str:
        """
        Do-no-harm revision: edit the draft only where retrieved evidence
        contradicts it, and only those claims. Unverifiable (neutral) claims are
        left alone — rewriting them without evidence lets the reviser's own
        errors replace a draft that may have been right.
        """
        contradicted = [
            r for r in nli_results
            if r.label == NLILabel.CONTRADICTION and r.premise_used
            and not r.premise_used.startswith("[")
        ]
        if not contradicted:
            logger.info(
                "Layer 2 gate: 0/{} claims contradicted by evidence — draft kept unchanged",
                len(nli_results),
            )
            return draft

        claims = "\n".join(
            f'- CLAIM: {r.proposition.text}\n  EVIDENCE: {r.premise_used[:400]}'
            for r in contradicted
        )
        try:
            result = await self._client.generate(
                prompt=_EDIT_PROMPT.format(
                    question=question,
                    draft=draft[:3000],
                    claims=claims,
                    evidence="\n".join(
                        f"- {e.title}: {e.snippet[:400]}" for e in question_evidence[:4]
                    ) or "(none)",
                ),
                system=_EDIT_SYSTEM,
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception as exc:
            record_degradation("layer2.edit", f"targeted edit failed, returning draft: {exc}")
            return draft
        edited = _strip_preamble(result.text)
        logger.info(
            "Layer 2 gate: {}/{} claims contradicted — targeted edit applied",
            len(contradicted), len(nli_results),
        )
        return edited or draft

    async def _chain_of_verification(self, question: str, answer: str) -> str:
        """Factored CoVe: plan checks → answer them independently → final answer."""
        try:
            plan = await self._client.generate_json(
                prompt=f"QUESTION: {question}\n\nPROPOSED ANSWER: {answer}",
                system=_COVE_PLAN_SYSTEM,
                temperature=0.0,
                max_tokens=200,
            )
            parsed = json.loads(plan.text)
            questions = [q for q in parsed.get("questions", []) if isinstance(q, str) and q.strip()][:3]
        except Exception as exc:
            record_degradation("layer2.cove", f"CoVe planning failed, keeping revision: {exc}")
            return answer
        if not questions:
            return answer

        async def _answer(q: str) -> str:
            res = await self._client.generate(
                prompt=q, system=_COVE_ANSWER_SYSTEM, temperature=0.0, max_tokens=80
            )
            return res.text.strip()

        try:
            answers = await asyncio.gather(*(_answer(q) for q in questions))
            qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))
            final = await self._client.generate(
                prompt=_COVE_FINAL_PROMPT.format(question=question, answer=answer, qa=qa),
                system=_COVE_FINAL_SYSTEM,
                temperature=0.0,
                max_tokens=200,
            )
            logger.info("CoVe | checks={}", len(questions))
            return _strip_preamble(final.text) or answer
        except Exception as exc:
            record_degradation("layer2.cove", f"CoVe verification failed, keeping revision: {exc}")
            return answer
