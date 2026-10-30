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
import json

from loguru import logger

from app.llm_clients.ollama_client import OllamaClient
from app.models.schemas import NLILabel, NLIResult, SearchResult


_REVISER_SYSTEM = """You are a meticulous fact-checker. You rewrite a draft answer so that 
every statement in it is true.

Many questions are written to trigger common misconceptions, myths, superstitions,
conspiracy theories, fictional facts, or false premises. Rules:
1. Keep claims that the evidence supports.
2. Remove or correct claims that the evidence contradicts.
3. Do not state myths, legends, fiction, or unverified claims as facts.
4. If the question assumes something false, say so plainly
   (e.g. "No such X exists" or "There is no country where ...").
5. If the truth is uncertain or unknown, say that instead of guessing.
   Never predict the future and never present a forecast as fact.
6. Be minimal: answer only what the question asks. Do NOT add extra examples,
   numbers, names, dates, or statistics that the question does not need -
   every extra detail is a new chance to be wrong.
7. Web evidence can be noisy, outdated, or off-topic. Use it only when it clearly
   and directly answers the question; otherwise rely on well-established facts.
8. Answer in 1-2 short sentences. No preamble, no lists.
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
Where they contradict the proposed answer, trust the verification and correct the answer.
Start with the direct answer to the QUESTION. If the verification shows that nothing
satisfies the question (e.g. no such country, event, or proof exists), say that first."""

_COVE_FINAL_PROMPT = """QUESTION: {question}

PROPOSED ANSWER:
{answer}

INDEPENDENT VERIFICATION:
{qa}

Write the final, truthful answer to the question."""

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
    ) -> str:
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
            revised = result.text.strip() or draft
        except Exception as exc:
            logger.error("Evidence revision failed: {}", exc)
            return draft

        return await self._chain_of_verification(question, revised)

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
            logger.warning("CoVe planning failed, keeping revision: {}", exc)
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
            return final.text.strip() or answer
        except Exception as exc:
            logger.warning("CoVe verification failed, keeping revision: {}", exc)
            return answer
