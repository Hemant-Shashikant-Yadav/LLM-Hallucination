"""
Prompt templates (PLAN §6). Each record stores the template id it was generated with.

The first line of every self-evaluation, probe and intervention prompt is fixed text, so the
simulated model (generate/fake.py) can tell the call types apart the same way a reader can.
"""

from __future__ import annotations

ANSWER_TEMPLATES: dict[str, tuple[str, str]] = {
    # The semantic-entropy papers' prompt: short answers keep NLI clustering and judging reliable
    "short": ("short_v1", "Answer the following question in a single brief but complete sentence.\n"
                          "Question: {question}\nAnswer:"),
    "number": ("worked_v1", "{question}\n\nSolve the problem step by step. End your response with a line "
                            "of the form 'Final answer: <answer>'."),
    "expr": ("worked_v1", "{question}\n\nSolve the problem step by step. End your response with a line "
                          "of the form 'Final answer: <answer>'."),
    "choice": ("choice_v1", "{question}\n\nThink it through briefly, then end your response with a line "
                            "of the form 'Answer: <letter>'."),
    "yesno": ("yesno_v1", "{question}\n\nThink it through briefly, then end your response with "
                          "'Answer: yes' or 'Answer: no'."),
}

EVIDENCE_HEAD = "Use the following information to answer."
HINT_HEAD = "Here is the beginning of a solution method. It does not give the final answer."

P_TRUE_TAIL = "Is the possible answer:\n(A) True\n(B) False\nReply with the single letter A or B."
P_TRUE = "Question: {question}\nHere are some brainstormed ideas:\n{ideas}\nPossible answer: {answer}\n" + P_TRUE_TAIL

VERBAL_TAIL = ("How confident are you that the proposed answer is correct? "
               "Reply with a number from 0 to 100 and nothing else.")
VERBAL = "Question: {question}\nProposed answer: {answer}\n" + VERBAL_TAIL

# Self-diagnosis probes for Objective 2 (PLAN §6 step 5): one token each, P(Yes) read from log-probs
PROBES: dict[str, str] = {
    "ambiguous": "Is this question ambiguous, so that it could reasonably be read in more than one way "
                 "with different correct answers?",
    "time_dependent": "Does the correct answer to this question change over time, so that it depends on "
                      "today's date?",
    "calculation": "Does answering this question require a calculation or a multi-step derivation?",
    "false_premise": "Does this question rest on a false assumption or premise?",
    "unanswerable": "Is this question impossible to answer with certainty, because the answer is unknown, "
                    "unknowable or a matter of opinion?",
}
PROBE_TAIL = "Reply with Yes or No only."
PROBE = "Question: {question}\n\n{probe} " + PROBE_TAIL


def answer_prompt(kind: str, question: str) -> tuple[str, str]:
    """(template id, prompt) for answering a question of this kind."""
    template_id, template = ANSWER_TEMPLATES[kind]
    return template_id, template.format(question=question)


def with_evidence(prompt: str, evidence: str) -> str:
    return f"{EVIDENCE_HEAD}\nInformation: {evidence}\n\n{prompt}"


def with_hint(prompt: str, hint: str) -> str:
    return f"{HINT_HEAD}\n{hint}\n\n{prompt}"


def p_true_prompt(question: str, ideas: list[str], answer: str) -> str:
    listed = "\n".join(f"- {i}" for i in ideas) or "- (none)"
    return P_TRUE.format(question=question, ideas=listed, answer=answer)


def verbal_prompt(question: str, answer: str) -> str:
    return VERBAL.format(question=question, answer=answer)


def probe_prompt(question: str, probe: str) -> str:
    return PROBE.format(question=question, probe=PROBES[probe])
