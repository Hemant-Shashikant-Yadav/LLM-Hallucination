# uncertainty/generate/prompts.py — prompt templates

Every prompt the study sends to a model under test. The first line of each self-evaluation,
probe and intervention prompt is fixed text, so the simulated model can tell the call types
apart.

| Part | What it does |
|---|---|
| `ANSWER_TEMPLATES`, `answer_prompt` | Short answers: the semantic-entropy papers' one-sentence prompt. Worked answers end on `Final answer:`, `Answer: <letter>` or `Answer: yes/no`, so they can be read. |
| `with_evidence`, `with_hint` | The two Objective 2 interventions. |
| `p_true_prompt` | Kadavath et al.: the samples as brainstormed ideas, then A (True) / B (False). |
| `verbal_prompt` | A 0–100 confidence. |
| `PROBES`, `probe_prompt` | Five self-diagnosis questions answered Yes/No: ambiguous, time-dependent, needs calculation, false premise, unanswerable. |
