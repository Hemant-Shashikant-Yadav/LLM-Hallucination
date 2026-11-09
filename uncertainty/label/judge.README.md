# uncertainty/label/judge.py — the LLM judge

Grades free-text answers (PLAN §7). Never the model under test and always another family.
Verdicts are cached on disk by (judge, prompt).

| Part | What it does |
|---|---|
| `MATCH_SYSTEM`, `ABSTAIN_SYSTEM`, `PREMISE_SYSTEM`, `GRADE_PROMPT` | Correct = gives a reference answer; says the question cannot be answered; rejects the false premise. |
| TruthfulQA | The exact prompt of `evaluation/truthfulqa_judge.py` (read-only), so labels mean the same as in Objectives 3–4. |
| `YESNO_SYSTEM` | Which way a yes/no answer goes, when it never says yes or no (as `evaluation/hard_eval.py`). |
| `Judge._ask` | JSON reply, three attempts with different seeds, counts failures; saves the cache every 50 verdicts. |
| `Judge.grade`, `Judge.yes_no` | One answer's verdict. |
| `gather_chunked` | Runs calls in chunks of 64. |
