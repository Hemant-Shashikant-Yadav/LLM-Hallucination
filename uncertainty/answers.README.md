# uncertainty/answers.py — reading answers

Reads the final answer of a response by its kind. A thin layer over the pipeline's
`app.answer_utils` (read-only), whose readers are already debugged on these benchmarks.

| Part | What it does |
|---|---|
| `read_yes_no` | The verdict a worked answer ends on (`… Answer: no`); the last one wins, where the pipeline's reader takes the first. |
| `read_answer(kind, text, question)` | The comparable key: final number or expression, option letter, yes/no; None for free text (compared by NLI instead) or an unreadable answer. |
| `cluster_keys` | Numbers and expressions that mean the same share one key (`0.5` = `1/2`). |
| `rule_correct` | Correctness by rules, as `evaluation/hard_eval.py` scores; None for a yes/no answer that never says yes or no (the judge reads it). |
| `brief` | A short form of an answer, for prompts that list several (P(True)). |
