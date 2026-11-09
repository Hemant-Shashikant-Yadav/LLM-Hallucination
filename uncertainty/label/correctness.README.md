# uncertainty/label/correctness.py — is each greedy answer right?

Rules for worked answers, the judge for free text, and a second judge from a third family
on a fixed subset (PLAN §7). Writes `labels.jsonl` and `labels_second.jsonl`; resumable.

| Part | What it does |
|---|---|
| `normalize`, `alias_match` | SQuAD-style normalisation; does the answer contain a gold alias (recorded beside the judge's verdict). |
| `grade_text` | Correctness of any answer text: rules, the yes/no judge, or the judge by the item's grading mode. Also grades intervention answers. |
| `label_model` | Labels every generated greedy answer; returns the judge's failure rate (a gate). |
| `_second_judge` | Re-grades a seeded subset of judge-graded answers, for the agreement check. |
