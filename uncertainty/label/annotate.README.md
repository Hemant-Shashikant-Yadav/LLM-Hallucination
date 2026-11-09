# uncertainty/label/annotate.py — human labels and the judge audit

The one manual step, after the Mac run, on either machine.

| Part | What it does |
|---|---|
| `GUIDE` | The decision tree written next to the CSV. |
| `export` | ~300 in-scope answers spread over sets and models; no prediction shown (no bias). |
| `import_labels` | Validates types and subtypes, writes `results/human_labels.csv` (one annotator per file). |
| `export_audit`, `import_audit` | 100 judge-graded answers, blind; agreement of the judge with the human. |
