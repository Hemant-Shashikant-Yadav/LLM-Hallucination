# uncertainty/label/interventions.py — Objective 2 interventions

The type of uncertainty is whichever intervention removes it (PLAN §9.3).

| Part | What it does |
|---|---|
| `candidates` | Uncertain or wrong answers that have evidence or a hint and no metadata type. |
| `generate_interventions` | Pass 1, model under test: 5 samples with oracle evidence and 5 with the reasoning hint. |
| `_majority` | Largest meaning cluster, its share, normalised entropy. |
| `resolve_interventions` | Pass 2, NLI and judge: resolved = majority answer right and samples agree (entropy at the 'sure' threshold, or at most one dissent); label = knowledge, reasoning, mixed, unresolved. |
