# uncertainty/sets/load.py — question sets: schema, tiers, split

The `Item` every part of the package passes around, and reading the frozen files.

| Part | What it does |
|---|---|
| `SETS` | The 11 sets in run order, with domain and answer kind. |
| `KINDS`, `STRUCTURED`, `ALEATORIC_SUBTYPES` | Answer kinds; kinds read by rules (vs free text); metadata uncertainty subtypes. |
| `Item` | id, set, kind, question (options included), gold answers, dev/test split, core flag, how a judge grades it (`match`, `truthfulqa`, `abstain`, `premise`), wrong reference answers, metadata uncertainty type, oracle evidence and reasoning hint for Objective 2, metadata. |
| `load_set`, `load_all` | Items of a set or all sets at a tier; `limit` keeps n per set (dry runs). |
| `spread` | n items evenly spaced through a set, so a dry run keeps every stratum (FreshQA lists all false premises first). |
| `write_set` | Used by the builder. |
