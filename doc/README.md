# Documentation

A defence-in-depth pipeline that reduces hallucinations of a small local LLM. One model
(qwen2.5:3b) runs every step; the pipeline checks its first answer and only changes it when
there is evidence it is wrong. On 100 GSM8K test questions it scores 88% against the raw
model's 84%, and it wins on harder maths and logic sets. It does not yet help on
adversarial factual questions (TruthfulQA).

| Read | For |
|---|---|
| [01_overview.md](01_overview.md) | The problem, the design rules, the three layers at a glance |
| [02_pipeline.md](02_pipeline.md) | How a question flows through the system, step by step, with the algorithms |
| [03_evaluation.md](03_evaluation.md) | Datasets, scoring, statistics, and the commands to reproduce every number |
| [04_results.md](04_results.md) | All results, and what helped and what didn't |
| [05_history.md](05_history.md) | What was built, in order, and the bugs that changed the numbers |
| [06_roadmap.md](06_roadmap.md) | Open problems and next steps |

**Code-level detail:** every file in `app/` has an explanation beside it
(`app/router.py` → `app/router.README.md`), section by section with line numbers. Start
with [router.README.md](../app/router.README.md). The evaluation folder is described in
[evaluation/README.md](../evaluation/README.md); moving to new hardware, in
[HANDOFF.md](../HANDOFF.md).
