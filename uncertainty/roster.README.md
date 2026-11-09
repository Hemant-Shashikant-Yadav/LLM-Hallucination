# uncertainty/roster.py — which models, which judge

The models measured (PLAN §4): several families, a size ladder inside each, and old vs new
generations. The roster is picked from the Mac's memory; `--models` overrides it.

| Part | What it does |
|---|---|
| `ModelSpec` | Tag, family, size, generation, memory needed, `think` flag (False for thinking models), `optional` (dropped first when `--max-hours` trims). `slug` is the folder name. |
| `ROSTER` | Twelve models; the 12–14B ones need 24 GB and are optional. |
| `PIPELINE_MODEL`, `FOCUS_DEFAULT` | `qwen2.5:3b` first; focus models get the extended set and the interventions. |
| `spec(tag)` | Roster entry, or family and size guessed from an unknown tag's name. |
| `pick_roster(memory_gb)` | Every model that fits. |
| `focus_models(tags, pipeline_model)` | Focus models in run order; a new pipeline model replaces the third. |
| `judge_for`, `second_judge_for` | Never the model under test, always another family: `llama3.1:8b` judges non-Llama models, a Qwen judges Llama models; the second judge (agreement check) comes from a third family. |
| `judges_needed` | Every judge the run will pull. |
