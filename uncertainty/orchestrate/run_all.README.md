# uncertainty/orchestrate/run_all.py — the one command

`python -m uncertainty run-all` (PLAN §1.1). Stage and model status in `results/run_state.json`;
re-running skips finished work.

| Part | What it does |
|---|---|
| `RunConfig` | Backend, results folder, roster override, pipeline model, tiers, time limit, stage range, dry-run limit, memory override. |
| `Run.stage_check` | Memory → roster and focus models; pull models and judges; Ollama version; log-prob mode; unit tests in a separate process. |
| `Run.stage_pilot` | 5 dev questions per set on the pipeline model through every later step (gates stop the run); speed → projected hours; trims for `--max-hours`. |
| `Run._hours`, `_project`, `_trim` | Projection from the pilot's tokens per question and speed (scaled by model size); levers: K = 6 for worked sets, third focus model to core, optional models dropped. |
| `Run.process_model` | Generate (gated) → unload → NLI → labels (gated) → scores → thresholds → interventions (focus models) → reports. A gate failure skips the model and is recorded; a skipped model is tried again on the next run. |
| `Run.reports` | Both reports and the summary; a report error is recorded, and generation is never lost. |
| `Run.stage_focus`, `stage_others`, `run` | Focus models first, then the rest smallest first. |
| `write_summary` | `reports/SUMMARY.md`: stages, models, log-prob mode, trims, headline results, every degradation. |
