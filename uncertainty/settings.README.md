# uncertainty/settings.py — settings for the study

All tunable numbers in one place, read from `UQ_*` environment variables or an optional
`uncertainty/uq.env`. Separate from the pipeline's `app/config.py`, so `DID_` settings are
never read or changed here.

| Part | What it does |
|---|---|
| `PACKAGE_DIR`, `PROJECT_ROOT` | Paths everything else is relative to. |
| Ollama | `ollama_base_url`, `ollama_timeout` (900 s per call), `num_ctx` (4,096), `concurrency` (4 in-flight requests: match `OLLAMA_NUM_PARALLEL`), `item_concurrency` (2 questions at once), `max_retries`. |
| Generation | `k_samples` (10), `sample_temperature` (1.0), `top_logprobs` (5), token limits per answer kind (short 128, choice/yes-no 768, number 1,024, expression 1,536), `logprob_decimals` (4, keeps files small), `seed_base`. |
| Objective 2 | `intervention_samples` (5 per intervention). |
| NLI | `nli_model` (DeBERTa-v3, as Layer 1), `nli_batch`, `nli_device` (auto: CUDA, then Apple MPS, then CPU). |
| Data | `sets_dir` (`data/uq_sets`), `results_dir`, `fake_results_dir`, `split_seed` (7), `dev_fraction` (0.3). |
| Statistics | `bootstrap` (1,000 resamples), `ece_bins` (15). |
