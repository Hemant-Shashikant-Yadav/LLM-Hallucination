# uncertainty/methods/nli.py — meaning equivalence for free text

DeBERTa-v3 NLI (the model Layer 1 uses): one matrix per free-text question over the greedy
answer and its valid samples, each prefixed with the question (Kuhn et al., 2023).

| Part | What it does |
|---|---|
| `valid_samples` | Samples that count: not empty, not cut off. |
| `nli_texts` | Question + the answer's first sentences, as Layer 1. |
| `NLIScorer` | Batched cross-encoder, label order read from the model config; CUDA, MPS or CPU, falling back to the CPU when a GPU back end fails a test pair. |
| `FakeNLI` | For dry runs: same meaning when the simulated answers match. |
| `nli_matrix` | All ordered pairs (110 at K = 10). |
| `save_matrices`, `load_matrices`, `build_set_matrices` | One compressed `.npz` per set; only missing questions are computed. |
