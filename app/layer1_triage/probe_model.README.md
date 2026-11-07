# app/layer1_triage/probe_model.py — legacy Layer 1: TinyLlama probe

> **Not used in the current setup.** Layer 1 runs in `semantic_entropy` mode
> (`semantic_entropy.py`). This file is used only with `DID_TRIAGE_MODE=surrogate_probe`,
> and its classifier gives meaningful output only with trained weights
> (`DID_PROBE_WEIGHTS_PATH`); none have been trained. Kept for the Phase 1 thesis design.

**Idea (Phase 1):** Ollama doesn't expose a model's internal states, so a small open model
(TinyLlama-1.1B) reads the question plus the draft, and a two-headed linear classifier on its
final hidden state predicts (1) how uncertain the answer is and (2) whether the risk is
about knowledge (→ Layer 2) or reasoning (→ Layer 3).

| Lines | What it does |
|---|---|
| 1–22 | Docstring: surrogate probing and the dual-head classifier. |
| 50–109 | **`DualHeadUncertaintyClassifier`**: Head 1 `H_sem = sigmoid(W₁h + b₁)`; Head 2 `logits = W₂h + b₂` over {knowledge, reasoning}. `forward` moves the input to the classifier's device and type and returns both. |
| 117–156 | **`SurrogateExtractor`**: holds TinyLlama, its tokenizer and hidden size. |
| 162–209 | Loads TinyLlama in a worker thread (half precision on CUDA) and reads its hidden size from the config. |
| 215–238 | `_get_final_layer_module`: finds the last transformer layer across common architectures. |
| 240–253 | `_hook_fn`: forward hook that keeps the **last token's** hidden state of the final layer. |
| 259–327 | `extract_hidden_state` / `_extract_sync`: formats `<|user|> question <|assistant|> draft`, runs one forward pass with the hook attached, returns a `[hidden_dim]` vector. |
| 333–345 | `unload`: frees the model and GPU memory. |
| 353–380 | **`TriageEngine`**: threshold, optional weights path, classifier, surrogate. |
| 386–419 | `initialize_probe`: builds the classifier; loads trained weights if present, else marks it untrained (proxy mode). `set_surrogate` attaches TinyLlama. |
| 425–479 | `triage`: runs the synchronous triage in a worker thread and logs the result. |
| 481–614 | `_triage_sync`, three paths: **A** TinyLlama hidden state → classifier → decision; **B** legacy activations from `activation_hook.py` → classifier (only if trained and sizes match); **C** fallback: a hallucination probability from token log-probabilities (0.5 when none) → entropy → keyword heuristic. |
| 620–634 | `_apply_threshold_from_classifier`: below `t_safe` → SAFE, else by Head 2's predicted type. |
| 636–665 | `_apply_threshold_heuristic`: counts reasoning words (`therefore`, `=`, `solve`…); 3+ → reasoning. |
| 671–686 | `_compute_semantic_entropy`: binary entropy of the hallucination probability. |
| 688–707 | `_estimate_from_logprobs`: mean log-probability mapped through a sigmoid centred at −2. |
| 713–836 | `train_probe`: trains both heads (binary cross-entropy for entropy, cross-entropy for type; weighted 0.5/0.5) with Adam, logging every 20 epochs; returns final losses and accuracy. |
| 838–845 | `save_probe`: writes trained weights. |
