# app/layer2_nli_rag/nli_verifier.py — does the evidence support the claim?

Runs a natural-language-inference model (DeBERTa-v3, `cross-encoder/nli-deberta-v3-base`)
that reads an evidence passage and a claim and says **entailment** (supports),
**contradiction** or **neutral**. It also serves Layer 1, which uses it to tell whether two
answers mean the same thing.

**Weighted consensus** (the Layer 2 contribution): instead of trusting the first passage,
every passage votes, weighted by its relevance `w`:

`S = Σ w · (P(entail) − P(contradict)) / Σ w`, with S from −1 to 1.
S ≥ 0.15 → claim **supported**; S ≤ −0.15 → **contradicted**; otherwise **neutral**
(thresholds from settings; the docstring's ±0.3 are the class defaults, overridden by the router).

| Lines | What it does |
|---|---|
| 1–28 | Docstring: the consensus formula and thresholds. |
| 62–70 | Label order for this model: 0 contradiction, 1 entailment, 2 neutral. |
| 72–105 | Constructor: model name, entailment confidence floor (0.65), accept/reject thresholds, device. |
| 111–133 | `load_model`: loads tokenizer and model in a worker thread (at server start). |
| 139–199 | **`verify_claim(premise, hypothesis)`**: one pair → label, confidence, all three probabilities. Entailment with confidence below 0.65 is downgraded to neutral. Inputs truncated to 512 tokens. |
| 201–203 | `is_loaded`. |
| 205–245 | **`score_pairs(pairs)`**: batched scoring (16 at a time) returning the three probabilities per pair. Used by Layer 1 meaning clustering. |
| 251–296 | **`_compute_consensus`**: the weighted formula above; no passages → neutral. |
| 298–305 | Verdict → label (accept → entailment, reject → contradiction). |
| 311–360 | **`verify_propositions(claims, evidence_map)`**: checks every claim in parallel; a failed check becomes neutral and is recorded as a degradation. |
| 362–452 | `_verify_proposition_consensus`: no evidence → neutral (`[No evidence found]`). Otherwise scores **every** passage (weight floor 0.01), keeps a per-passage breakdown, computes the consensus, and reports the label, confidence = \|S\|, and the most supportive passage. |
| 458–515 | **`aggregate_results`**: counts supported/contradicted/neutral claims. Also builds a crude revision (remove contradicted claims, append "Factual Corrections"), which the router immediately replaces with the reviser's evidence-gated edit. |
| 521–529 | `unload`: frees the model. |
