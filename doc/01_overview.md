# 1. Overview

## The problem

Small language models that run on a laptop answer confidently even when they are wrong:
they miscalculate word problems, repeat popular myths, and give out-of-date facts. Bigger
models help, but cost hardware. This project asks: **can a pipeline around one small model
make it more accurate, without using a bigger model anywhere?**

## Design rules

1. **One model in every role.** The same model (now `qwen2.5:3b`) writes the draft, the
   samples, the claim checks, the edits and the programs. Any gain comes from the pipeline,
   not a stronger helper. The model may be swapped; it may not be mixed.
2. **An independent judge for evaluation only.** Answers that need judging (TruthfulQA,
   yes/no) are graded by a *different* model (`llama3.1:8b`), the same way for raw and
   pipeline answers. A model grading itself shares its own misconceptions.
3. **Do no harm.** A layer may change the draft only with evidence it is wrong: evidence
   that contradicts a claim, a majority of samples, or an exact calculation. Ties and weak
   signals keep the draft.
4. **Pay only when unsure.** If the model's answers agree, the draft is returned at once
   (the fast path). Verification runs only when they disagree.
5. **Answer anything basic.** Beyond benchmarks, any simple web or maths question must
   work (16-question smoke test).
6. **Fair measurement.** "Raw" is the exact draft the pipeline started from (paired
   comparison); thresholds are tuned on dev items 500–539, results reported on test items.

## The three layers

```
question → draft answer (greedy)
        → task type: maths / factual / time-sensitive
        → LAYER 1  triage: sample 4 more answers, measure disagreement (semantic entropy)
              ├─ answers agree  → return the draft            (fast path, ~60% of GSM8K)
              ├─ multiple choice or yes/no → vote over the samples
              ├─ factual / current events → LAYER 2
              └─ maths → LAYER 3

LAYER 2  NLI-RAG (facts)
   split the draft into claims → web search per claim → an NLI model (DeBERTa-v3) weighs
   every passage → edit only the claims the evidence contradicts.
   Current-events questions are answered from today's search results instead.

LAYER 3  HalluClean (maths)
   exact calculator for plain arithmetic → else plan → 6–12 more samples → 5 Python
   programs run in a sandbox → vote: samples count 1, programs 0.5; ties keep the draft.
```

| Layer | Detects or fixes | Key files |
|---|---|---|
| Routing | Which checks a question needs | `app/answer_utils.py`, `app/layer1_triage/task_router.py` |
| 1 Triage | Whether the model is unsure | `app/layer1_triage/semantic_entropy.py` |
| 2 NLI-RAG | Factual claims contradicted by evidence; stale facts | `app/layer2_nli_rag/` |
| 3 HalluClean | Wrong reasoning and arithmetic | `app/layer3_halluclean/program_verifier.py` |
| Orchestration | The whole flow, audit trail | `app/router.py` |

## Technology

FastAPI (async API), Ollama (local LLM serving), Hugging Face Transformers (DeBERTa-v3 NLI),
sympy (safe symbolic maths), DuckDuckGo search (`ddgs`), Pydantic settings, loguru logging
with per-query trace IDs. Hardware so far: one GTX 1650 (4 GB).

## Research basis

Semantic entropy (Kuhn et al., 2023; Farquhar et al., 2024), NLI-based fact checking with
retrieval (RARR, Gao et al., 2023), Chain-of-Verification (Dhuliawala et al., 2023),
self-consistency (Wang et al., 2023), program-aided reasoning (PAL, Gao et al., 2023;
PROVE, Toh et al., 2024), and the finding that self-correction without outside feedback
hurts (Huang et al., 2024), which is why Layer 3 uses programs and votes rather than a
self-judge.
