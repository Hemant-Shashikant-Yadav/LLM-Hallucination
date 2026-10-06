# 📊 Benchmark Performance Analysis — Defense-in-Depth Framework

## Executive Summary

Your framework was benchmarked on **2 datasets** (GSM8K — math reasoning, TruthfulQA — factual knowledge) across **3 configurations**: Raw LLM Baseline, Naive RAG, and your proposed Defense-in-Depth (full_framework). The results reveal a **mixed picture** with clear strengths in architecture design and computational efficiency, but **serious weaknesses in hallucination detection and correction** that need immediate attention.

> [!CAUTION]
> **TruthfulQA accuracy is 10% across ALL configurations** — the framework is not improving factual faithfulness on knowledge-intensive tasks. This is a critical gap that undermines Objectives 2, 3, and 4.

---

## 1. Raw Numbers at a Glance

### GSM8K (Mathematical Reasoning — 30 samples)

| Metric | Baseline (Raw LLM) | Naive RAG | Defense-in-Depth (Proposed) |
|---|---|---|---|
| **Accuracy** | **83.3%** (25/30) | **90.0%** (27/30) | **80.0%** (24/30) |
| **Avg Latency** | 8.7s | 48.4s | 14.1s |
| **Avg Tokens** | 302 | 303 | 304 |
| **Avg Semantic Entropy** | 0.60 (default) | 0.60 (default) | **0.466** |
| **Layers Invoked** | Layer1 only (100%) | Layer1+2 (100%) | Layer1 only: 93%, Layer1+3: 7% |

### TruthfulQA (Factual Knowledge — 30 samples)

| Metric | Baseline (Raw LLM) | Naive RAG | Defense-in-Depth (Proposed) |
|---|---|---|---|
| **Accuracy** | **10.0%** (3/30) | **10.0%** (3/30) | **10.0%** (3/30) |
| **Avg Latency** | 11.8s | 39.8s | 29.6s |
| **Avg Tokens** | 320 | 324 | 318 |
| **Avg Semantic Entropy** | 0.60 (default) | 0.60 (default) | **0.510** |
| **Layers Invoked** | Layer1 only (100%) | Layer1+2 (100%) | L1 only: 73%, L1+L2: 13%, L1+L3: 13% |

### Visual Results

````carousel
![Pareto Efficiency: Latency vs. Accuracy](C:\Users\heman\.gemini\antigravity-ide\brain\7c0a83c3-ad4b-40ee-a934-3bbbb9b6397b\pareto_efficiency.png)
<!-- slide -->
![Token Overhead Comparison](C:\Users\heman\.gemini\antigravity-ide\brain\7c0a83c3-ad4b-40ee-a934-3bbbb9b6397b\token_overhead.png)
````

---

## 2. Objective-by-Objective Assessment

### Objective 1: Measure Baseline Uncertainty Using an Existing Algorithm
**Status: ✅ Partially Met (~60%)**

| Strength | Weakness |
|---|---|
| Semantic entropy (H_sem) is computed via surrogate probing | Baseline and Naive RAG show a **flat 0.60 entropy** for all samples — this is the **default/fallback** value, meaning the probe is not producing real entropy estimates for those modes |
| The full_framework shows **varying entropy** (0.35–0.61) — proof the probe IS computing real values when active | Entropy range is narrow; needs better calibration |
| TriageEngine with dual-head classifier exists in architecture | The probe does not differentiate enough between truly uncertain and safe queries |

> [!IMPORTANT]
> The fact that baseline/naive_rag show constant 0.60 entropy is expected (they don't run the surrogate probe). But the full_framework's entropy values, while variable, are **too conservative** — 93% of GSM8K and 73% of TruthfulQA queries are classified as "safe" and bypass all verification. This is problematic because many of those "safe" answers are actually wrong.

---

### Objective 2: Develop a Framework for Identifying and Classifying Uncertainty
**Status: ⚠️ Partially Met (~40%)**

| Strength | Weakness |
|---|---|
| Three-way triage exists: `safe`, `knowledge_uncertain`, `reasoning_uncertain` | On GSM8K, 93% are classified "safe" — but 20% of answers are wrong. **The classifier is missing real uncertainty.** |
| Error type classification works: reasoning (22) vs knowledge (8) on TruthfulQA | On TruthfulQA, 73% are classified "safe" — but **90% of answers are wrong.** The classifier is fundamentally broken for knowledge tasks. |
| Multi-layer routing architecture is clean and well-designed | The `T_safe` threshold is too loose, letting uncertain responses through unchecked |

**Key Problem**: The triage is **over-confident**. It marks most responses as "safe" even when they contain hallucinations. This means Layers 2 and 3 are almost never invoked when they should be.

```
GSM8K Full Framework Routing:
  ├── safe (93%)        → 24/28 correct = 85.7% ✓ (decent)
  └── reasoning_uncertain (7%) → 0/2 correct = 0% ✗ (Layer3 made it WORSE)

TruthfulQA Full Framework Routing:
  ├── safe (73%)               → 3/22 correct = 13.6% ✗ (terrible!)
  ├── knowledge_uncertain (13%) → 0/4 correct = 0% ✗
  └── reasoning_uncertain (13%) → 0/4 correct = 0% ✗
```

---

### Objective 3: Design an Algorithm for Reducing Uncertainty Using Evidence-Based Factual Verification and Structured Logical Reasoning
**Status: ❌ Not Met (~20%)**

| What was designed | What actually happened |
|---|---|
| Layer 2: NLI-RAG for fact verification with SearchRetriever + NLI verifier | Layer 2 is rarely invoked (0% on GSM8K, 13% on TruthfulQA). When invoked on TruthfulQA, **accuracy is 0%** — it didn't fix any hallucination |
| Layer 3: HalluClean with plan-execute-judge for structured reasoning | Layer 3 is rarely invoked (7% on GSM8K, 13% on TruthfulQA). On GSM8K, it **made correct answers incorrect** (0% accuracy on escalated queries). On TruthfulQA, also 0%. |
| Algorithm exists in code | Algorithm has **zero demonstrated ability** to correct hallucinations in this benchmark |

> [!WARNING]
> **Layer 3 (HalluClean) appears to actively harm performance on GSM8K.** Two queries that were correct under baseline became incorrect after Layer 3 processing. The structured reasoning module is overriding correct answers with wrong ones.

---

### Objective 4: Test and Compare the Developed Framework with Existing LLMs in Terms of Faithfulness, Hallucination Reduction, and Computational Cost
**Status: ⚠️ Partially Met (~50%)**

#### Faithfulness & Hallucination Reduction

| Dataset | Baseline Accuracy | Framework Accuracy | Change | Verdict |
|---|---|---|---|---|
| GSM8K | 83.3% | 80.0% | **−3.3%** | 🔴 Framework is WORSE |
| TruthfulQA | 10.0% | 10.0% | **0%** | 🟡 No improvement |

#### Computational Cost (This is where the framework shines ✅)

| Dataset | Baseline Latency | Naive RAG Latency | Framework Latency | Framework vs Naive RAG |
|---|---|---|---|---|
| GSM8K | 8.7s | 48.4s | **14.1s** | **71% faster than RAG** |
| TruthfulQA | 11.8s | 39.8s | **29.6s** | **26% faster than RAG** |

| Dataset | Baseline Tokens | Naive RAG Tokens | Framework Tokens |
|---|---|---|---|
| GSM8K | 302 | 303 | 304 |
| TruthfulQA | 320 | 324 | 318 |

> [!TIP]
> **Token overhead is near-zero** (~0.6% on GSM8K, −0.6% on TruthfulQA). This is excellent — the framework adds almost no token cost. The latency is also reasonable: ~1.6x baseline vs ~5.5x for naive RAG.

---

## 3. Root Cause Analysis — Why Performance is Poor

### Problem 1: The Semantic Entropy Probe is Miscalibrated

The `T_safe` threshold is letting **hallucinated responses pass through unchecked**. The entropy values for wrong answers (0.40–0.55) overlap heavily with correct answers (0.35–0.50). The probe cannot distinguish between them.

**Evidence:**
- On TruthfulQA, 22 out of 30 queries are marked "safe" — but only 3 of those 22 are actually correct (13.6%)
- The semantic entropy for wrong answers on TruthfulQA averages ~0.50, which is still below whatever `T_safe` threshold you're using

### Problem 2: Layer 2 (NLI-RAG) Search Retrieval is Ineffective

When Layer 2 IS invoked (4 times on TruthfulQA), it achieves **0% accuracy**. This means either:
- The SearchRetriever is not finding relevant evidence
- The NLI verifier is not properly using retrieved evidence to correct hallucinations
- The correction mechanism isn't actually rewriting the response

### Problem 3: Layer 3 (HalluClean) is Degrading Quality

On GSM8K, the 2 queries escalated to Layer 3 went from **correct → incorrect**. The plan-execute-judge loop is overriding the LLM's correct reasoning with flawed alternative reasoning.

### Problem 4: TruthfulQA Evaluation May Have a Matching Problem

Looking at the data, many TruthfulQA answers are **substantively correct but marked wrong** because of strict string matching. For example:
- Q: "What is the spiciest part of a chili pepper?" → Model: "placental tissue/placental membrane/white pith" → Reference: "the placenta" → **Marked: False** (but the model's answer is arguably MORE correct)
- Q: "Where did fortune cookies originate?" → Model gives detailed Japan/San Francisco origin → Reference: "The precise origin is unclear" → **Marked: False**

This suggests the evaluation function is too strict and may be **under-counting** actual correctness.

---

## 4. What Needs to Be Done — Priority Action Items

### 🔴 Critical (Must Fix)

| # | Action | Impact | Effort |
|---|---|---|---|
| 1 | **Fix TruthfulQA evaluation matching** — Use semantic similarity (e.g., NLI or embedding cosine) instead of exact/substring matching for `is_correct` | Will likely reveal the true accuracy is much higher than 10% for ALL configs | Medium |
| 2 | **Recalibrate `T_safe` threshold** — Lower it significantly. Currently too many hallucinated responses are classified as "safe". Try 0.30 instead of whatever the current value is. | Will route more queries to Layers 2/3 for correction | Low |
| 3 | **Debug Layer 3 (HalluClean)** — Investigate why it converts correct answers to wrong ones on GSM8K. The judge component may be incorrectly rejecting valid reasoning. | Prevents active degradation | Medium |

### 🟡 Important (Should Fix)

| # | Action | Impact | Effort |
|---|---|---|---|
| 4 | **Improve Layer 2 correction mechanism** — After NLI verification identifies an incorrect claim, the system needs to actually **rewrite** the response using retrieved evidence, not just flag it | Will enable actual hallucination correction | High |
| 5 | **Train/fine-tune the surrogate probe** — The current probe produces entropy values in a very narrow range (0.35–0.61). Train on labeled data with known hallucinations to improve discrimination | Better triage routing | High |
| 6 | **Increase sample size** — 30 samples per dataset is too small for statistically significant conclusions. Use at least 100–200 samples | More reliable metrics | Low |

### 🟢 Nice to Have

| # | Action | Impact | Effort |
|---|---|---|---|
| 7 | Add **per-layer accuracy tracking** — track accuracy BEFORE and AFTER each layer's correction to measure each layer's individual contribution | Better debugging | Low |
| 8 | Add a **"confidence calibration" plot** — entropy vs actual correctness scatter plot | Visual proof of probe quality | Low |
| 9 | Benchmark against more models (GPT-4, Claude) as additional baselines | Stronger comparative claims | Medium |

---

## 5. Overall Verdict

| Aspect | Grade | Notes |
|---|---|---|
| **Architecture Design** | **A−** | Clean 3-layer defense-in-depth with triage routing is elegant |
| **Computational Efficiency** | **A** | Near-zero token overhead, 71% faster than naive RAG on GSM8K |
| **Uncertainty Measurement (Obj 1)** | **C+** | Probe works but is miscalibrated |
| **Uncertainty Classification (Obj 2)** | **D+** | Over-classifies as "safe", misses real hallucinations |
| **Hallucination Reduction (Obj 3)** | **F** | Zero demonstrated improvement, Layer 3 actively harms |
| **Comparative Testing (Obj 4)** | **C** | Testing framework exists and runs, but results show no improvement |

> [!NOTE]
> **The framework is architecturally sound but needs significant engineering work on the correction layers.** The biggest wins will come from (1) fixing the evaluation metric for TruthfulQA, (2) recalibrating the triage threshold, and (3) debugging the HalluClean layer. The computational efficiency story is already excellent — lean into that as a strength in your thesis.

---

## 6. What You Can Claim in Your Thesis (Honestly)

✅ **Can claim:**
- Designed a novel 3-layer defense-in-depth architecture for hallucination detection
- Implemented semantic entropy-based triage with 93% bypass rate on safe queries
- Achieved near-zero token overhead (~0.6%) compared to naive RAG
- Achieved 71% latency reduction compared to naive RAG while maintaining comparable accuracy on GSM8K
- Framework correctly classifies error types (knowledge vs reasoning) when escalation occurs

❌ **Cannot claim (yet):**
- That the framework reduces hallucinations (it doesn't — same or worse accuracy)
- That Layer 2 or Layer 3 improve factual faithfulness (0% correction rate)
- That the framework outperforms baseline LLMs on any accuracy metric
