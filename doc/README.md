# M.Tech Thesis Documentation Suite
## Defense-in-Depth Hallucination Mitigation Framework for Large Language Models

This directory contains the complete academic and engineering documentation tracking the evolution of the research project across its major phases:

---

### Directory Structure

```
d:\Coding\Project\LLM-Hallucination\doc
│
├── phase1/                                # Baseline Implementation Documentation
│   ├── README.md                          # Phase 1 Overview & Component Directory
│   ├── 01_theoretical_foundations.md      # Chapter 3: SEPs, NLI-RAG & HalluClean Formulation
│   ├── 02_code_architecture_and_components.md # Chapter 4: Baseline Code Breakdown (Commit 4187783)
│   ├── 03_limitations_and_gap_analysis.md # Chapter 4/5: Opaque Generator & Heuristic Routing Gaps
│   └── 04_thesis_integration_guide.md    # LaTeX Templates & Baseline Equations for main.tex
│
└── phase2/                                # Improvised Novelties & Evaluation Documentation
    ├── README.md                          # Phase 2 Overview & Staged Changes Matrix
    ├── 01_theoretical_foundations.md      # Chapter 3: Surrogate Probing & Consensus Formulations
    ├── 02_code_architecture_and_diffs.md  # Chapter 4: DualHeadClassifier, Hook Diffs & Pydantic v2
    ├── 03_experimental_setup_and_interim_results.md # Chapter 5: Hardware Setup, Latency & Live Trace Logs
    └── 04_thesis_integration_guide.md    # LaTeX Templates, Algorithm Pseudocode & BibTeX Citations
```

---

### Comparative Evolution Matrix

| Research Dimension | Phase 1 (Baseline Commit `4187783`) | Phase 2 (Improvised Framework) |
| :--- | :--- | :--- |
| **Layer 1 Generator Probing** | Mocked/Broken for quantized backends | **Surrogate Proxy Probing:** TinyLlama-1.1B forward hook capturing terminal hidden vector $h^* \in \mathbb{R}^{2048}$ |
| **Failure-Mode Routing** | Brittle prompt keyword matching (`"calculate"`, `"solve"`) | **Dual-Head Linear Classifier:** Latent manifold projecting $h^*$ to entropy $H_{sem}$ and error logits $\mathbf{z} \in \mathbb{R}^2$ |
| **Layer 2 RAG Verification** | Binary single-pair thresholding; highly sensitive to noise | **Consensus Aggregation:** Bounded net polarity metric $S_{consensus}(p_i) \in [-1, 1]$ across multi-evidence sets |
| **Type Safety & Auditing** | Legacy loose schemas | Strict **Pydantic v2** immutable models with microsecond latency tracing |
| **Empirical Evaluation** | Placeholder test script | Automated async benchmarking on **TruthfulQA** ($N=30$) and **GSM8K** ($N=30$) |
| **Thesis Visualizations** | None | Automated 300-DPI **Pareto Efficiency** curves and **Token Overhead** charts |

---

### How to Use This Documentation for Your M.Tech Thesis

1. **Chapter 1 & 2 (Introduction & Literature Review):**
   * Reference [`doc/phase1/01_theoretical_foundations.md`](file:///d:/Coding/Project/LLM-Hallucination/doc/phase1/01_theoretical_foundations.md) and [`doc/phase1/03_limitations_and_gap_analysis.md`](file:///d:/Coding/Project/LLM-Hallucination/doc/phase1/03_limitations_and_gap_analysis.md) for literature gap context.
2. **Chapter 3 (Proposed Methodology):**
   * Use [`doc/phase2/01_theoretical_foundations.md`](file:///d:/Coding/Project/LLM-Hallucination/doc/phase2/01_theoretical_foundations.md) and drop in the LaTeX equations from [`doc/phase2/04_thesis_integration_guide.md`](file:///d:/Coding/Project/LLM-Hallucination/doc/phase2/04_thesis_integration_guide.md).
3. **Chapter 4 (System Architecture & Implementation):**
   * Combine [`doc/phase1/02_code_architecture_and_components.md`](file:///d:/Coding/Project/LLM-Hallucination/doc/phase1/02_code_architecture_and_components.md) and [`doc/phase2/02_code_architecture_and_diffs.md`](file:///d:/Coding/Project/LLM-Hallucination/doc/phase2/02_code_architecture_and_diffs.md) to showcase the before-and-after evolution.
4. **Chapter 5 (Experimental Results & Discussion):**
   * Incorporate the testbed specifications and live traces from [`doc/phase2/03_experimental_setup_and_interim_results.md`](file:///d:/Coding/Project/LLM-Hallucination/doc/phase2/03_experimental_setup_and_interim_results.md), inserting the final CSV results and generated PNG plots.
