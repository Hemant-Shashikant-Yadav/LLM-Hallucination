# Phase 2: Defense-in-Depth Framework Architectural Improvisation
## M.Tech Thesis Research & Engineering Documentation

This directory contains comprehensive theoretical, architectural, and experimental documentation detailing the transition of the **Defense-in-Depth LLM Hallucination Mitigation Framework** from **Phase 1 (Baseline Prototype)** to **Phase 2 (Algorithmic Novelties & Rigorous Evaluation)**.

---

### Executive Summary

In Phase 1, the framework established a three-layer concept:
1. An internal triage layer using heuristic logit analysis,
2. An external NLI-RAG retrieval verification layer, and
3. A symbolic multi-stage reasoning engine (HalluClean).

While functional as a proof-of-concept, Phase 1 suffered from four critical research limitations:
1. **Opaque Generator Dilemma:** Quantized 4-bit local generators (e.g., Ollama Llama 3.1:8B) do not expose internal transformer activations or hidden states via API, rendering white-box probing impossible.
2. **Heuristic Error Triage:** Routing between knowledge-based hallucination (requiring retrieval) and reasoning-based hallucination (requiring multi-step planning) relied on simplistic keyword heuristics rather than latent representations.
3. **Fragile Single-Pair NLI:** Evidence verification in Layer 2 evaluated isolated proposition-evidence pairs without aggregating multi-source consensus, leading to high variance and sensitivity to retriever noise.
4. **Lack of Automated Empirical Benchmarking:** No unified, asynchronous evaluation harness existed to benchmark accuracy, latency, and token overhead against standardized datasets (TruthfulQA and GSM8K).

**Phase 2 resolves each of these limitations** with formal algorithmic solutions designed specifically for publication and M.Tech thesis evaluation:
- **Novelty 1: Surrogate Proxy Probing (`app/layer1_triage/probe_model.py`)** using TinyLlama-1.1B to capture final-layer hidden states ($h_t^{(L)} \in \mathbb{R}^{2048}$).
- **Novelty 2: Dual-Head Uncertainty Classifier (`app/layer1_triage/probe_model.py` & `app/router.py`)** predicting both continuous Semantic Entropy ($H_{sem}$) and categorical error logits (Knowledge vs. Reasoning).
- **Novelty 3: Consensus-Driven NLI Decision Rule (`app/layer2_nli_rag/nli_verifier.py`)** implementing an evidence-aggregated consensus score $S_{consensus}(p) \in [-1, 1]$.
- **Novelty 4: Automated Asynchronous Benchmarking & Visualization Suite (`evaluation/`)** executing 3-way evaluation across Baseline, Naive RAG, and Proposed Defense-in-Depth with automated LaTeX table output and publication-quality plots.

---

### Phase 2 Documentation Roadmap

| Document | Focus | Target Thesis Chapters |
| :--- | :--- | :--- |
| [**`01_theoretical_foundations.md`**](file:///d:/Coding/Project/LLM-Hallucination/doc/phase2/01_theoretical_foundations.md) | Formal mathematical derivations, loss functions, representational alignment, consensus bounds, and Pareto trade-off theory. | Chapter 3: Proposed Methodology & Theoretical Formulation |
| [**`02_code_architecture_and_diffs.md`**](file:///d:/Coding/Project/LLM-Hallucination/doc/phase2/02_code_architecture_and_diffs.md) | File-by-file audit of code changes, Pydantic v2 schemas, PyTorch hook mechanisms, and Before/After code diffs. | Chapter 4: System Implementation & Software Architecture |
| [**`03_experimental_setup_and_interim_results.md`**](file:///d:/Coding/Project/LLM-Hallucination/doc/phase2/03_experimental_setup_and_interim_results.md) | Benchmark configuration, hardware setup, latency analysis, warm vs. cold inference profiles, and interim metrics. | Chapter 5: Experimental Evaluation & Results |
| [**`04_thesis_integration_guide.md`**](file:///d:/Coding/Project/LLM-Hallucination/doc/phase2/04_thesis_integration_guide.md) | Ready-to-copy LaTeX code snippets, tables, algorithm pseudocode, and bibtex entries for `main.tex`. | Thesis Integration & Final Submission |

---

### Key File Changes Matrix (Staged in Git)

| Component | File Path | Phase 1 State | Phase 2 State (Current) | Research Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1 Probe** | `app/layer1_triage/probe_model.py` | Simple logistic regression probe stub | `SurrogateExtractor` + `DualHeadUncertaintyClassifier` with PyTorch forward hook | Solves black-box probe extraction for quantized LLMs |
| **Layer 2 NLI** | `app/layer2_nli_rag/nli_verifier.py` | Binary single-pair thresholding | Multi-evidence consensus aggregator $S_{consensus}(p)$ with $(T_{accept}, T_{reject})$ | Eliminates noise injection from individual conflicting search results |
| **Data Schemas** | `app/models/schemas.py` | Basic pipeline result schemas | Pydantic v2 audited models (`ErrorClassification`, `DualHeadProbeResult`, `NLIVerificationConsensus`) | Guarantees strict type safety, serialization, and thesis logging |
| **System Config** | `app/config.py` | Generic threshold constants | Structured settings with surrogate parameters, consensus thresholds, and fallback flags | Centralized experimental hyperparameters |
| **Orchestrator** | `app/router.py` | Static heuristic branching | Latency-aware async triage state machine with Fast Path bypass | Realizes Pareto-optimal latency-accuracy efficiency |
| **Benchmark** | `evaluation/benchmark_runner.py` | Placeholder script | Async multi-dataset runner (TruthfulQA + GSM8K) for 3 configurations | Generates empirical ground truth for thesis tables |
| **Visualization** | `evaluation/plot_generator.py` | Placeholder script | 300-DPI Matplotlib/Seaborn Pareto curve and token overhead generator | Produces publication-ready camera-ready figures |
