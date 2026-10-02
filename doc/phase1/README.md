# Phase 1: Pure-Python Defense-in-Depth Framework Baseline
## M.Tech Thesis Research & Engineering Documentation
### Baseline Commit: `4187783ca0c74f98a71c956f9ae84a925b3552f9` (`feat(main): phase 1 implementaion`)

---

### Executive Summary

Phase 1 established the foundational pure-Python architectural prototype for the **Defense-in-Depth LLM Hallucination Mitigation Framework**. 
Prior research in hallucination mitigation typically treated detection and correction as separate, disconnected systems:
1. **Pre-generation interventions** (e.g., knowledge editing, constrained decoding) that fail to adapt dynamically to runtime epistemic context;
2. **Naive Retrieval-Augmented Generation (RAG)** that injects noisy, unverified external context into the prompt, multiplying token expenditure; and
3. **Multi-Agent debate protocols** that incur prohibitive computational latency by running multiple redundant autoregressive generations.

Phase 1 unified these disjoint paradigms into an asynchronous, three-tier hierarchical pipeline operating within a pure-Python microservice architecture (FastAPI).

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Phase 1 Pure-Python Architecture                    │
│                                                                        │
│  FastAPI Async Gateway (asyncio / uvicorn)                             │
│     │                                                                  │
│     ├── Layer 1: Internal Triage via Semantic Entropy Probes (SEPs)    │
│     │   └── PyTorch Logistic Estimator on Hidden States (h_t^(l))     │
│     │                                                                  │
│     ├── Layer 2: Hypothesis-Testing RAG with NLI Verification         │
│     │   └── Atomic Proposition Extraction + DuckDuckGo + DeBERTa-v3    │
│     │                                                                  │
│     └── Layer 3: Structured Reasoning via HalluClean State Machine     │
│         └── 4-Stage Symbolic DAG (Planner -> Executor -> Judge)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1 Documentation Directory

| Document | Focus | Target Thesis Chapters |
| :--- | :--- | :--- |
| [**`01_theoretical_foundations.md`**](file:///d:/Coding/Project/LLM-Hallucination/doc/phase1/01_theoretical_foundations.md) | Theoretical premise, mathematical formulations for SEPs, NLI hypothesis testing, and the 4-stage HalluClean paradigm. | Chapter 1 & 3: Introduction, Problem Formulation & Baseline Theory |
| [**`02_code_architecture_and_components.md`**](file:///d:/Coding/Project/LLM-Hallucination/doc/phase1/02_code_architecture_and_components.md) | Detailed component-by-component software design of commit `4187783` (FastAPI, Ollama client, PyTorch probes, NLI, and HalluClean DAG). | Chapter 4: Baseline System Implementation & Software Architecture |
| [**`03_limitations_and_gap_analysis.md`**](file:///d:/Coding/Project/LLM-Hallucination/doc/phase1/03_limitations_and_gap_analysis.md) | In-depth academic critique of Phase 1 bottlenecks (Opaque Quantized Generator, Heuristic Routing, Single-Pair NLI noise) motivating Phase 2. | Chapter 4 & 5: Architectural Limitations & Research Gap Analysis |
| [**`04_thesis_integration_guide.md`**](file:///d:/Coding/Project/LLM-Hallucination/doc/phase1/04_thesis_integration_guide.md) | Drop-in LaTeX templates, system architecture figures, mathematical formulations, and BibTeX citations for `main.tex`. | Thesis Integration & Literature Grounding |

---

### Component Overview at Commit `4187783`

| Module | File Path | Phase 1 Responsibility |
| :--- | :--- | :--- |
| **API Gateway** | `app/main.py` | FastAPI server entrypoint with lifespan event handler for asynchronous model initialization. |
| **System Config** | `app/config.py` | Central configuration using Pydantic Settings ($T_{safe} = 0.55$, Ollama model names, API paths). |
| **Triage Router** | `app/router.py` | Central dispatch orchestrator evaluating probe uncertainty and routing requests. |
| **Layer 1 Probe** | `app/layer1_triage/probe_model.py` | Logistic regression probe parameterized over hidden states for semantic entropy estimation. |
| **Activation Hook** | `app/layer1_triage/activation_hook.py` | PyTorch forward-hook utilities designed to intercept intermediate transformer activations. |
| **Claim Parser** | `app/layer2_nli_rag/proposition_parser.py` | Zero-shot LLM prompt decomposing drafts into atomic propositions. |
| **Search Retriever** | `app/layer2_nli_rag/search_retriever.py` | Asynchronous web search retriever querying DuckDuckGo and local ChromaDB collections. |
| **NLI Verifier** | `app/layer2_nli_rag/nli_verifier.py` | Hugging Face cross-encoder (DeBERTa-v3) evaluating directional entailment vs. contradiction. |
| **HalluClean Planner** | `app/layer3_halluclean/planner.py` | Symbolic task planner compiling complex reasoning prompts into execution graphs. |
| **HalluClean Executor**| `app/layer3_halluclean/executor.py` | Sequential step-by-step reasoning executor maintaining isolated local contexts. |
| **HalluClean Judge** | `app/layer3_halluclean/judge.py` | Final constraint checker and multi-trace synthesis judge. |
| **Ollama Client** | `app/llm_clients/ollama_client.py` | Asynchronous HTTP client wrapping Ollama's `/api/generate` endpoint. |
| **Data Schemas** | `app/models/schemas.py` | Pydantic data models defining the request/response contracts. |
