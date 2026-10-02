# Code Architecture & Component Breakdown: Phase 1
## M.Tech Thesis Chapter 4: Baseline System Implementation & Software Architecture
### Reference Commit: `4187783ca0c74f98a71c956f9ae84a925b3552f9` (`feat(main): phase 1 implementaion`)

---

### 1. System Architecture & Repository Layout

In commit `4187783`, the entire framework was established as an asynchronous, pure-Python microservice using **FastAPI**, **Uvicorn**, and **PyTorch**.

```
d:\Coding\Project\LLM-Hallucination (at commit 4187783)
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI entrypoint, lifespan loader & REST routes
│   ├── config.py                   # Pydantic Settings configuration registry
│   ├── router.py                   # Central triage dispatch state machine
│   ├── models/
│   │   └── schemas.py              # Pydantic v1/v2 data transfer models
│   ├── layer1_triage/
│   │   ├── probe_model.py          # PyTorch LogisticProbe class & entropy estimator
│   │   └── activation_hook.py      # PyTorch forward hook utilities for hidden states
│   ├── layer2_nli_rag/
│   │   ├── proposition_parser.py   # Atomic proposition extraction via zero-shot prompts
│   │   ├── search_retriever.py     # Async DuckDuckGo search & ChromaDB retriever
│   │   └── nli_verifier.py         # DeBERTa-v3 cross-encoder NLI verification
│   ├── layer3_halluclean/
│   │   ├── planner.py              # Symbolic task-oriented DAG planner
│   │   ├── executor.py             # Plan-guided sequential reasoning loop
│   │   └── judge.py                # Final constraint validation & output synthesis
│   └── llm_clients/
│       └── ollama_client.py        # Asynchronous HTTP wrapper for local Ollama daemon
├── evaluation/
│   ├── benchmark_runner.py         # Initial batch evaluation script
│   ├── metrics.py                  # Factual accuracy and latency metric calculators
│   └── plot_generator.py           # Publication plotting utilities
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

### 2. Component-by-Component Implementation Analysis

#### 2.1. API Gateway & Lifespan Architecture (`app/main.py`)
`app/main.py` establishes the asynchronous HTTP interface:
* **Lifespan Manager:** Utilizes FastAPI's `asynccontextmanager` to pre-load heavy machine learning models (PyTorch probe weights, DeBERTa-v3 cross-encoder, ChromaDB vector collections) into GPU memory at server startup, avoiding per-request initialization penalties.
* **REST Endpoints Exposed:**
  * `GET /api/v1/health`: Readiness probe checking Ollama connectivity and GPU availability.
  * `POST /api/v1/query`: Full end-to-end Defense-in-Depth pipeline.
  * `POST /api/v1/query/layer1`: Isolated Layer 1 evaluation (Baseline/Raw LLM).
  * `POST /api/v1/query/layer2`: Isolated Layer 2 evaluation (Naive RAG).
  * `POST /api/v1/query/layer3`: Isolated Layer 3 evaluation (HalluClean).

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Defense-in-Depth Framework...")
    # Initialize Ollama client, probe model, NLI verifier, and router
    app.state.router = QueryRouter(settings=settings)
    await app.state.router.initialize()
    yield
    logger.info("Shutting down Defense-in-Depth Framework...")
```

---

#### 2.2. Configuration Registry (`app/config.py`)
`app/config.py` centralizes all hyperparameters using `pydantic-settings`:
* `ollama_base_url`: Host address for the local inference daemon (`http://localhost:11434`).
* `default_model`: Generator model identifier (`llama3.1:8b`).
* `safe_threshold` ($T_{safe}$): Semantic entropy threshold (`0.55`).
* `nli_model_name`: Cross-encoder checkpoint (`cross-encoder/nli-deberta-v3-small`).
* `nli_entailment_threshold` ($\tau_{ent}$): Entailment cutoff (`0.60`).
* `nli_contradiction_threshold` ($\tau_{con}$): Contradiction cutoff (`0.50`).

---

#### 2.3. Triage Router State Machine (`app/router.py`)
The `QueryRouter` orchestrates the multi-tier dispatch logic:
1. Dispatches the user prompt to Ollama to generate an initial draft response $y$.
2. Invokes Layer 1 triage to compute semantic entropy $H_{sem}$.
3. **Fast Path Check:** If $H_{sem} < T_{safe}$, execution terminates immediately, returning the draft.
4. **Heuristic Failure-Mode Branching:** If $H_{sem} \ge T_{safe}$, Phase 1 inspects the prompt for mathematical/logical keywords (`"calculate"`, `"step"`, `"prove"`, `"solve"`):
   * If logical keywords match $\rightarrow$ Routes to Layer 3 (HalluClean).
   * Otherwise $\rightarrow$ Routes to Layer 2 (NLI-RAG).

```python
# Phase 1 Decision Branching Logic in router.py
if triage_result.decision == TriageDecision.SAFE:
    return QueryResponse(answer=draft, layers_invoked="layer1_only")

# Keyword heuristic in Phase 1:
if any(kw in request.prompt.lower() for kw in ["calculate", "prove", "step-by-step", "solve"]):
    return await self._process_layer3(request, draft, triage_result)
else:
    return await self._process_layer2(request, draft, triage_result)
```

---

#### 2.4. Layer 1: Probing & Activation Hooks (`probe_model.py` & `activation_hook.py`)
* `ActivationBundle`: A container tracking layer activations, attention matrices, and token IDs.
* `ActivationHook`: Uses PyTorch's `register_forward_hook` on transformer decoder blocks to intercept tensor activations during generation.
* `LogisticProbe`: A PyTorch linear layer ($d \rightarrow 1$) with sigmoid activation trained to distinguish correct from hallucinatory hidden states:
  $$P(\text{hallucination}) = \sigma(\mathbf{W} h + b)$$

---

#### 2.5. Layer 2: Hypothesis-Testing NLI-RAG (`proposition_parser.py`, `search_retriever.py`, `nli_verifier.py`)
* **Proposition Parser:** Uses a zero-shot prompt with strict JSON formatting to decompose complex text into declarative statements:
  `[{"id": 1, "statement": "Watermelon seeds are safe to ingest."}]`
* **Search Retriever:** Issues queries to DuckDuckGo via `ddgs` or searches a local ChromaDB SQLite vector store using embeddings.
* **NLI Verifier:** Loads `cross-encoder/nli-deberta-v3-small` via Hugging Face `transformers`. Computes softmax probabilities over `[contradiction, neutral, entailment]` for each (hypothesis, evidence) pair.

---

#### 2.6. Layer 3: HalluClean Symbolic State Machine (`planner.py`, `executor.py`, `judge.py`)
* `HalluCleanPlanner`: Prompts the generator to construct a directed execution graph:
  ```json
  {"steps": [{"step_id": 1, "description": "Identify missing person criteria", "dependencies": []}]}
  ```
* `HalluCleanExecutor`: Computes topological sort over graph dependencies and executes each step with context isolated to only the outputs of prerequisite steps.
* `HalluCleanJudge`: Compares all step outputs against the original prompt constraints, validates consistency, and synthesizes the final response.

---

#### 2.7. Asynchronous LLM Client (`ollama_client.py`)
* Wraps the local Ollama `/api/generate` and `/api/chat` endpoints.
* Configured with `httpx.AsyncClient` supporting streaming responses, timeout handling (default 120s), and automated JSON payload validation.

---

### 3. Summary of Baseline Architecture Strengths

1. **Pure-Python Consistency:** Eliminates inter-process serialization overhead; all ML models (PyTorch, Hugging Face, ChromaDB) live within a single Python runtime.
2. **Modular Layer Decoupling:** Layers 1, 2, and 3 are cleanly decoupled and can be invoked independently or orchestrated in unison.
3. **Async-First Execution:** Leverages Python `asyncio` to prevent I/O blocking during network search and local model execution.
