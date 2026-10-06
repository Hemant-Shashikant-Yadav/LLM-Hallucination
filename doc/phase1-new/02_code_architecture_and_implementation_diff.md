# Code Architecture & Component Implementation: Phase 1
## M.Tech Thesis Chapter 4: Baseline Software Architecture & Implementation Engineering
### Baseline Commit: `4187783ca0c74f98a71c956f9ae84a925b3552f9` (`feat(main): phase 1 implementaion`)

---

### 1. Repository Layout & Software Architecture

In commit `4187783`, the baseline framework was structured as a modular, pure-Python asynchronous service using **FastAPI**, **Uvicorn**, and **PyTorch**:

```
d:\Coding\Project\LLM-Hallucination (commit 4187783)
├── app/
│   ├── main.py                     # FastAPI entrypoint & model lifespan loader
│   ├── config.py                   # Pydantic Settings registry (T_safe = 0.55)
│   ├── router.py                   # Central triage dispatch state machine
│   ├── models/
│   │   └── schemas.py              # Data models (QueryRequest, QueryResponse, etc.)
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
│       └── ollama_client.py        # Asynchronous HTTP wrapper for Ollama daemon
├── evaluation/
│   ├── benchmark_runner.py         # Initial batch evaluation script
│   ├── metrics.py                  # Accuracy and latency metric calculators
│   └── plot_generator.py           # Publication plotting utilities
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

### 2. Component-by-Component Implementation Analysis

#### 2.1. Lifespan Architecture & FastAPI Server (`app/main.py`)
`app/main.py` established an async lifespan context manager to instantiate singleton resources:
* Pre-loads the cross-encoder NLI model (`cross-encoder/nli-deberta-v3-small`) on CUDA.
* Instantiates `OllamaClient` and the `QueryRouter`.
* Mounts REST routes for `/api/v1/health`, `/api/v1/query`, and isolated layer endpoints.

#### 2.2. Configuration Registry (`app/config.py`)
Hyperparameters were declared using `pydantic-settings`:
```python
class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama2:7b"
    safe_threshold: float = 0.55             # T_safe
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"
    nli_entailment_threshold: float = 0.60   # tau_ent
    nli_contradiction_threshold: float = 0.50 # tau_con
```

#### 2.3. Triage Router State Machine (`app/router.py`)
`app/router.py` orchestrated the end-to-end execution flow:
1. Calls `ollama_client.generate()` to obtain draft response $y$.
2. Invokes Layer 1 `triage()` to evaluate $H_{sem}$.
3. **Fast Path Exit:** If $H_{sem} < T_{safe}$, returns the response immediately.
4. **Lexical Failure-Mode Branching:** If $H_{sem} \ge T_{safe}$, checks for arithmetic/logical keywords:
```python
# Phase 1 Lexical Branching Logic in router.py
if triage_result.decision == TriageDecision.SAFE:
    return QueryResponse(answer=draft, layers_invoked="layer1_only")

# Hardcoded keyword check in Phase 1
if any(kw in request.prompt.lower() for kw in ["calculate", "prove", "step-by-step", "solve"]):
    return await self._process_layer3(request, draft, triage_result)
else:
    return await self._process_layer2(request, draft, triage_result)
```

#### 2.4. Layer 1: Probing & Activation Hooks (`probe_model.py` & `activation_hook.py`)
* `ActivationHook`: Attempted to register PyTorch hooks on the generator:
```python
class ActivationHook:
    def __init__(self, model: nn.Module, layer_idx: int = -1):
        self.hook = model.layers[layer_idx].register_forward_hook(self._hook_fn)
```
* `LogisticProbe`: A single linear layer over hidden dimension $d$:
```python
class LogisticProbe(nn.Module):
    def __init__(self, input_dim: int = 4096):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))
```

#### 2.5. Layer 2: NLI-RAG Verification (`proposition_parser.py`, `nli_verifier.py`)
* `proposition_parser.py`: Uses an LLM zero-shot prompt with strict JSON output to decompose text into atomic propositions.
* `nli_verifier.py`: Evaluates propositions sequentially and short-circuits on the first match:
```python
# Phase 1 short-circuit verification in nli_verifier.py
for doc in evidence_documents:
    scores = self._score_pair(proposition.statement, doc.text)
    if scores['contradiction'] >= self.settings.nli_contradiction_threshold:
        proposition.status = "contradicted"
        break  # Short-circuit on first contradiction
```

#### 2.6. Layer 3: HalluClean 4-Stage Symbolic DAG (`planner.py`, `executor.py`, `judge.py`)
* `HalluCleanPlanner`: Decomposes complex queries into an `ExecutionGraph` containing `PlanNode` objects with explicit dependencies.
* `HalluCleanExecutor`: Sorts nodes topologically and invokes Ollama sequentially with local context isolation.
* `HalluCleanJudge`: Evaluates step outputs against initial prompt constraints and emits the final answer.

---

### 3. Step-by-Step Phase 1 Execution Flow Trace

```
[Client Request: POST /api/v1/query]
       │
       ▼
[app/main.py -> Lifespan Router]
       │
       ▼
[Ollama HTTP Call: /api/generate] ──► Draft Response y (String)
       │
       ▼
[app/layer1_triage/probe_model.py]
       │  Attempt PyTorch hook on Ollama -> Fails (Ollama is black-box C++)
       │  Fallback: Output logprob or uncalibrated mock values
       ▼
[app/router.py Branching]
       ├── If H_sem < 0.55: Fast Path Exit -> HTTP 200 OK
       └── If H_sem >= 0.55:
           ├── If "calculate" in prompt: Route to Layer 3 (HalluClean)
           └── Else: Route to Layer 2 (NLI-RAG)
                  │
                  ▼
           [Layer 2 NLI-RAG]
           Extract propositions -> DuckDuckGo Search -> DeBERTa-v3
           Short-circuit on first contradiction -> HTTP 200 OK
```

---

### 4. Technical Shortcomings Discovered in Phase 1

1. **Quantization Hook Disconnect:** The PyTorch hook could not intercept tensors from Ollama's quantized 4-bit C++ runtime, making Layer 1 non-functional for local quantized models.
2. **Lexical Brittleness:** Keyword-based failure-mode branching misrouted factual math questions to HalluClean and logic puzzles to web search.
3. **Short-Circuit Vulnerability:** A single noisy web search passage marked valid answers as contradicted, triggering false-positive hallucination flags.
