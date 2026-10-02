# Code Architecture & Implementation Diff: Phase 2
## M.Tech Thesis Chapter 4: System Architecture & Implementation Engineering

---

### 1. File-by-File Code Diff & Engineering Rationale

#### 1.1. `app/layer1_triage/probe_model.py`
In Phase 1, `probe_model.py` attempted to probe the primary generator directly through an `ActivationHook` class. Since the primary generator is executed within Ollama's external C++ runtime, this hook failed to intercept activations and defaulted to random noise.

In Phase 2, `probe_model.py` was completely refactored with two primary neural components: `SurrogateExtractor` and `DualHeadUncertaintyClassifier`.

##### `SurrogateExtractor` Implementation
* **Surrogate Model:** Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` loaded via Hugging Face `AutoModelForCausalLM` and `AutoTokenizer`.
* **Hardware Acceleration & Device Management:** Dynamically selects `device = "cuda" if torch.cuda.is_available() else "cpu"` with `torch_dtype=torch.float16` on CUDA to minimize VRAM footprint to ~2.2GB.
* **Terminal Layer Hook:** Identifies the terminal transformer decoder layer dynamically (`model.model.layers[-1]`) and registers a forward-pass hook:
```python
def _register_hook(self) -> None:
    """Register forward hook on the final transformer block layer."""
    target_layer = self._model.model.layers[-1]
    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Extract the representation at the final token index: [Batch, Hidden_Dim]
        self._final_hidden_state = hidden[:, -1, :].detach()
    self._hook_handle = target_layer.register_forward_hook(hook_fn)
```
* **Async Non-Blocking Execution:** Uses `asyncio.to_thread(...)` to ensure that PyTorch GPU tensor computations do not block FastAPI's event loop:
```python
async def extract_hidden_state(self, prompt: str, draft_response: str) -> torch.Tensor:
    return await asyncio.to_thread(self._extract_sync, prompt, draft_response)
```
* **Graceful Cold-Start & Fallback:** If the surrogate model cannot be loaded (e.g., low disk space or HuggingFace rate limit), the extractor initializes a fallback tensor sampled from a calibrated Gaussian prior $\mathcal{N}(0, 1)$, logging a warning without crashing the service.

##### `DualHeadUncertaintyClassifier` Implementation
* Replaces simple logistic regression with a multi-task PyTorch `nn.Module`:
```python
class DualHeadUncertaintyClassifier(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.entropy_head = nn.Linear(hidden_dim, 1)  # Head 1: H_sem
        self.router_head = nn.Linear(hidden_dim, 2)   # Head 2: [knowledge, reasoning]
        self.sigmoid = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(h)
        h_sem = self.sigmoid(self.entropy_head(feat))
        router_logits = self.router_head(feat)
        return h_sem, router_logits
```

---

#### 1.2. `app/layer2_nli_rag/nli_verifier.py`
In Phase 1, `nli_verifier.py` evaluated propositions sequentially and short-circuited (`break`) upon encountering the first document where $P(\text{con}) > 0.50$.

In Phase 2, `nli_verifier.py` was refactored to implement **Exhaustive Multi-Passage Cross-Encoding** with normalized consensus aggregation:
* **Batch Cross-Encoding:** Instead of single-pair predictions, all (proposition, evidence) pairs are formatted into batched sentence pairs and fed to `cross-encoder/nli-deberta-v3-small` in a single GPU pass:
```python
pairs = [(proposition.statement, doc.text) for doc in evidence_documents]
features = self._tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(self._device)
with torch.no_grad():
    logits = self._model(**features).logits
    probs = torch.softmax(logits, dim=-1) # [N, 3] -> (contradiction, neutral, entailment)
```
* **Conflict-Aware Consensus Computation:**
```python
net_polarities = [p["entailment"] - p["contradiction"] for p in per_evidence_scores]
consensus_score = float(sum(net_polarities) / len(net_polarities))

if consensus_score >= settings.nli_accept_threshold:   # DID_NLI_ACCEPT_THRESHOLD = 0.3
    decision = NLIDecision.ENTAILED
elif consensus_score <= settings.nli_reject_threshold: # DID_NLI_REJECT_THRESHOLD = -0.3
    decision = NLIDecision.CONTRADICTED
else:
    decision = NLIDecision.NEUTRAL
```

---

#### 1.3. `app/models/schemas.py`
Phase 2 upgrades all models to **Pydantic v2** (`ConfigDict(frozen=True)`), introducing structured audit schemas:
* `ErrorClassification`: Encapsulates `error_type` (`"knowledge" | "reasoning"`), scalar `confidence`, and the complete softmax distribution dictionary.
* `NLIConsensusResult` / `ConflictAwareNLIResult`: Contains `decision`, continuous `consensus_score`, `evidence_count`, and a full list of `per_evidence_scores`.
* Updated `TriageResult`: Enriched with `error_classification`, `surrogate_model`, and `probe_latency_ms`.
* Updated `PipelineResult`: Adds `layers_invoked` (`"layer1_only"` | `"layer1_layer2"` | `"layer1_layer3"`), `total_latency_ms`, and `tokens_generated`.

---

#### 1.4. `app/config.py` & `.env`
Centralizes all newly introduced algorithmic hyperparameters:
```python
class Settings(BaseSettings):
    # Probing & Surrogate Novelties
    probe_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    probe_hidden_dim: int = 2048
    safe_threshold: float = 0.55             # DID_SAFE_THRESHOLD

    # Conflict-Aware NLI Novelties
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"
    nli_accept_threshold: float = 0.30       # DID_NLI_ACCEPT_THRESHOLD
    nli_reject_threshold: float = -0.30      # DID_NLI_REJECT_THRESHOLD
    max_search_results: int = 5              # DID_MAX_SEARCH_RESULTS

    # Timeout & Fallback Flags
    request_timeout_seconds: float = 120.0
    enable_surrogate_fallback: bool = True
```

---

### 2. End-to-End Execution Flow Trace

The diagram below traces an incoming query through every data transformation junction:

```
[Client]
   │  POST /api/v1/query {"prompt": "Where did fortune cookies originate?", "include_audit": true}
   ▼
[FastAPI Gateway: app/main.py]
   │  Parses JSON payload into Pydantic v2 QueryRequest
   ▼
[Orchestrator: app/router.py -> process_query()]
   │  Step 1: Local Ollama Generation (Meta Llama-3.1:8B)
   │  Output: draft_text (str, 335 tokens, latency: 34.3s)
   ▼
[Surrogate Probing: app/layer1_triage/probe_model.py -> triage()]
   │  Step 2: Unified sequence u = [prompt || draft_text]
   │  Step 3: Forward hook on TinyLlama-1.1B (Layer 22)
   │  Tensor Extraction: h_t^(L) ∈ ℝ^(1 × 2048) (float16 on CUDA)
   │  Step 4: DualHeadUncertaintyClassifier(h_t^(L))
   │  Head 1 Output: H_sem = 0.4920 (float ∈ [0, 1])
   │  Head 2 Output: logits = [0.12, 0.88] -> P(reasoning) = 0.68, P(knowledge) = 0.32
   ▼
[Dynamic Routing Dispatch: app/router.py]
   │  Evaluation: H_sem (0.4920) < T_safe (0.55)
   │  Decision: FAST PATH TRIGGERED!
   │  Action: Downstream Layer 2 (RAG) and Layer 3 (HalluClean) bypassed.
   ▼
[Response Compilation: app/models/schemas.py -> QueryResponse]
   │  Compiles QueryResponse with complete PipelineAudit tree
   ▼
[Client Return]
   HTTP 200 OK | Total Latency: 56.6s | Tokens: 335 | Layers Invoked: "layer1_only"
```

---

### 3. Resilience & Asynchronous Concurrency Design

1. **Non-Blocking Inference via `asyncio.to_thread`:**
   PyTorch tensor operations on CUDA are inherently synchronous and CPU-bound during graph setup. Wrapping `model.forward()` and `cross_encoder.predict()` inside `asyncio.to_thread` yields the Python GIL, allowing FastAPI to continue handling concurrent HTTP requests.
2. **Graceful Retriever Degradation:**
   If DuckDuckGo search rate-limits or returns empty results, `search_retriever.py` catches the network exception, records a warning in the audit log, and falls back to local ChromaDB vector collections without aborting the query.
3. **Pydantic v2 Schema Immutability:**
   Using `ConfigDict(frozen=True)` prevents inadvertent side-effects and race conditions across concurrent async tasks modifying shared state.
