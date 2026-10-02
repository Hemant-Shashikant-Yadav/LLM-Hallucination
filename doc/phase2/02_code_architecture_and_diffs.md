# Code Architecture & Implementation Diffs: Phase 1 vs. Phase 2
## M.Tech Thesis Chapter 4: System Implementation & Software Architecture

---

### 1. Architectural Overview & Component Directory

Phase 2 transitions the framework from a conceptual proof-of-concept into a robust, type-safe, asynchronous microservice architecture compliant with **Pydantic v2** and **PyTorch 2.x**.

```
d:\Coding\Project\LLM-Hallucination
├── app/
│   ├── config.py                     # Centralized hyperparameter & threshold registry
│   ├── main.py                       # FastAPI entrypoint and lifespan resource loader
│   ├── router.py                     # Dynamic triage state machine (Fast Path, L2, L3)
│   ├── models/
│   │   └── schemas.py                # Pydantic v2 strict data schemas & audit models
│   ├── layer1_triage/
│   │   ├── probe_model.py            # SurrogateExtractor + DualHeadUncertaintyClassifier
│   │   └── activation_hook.py        # Tensor extraction helpers & PyTorch forward hooks
│   ├── layer2_nli_rag/
│   │   ├── proposition_parser.py     # Decomposes draft responses into atomic propositions
│   │   ├── search_retriever.py       # Async multi-source retriever (DuckDuckGo / ChromaDB)
│   │   └── nli_verifier.py           # Consensus-driven cross-encoder NLI (DeBERTa-v3)
│   └── layer3_halluclean/
│       ├── planner.py                # Symbolic task graph decomposition
│       ├── executor.py               # Step-by-step reasoning execution loop
│       └── judge.py                  # Formal constraint checking & output synthesis
├── evaluation/
│   ├── benchmark_runner.py           # Async 3-way evaluation harness (TruthfulQA + GSM8K)
│   └── plot_generator.py             # 300-DPI publication chart generator
└── doc/
    └── phase2/                       # Comprehensive M.Tech research documentation
```

---

### 2. File-by-File Technical Deep Dive & Git Diffs

#### 2.1. `app/layer1_triage/probe_model.py`
* **Phase 1 Problem:** Contained a placeholder logistic regression probe that attempted to extract activations directly from Ollama. Since Ollama’s REST API does not expose activations, the probe defaulted to random noise or mock values.
* **Phase 2 Implementation:**
  1. `SurrogateExtractor`: Loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` onto GPU (`device="cuda"`). Registers a forward hook on `model.model.layers[-1]` to capture the hidden activation tensor $h^* \in \mathbb{R}^{2048}$ at the sequence termination token without generating autoregressive tokens.
  2. `DualHeadUncertaintyClassifier`: Implements a multi-task PyTorch module with shared input dimension ($d=2048$), outputting continuous semantic entropy ($H_{sem} \in [0, 1]$) and categorical error logits ($z \in \mathbb{R}^2$).

##### Code Diff Highlights:
```python
# =============================================================================
# PHASE 2: DualHeadUncertaintyClassifier Implementation
# =============================================================================
class DualHeadUncertaintyClassifier(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.entropy_head = nn.Linear(hidden_dim, 1)   # Head 1: H_sem in [0, 1]
        self.router_head = nn.Linear(hidden_dim, 2)    # Head 2: Logits in R^2 (knowledge, reasoning)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(h)
        h_sem = self.sigmoid(self.entropy_head(feat))
        router_logits = self.router_head(feat)
        return h_sem, router_logits

# =============================================================================
# PHASE 2: Surrogate Forward Hook Extraction
# =============================================================================
def _register_hook(self) -> None:
    target_layer = self._model.model.layers[-1]
    def hook_fn(module, inputs, output):
        # output is tuple: (hidden_states, ...)
        hidden = output[0] if isinstance(output, tuple) else output
        self._final_hidden_state = hidden[:, -1, :].detach() # [B, d]
    self._hook_handle = target_layer.register_forward_hook(hook_fn)
```

---

#### 2.2. `app/layer2_nli_rag/nli_verifier.py`
* **Phase 1 Problem:** Evaluated isolated (proposition, evidence) pairs sequentially using binary thresholding. A single irrelevantly retrieved document could erroneously mark a valid claim as contradicted.
* **Phase 2 Implementation:**
  1. Implements multi-evidence consensus aggregation:
     $$S_{consensus}(p_i) = \frac{1}{|E|} \sum_{e \in E} [P(\text{entailment}) - P(\text{contradiction})]$$
  2. Implements calibrated tri-state thresholds ($T_{accept} = +0.3$, $T_{reject} = -0.3$).
  3. Batches DeBERTa-v3 cross-encoder evaluations on CUDA for maximum inference throughput.

##### Code Diff Highlights:
```python
# =============================================================================
# PHASE 2: Consensus Score & Tri-State Decision Rule
# =============================================================================
def compute_consensus(self, scores: list[dict[str, float]]) -> NLIConsensusResult:
    if not scores:
        return NLIConsensusResult(decision=NLIDecision.NEUTRAL, consensus_score=0.0)
    
    # Net evidence polarity per document: P(entailment) - P(contradiction)
    net_polarities = [s["entailment"] - s["contradiction"] for s in scores]
    consensus_score = float(sum(net_polarities) / len(net_polarities))
    
    if consensus_score >= settings.nli_accept_threshold:   # +0.3
        decision = NLIDecision.ENTAILED
    elif consensus_score <= settings.nli_reject_threshold: # -0.3
        decision = NLIDecision.CONTRADICTED
    else:
        decision = NLIDecision.NEUTRAL
        
    return NLIConsensusResult(
        decision=decision,
        consensus_score=round(consensus_score, 4),
        evidence_count=len(scores),
        per_evidence_scores=scores,
    )
```

---

#### 2.3. `app/models/schemas.py`
* **Phase 1 Problem:** Legacy Pydantic v1 patterns without rigorous nested structures for auditing internal probe activations or consensus distributions.
* **Phase 2 Implementation:**
  1. Updated strictly to **Pydantic v2** (`BaseModel`, `Field`, `ConfigDict(frozen=True)`).
  2. Added formal schema definitions for:
     * `ErrorClassification`: Encapsulating `error_type` ("knowledge" | "reasoning"), `confidence`, and categorical probability distributions.
     * `DualHeadProbeResult`: Storing `semantic_entropy`, `error_classification`, `surrogate_model`, and `inference_latency_ms`.
     * `NLIVerificationConsensus`: Tracking multi-evidence consensus scores, evidence counts, and per-document entailment/contradiction breakdowns.
     * `PipelineAudit`: Comprehensive end-to-end tracing schema recording wall-clock latency, token count, activated layer path, and triage rationale.

##### Code Diff Highlights:
```python
class ErrorClassification(BaseModel):
    model_config = ConfigDict(frozen=True)
    error_type: Literal["knowledge", "reasoning"] = Field(
        description="Predicted failure mode from surrogate hidden state probe"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(
        description="Full softmax distribution over categorical error types"
    )

class TriageResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    decision: TriageDecision
    semantic_entropy: float = Field(ge=0.0, le=1.0)
    hallucination_probability: float = Field(ge=0.0, le=1.0)
    error_classification: ErrorClassification | None = None
    surrogate_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    probe_latency_ms: float = 0.0
```

---

#### 2.4. `app/router.py`
* **Phase 1 Problem:** Hardcoded conditional branches based on text regexes (`if "calculate" in prompt: route_layer3`).
* **Phase 2 Implementation:**
  1. **Fast Path Routing:** If `triage_result.semantic_entropy < settings.safe_threshold` ($T_{safe} = 0.55$), the router marks the draft as secure, immediately returning the response without invoking external networks or search APIs.
  2. **Categorical Error Routing:** If $H_{sem} \ge T_{safe}$:
     * If `error_type == "knowledge"` $\rightarrow$ Routes to Layer 2 (Hypothesis RAG + NLI consensus).
     * If `error_type == "reasoning"` $\rightarrow$ Routes to Layer 3 (HalluClean symbolic graph executor).
  3. Integrated latency tracking using `time.perf_counter()` across every invoked sub-module.

##### Code Diff Highlights:
```python
# Fast Path: bypass all heavy external verification
if triage_result.decision == TriageDecision.SAFE:
    logger.info("FAST PATH | H_sem={:.4f} < T_safe={} | {:.1f}ms total",
                triage_result.semantic_entropy, self.settings.safe_threshold, wall_ms)
    return QueryResponse(
        answer=draft,
        pipeline_result=PipelineResult(
            query=request.prompt,
            final_answer=draft,
            layers_invoked="layer1_only",
            total_latency_ms=wall_ms,
            tokens_generated=len(draft.split()),
            triage_result=triage_result,
        ),
    )

# High uncertainty: route based on Head 2 classification
error_type = triage_result.error_classification.error_type
if error_type == "reasoning":
    return await self._process_layer3(request, draft, triage_result, t0)
else:
    return await self._process_layer2(request, draft, triage_result, t0)
```

---

#### 2.5. `evaluation/benchmark_runner.py` & `plot_generator.py`
* **Phase 1 Problem:** Manual ad-hoc test scripts without standard dataset loaders or automated statistical aggregations.
* **Phase 2 Implementation:**
  1. `benchmark_runner.py`:
     * Directly streams 30 validation samples from Hugging Face `truthfulqa/truthful_qa` (generation split) and 30 test samples from `gsm8k` (main split).
     * Automatically queries all three pipeline modes sequentially:
       * Mode 1: Baseline (`POST /api/v1/query/layer1`)
       * Mode 2: Naive RAG (`POST /api/v1/query/layer2`)
       * Mode 3: Defense-in-Depth (`POST /api/v1/query`)
     * Computes factual accuracy: substring overlap against key sets for TruthfulQA; numerical regex parsing for GSM8K.
     * Generates camera-ready LaTeX tables (`DataFrame.to_latex(booktabs=True)`) directly to stdout.
  2. `plot_generator.py`:
     * Generates publication-ready 300-DPI charts:
       * `evaluation/plots/pareto_efficiency.png` (Latency vs. Accuracy Pareto frontier).
       * `evaluation/plots/token_overhead.png` (Token overhead per query across configurations).

---

### 3. Summary of System Improvements

| Metric / Dimension | Phase 1 (Baseline) | Phase 2 (Improvised) |
| :--- | :--- | :--- |
| **Activation Probing** | Mocked/Opaque | Real PyTorch Hook on TinyLlama-1.1B ($d=2048$) |
| **Routing Mechanism** | Keyword regex matching | Dual-head latent manifold classification |
| **RAG Verification** | Binary single-pair check | Multi-evidence consensus metric $S_{consensus} \in [-1, 1]$ |
| **Data Integrity** | Loose dicts | Strict Pydantic v2 audited models |
| **Benchmark Suite** | None | Async evaluation on TruthfulQA & GSM8K |
| **Thesis Deliverables** | Ad-hoc text prints | Automated LaTeX tables & 300-DPI charts |
