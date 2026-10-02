# Theoretical & Academic Novelties: Phase 2 Defense-in-Depth Framework
## M.Tech Thesis Chapter 3 (Methodology) & Chapter 4 (System Design)

---

### 1. Motivation & Research Gaps Addressed

In safety-critical deployments, Large Language Models (LLMs) parameterized by $\theta$ suffer from factual confabulations and deductive breakdowns. While Phase 1 established a baseline three-tier defense hierarchy, its empirical and methodological foundations exhibited three severe shortcomings that precluded peer-reviewed academic publication:

#### 1.1. The Quantization/Black-Box Paradox
In contemporary research and production environments, resource constraints necessitate hosting primary generative models ($G$) using quantized 4-bit weights (AWQ / GGUF) via specialized inference engines like Ollama or vLLM. 
* **The Failure Mechanism:** These C++ runtimes (`llama.cpp`) execute quantized matrix operations natively and expose only high-level network APIs (`POST /api/generate`). Intermediate transformer activations, attention maps, and continuous token representations ($h_t^{(l)} \in \mathbb{R}^d$) are discarded inside the compiled runtime and are strictly inaccessible over HTTP.
* **The Paradox:** True internal activation probing (e.g., Semantic Entropy Probes, linear truth probes) mathematically requires tensor access to hidden states. In Phase 1, the triage probe could not access activations from the quantized Ollama daemon, reducing the internal triage layer to mock values or falling back to uncalibrated output logit heuristics.

#### 1.2. Superficial Lexical Routing vs. Latent Manifold Geometry
In Phase 1, routing between knowledge retrieval (Layer 2) and symbolic reasoning (Layer 3) was performed using brittle regular expression keyword matching on the raw prompt string (`if "calculate" in prompt: route_layer3`).
* **The Failure Mechanism:** Lexical heuristics fail to capture the latent semantics of model uncertainty. Factual questions containing arithmetic terminology (e.g., *"Calculate the orbital velocity of the International Space Station"*) were erroneously classified as reasoning problems, while subtle compositional deduction riddles lacking trigger keywords were routed to web search.

#### 1.3. Short-Circuiting Fragility in Single-Passage NLI Verification
Phase 1 evaluated atomic propositions against retrieved web documents sequentially and short-circuited on the first match (`break` on contradiction).
* **The Failure Mechanism:** External search retrievers (DuckDuckGo, vector databases) frequently surface contradictory, colloquial, or irrelevant text snippets. A single noisy snippet yielding $P(\text{contradiction}) > 0.50$ caused the framework to unconditionally reject a factually sound proposition, causing catastrophic error propagation and hallucinated regenerations.

---

### 2. Mathematical Formalisms of Phase 2 Novelties

To resolve these research gaps, Phase 2 implements three core algorithmic novelties:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Phase 2 Algorithmic Novelties Pipeline                          │
│                                                                                        │
│  User Query (x) ──► Quantized Generator G (Ollama Llama-3.1:8B) ──► Draft Response (y) │
│                                                                             │          │
│  ┌──────────────────────────────────────────────────────────────────────────┘          │
│  ▼                                                                                     │
│  [Novelty 1: Surrogate Proxy Probing]                                                  │
│  Unified Sequence: u = [x || y]                                                        │
│  Forward Hook on TinyLlama-1.1B (Layer L) ──► Latent Vector h_t^(L) in R^2048          │
│                                                              │                         │
│  ┌───────────────────────────────────────────────────────────┘                         │
│  ▼                                                                                     │
│  [Novelty 2: Dual-Head Uncertainty Classifier]                                         │
│  Shared Projection Layer Norm: feat = LayerNorm(W_s h + b_s)                            │
│  ├── Head 1: H_sem = sigma(W_1 feat + b_1) in [0, 1]                                   │
│  └── Head 2: y_hat_type = Softmax(W_2 feat + b_2) in R^2 (Knowledge vs. Reasoning)     │
│                                                                                        │
│  Decision Dispatch:                                                                    │
│  ├── If H_sem < T_safe (0.55) ──────────────► FAST PATH (Bypass external verification) │
│  └── If H_sem >= T_safe (0.55):                                                        │
│      ├── If y_hat_type == Knowledge ────────► Layer 2: Conflict-Aware NLI-RAG          │
│      └── If y_hat_type == Reasoning ────────► Layer 3: HalluClean Symbolic DAG         │
│                                                                                        │
│  [Novelty 3: Conflict-Aware NLI Formulation] (Layer 2)                                 │
│  Multi-evidence consensus score:                                                       │
│  S(p_i) = sum_j w_j [P(entail)_j - P(contradict)_j] / sum_j w_j in [-1, +1]            │
│  Bounded Decision: tau_accept (+0.3) vs. tau_reject (-0.3)                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 2.1. Novelty 1: Surrogate Proxy Probing
Let $G$ be the opaque, 4-bit quantized generator producing draft completion $y$ for query $x$:
$$y = G(x)$$

We define an open-weights, white-box surrogate model $S_\phi$ sharing structural and tokenization lineage with the generator (specifically `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, dimension $d = 2048$, $L = 22$ transformer decoder layers).

We construct the unified causal context sequence $\mathbf{u}$ by concatenating the prompt $x$ and the generated draft $y$:
$$\mathbf{u} = x \oplus y = (u_1, u_2, \dots, u_M), \quad u_m \in \mathcal{V}$$

The unified sequence $\mathbf{u}$ is fed to $S_\phi$ in a single non-autoregressive forward evaluation pass. We register a PyTorch forward hook on the terminal transformer decoder layer $L$:
$$\mathbf{H}^{(L)} = S_\phi^{(L)}(\mathbf{u}) \in \mathbb{R}^{M \times d}$$

The latent representation vector $h_t^{(L)} \in \mathbb{R}^d$ is pooled at the sequence termination index $M$:
$$h_t^{(L)} = \mathbf{H}^{(L)}[M, :] = \text{SurrogateEncoder}(x \oplus y)$$

*Theoretical Justification:* Because modern autoregressive decoders (Llama-3, TinyLlama) map semantic consistency into geometric manifolds within the upper transformer layers, passing the generated draft $y$ conditioned on prompt $x$ through $S_\phi$ induces a surrogate representation $h_t^{(L)}$ whose latent geometry reflects epistemic uncertainty, factual consistency, and semantic cohesion.

#### 2.2. Novelty 2: Dual-Head Uncertainty Classification
Rather than employing disparate heuristics, Phase 2 implements a multi-task neural head operating over the surrogate hidden vector $h_t^{(L)} \in \mathbb{R}^{2048}$.

A shared projection manifold projects the high-dimensional hidden state:
$$\mathbf{f} = \text{Dropout}\left(\text{ReLU}\left(\text{LayerNorm}\left(\mathbf{W}_s h_t^{(L)} + \mathbf{b}_s\right)\right)\right) \in \mathbb{R}^{d_{hidden}}$$
where $d_{hidden} = 256$, $\mathbf{W}_s \in \mathbb{R}^{d_{hidden} \times d}$, and $\mathbf{b}_s \in \mathbb{R}^{d_{hidden}}$.

##### Head 1: Continuous Semantic Entropy ($H_{sem}$)
Head 1 estimates continuous semantic entropy $H_{sem} \in [0, 1]$, quantifying the dispersion of semantic equivalence classes:
$$H_{sem} = \sigma\left(\mathbf{W}_1 \mathbf{f} + b_1\right) = \frac{1}{1 + e^{-(\mathbf{W}_1 \mathbf{f} + b_1)}}$$
where $\mathbf{W}_1 \in \mathbb{R}^{1 \times d_{hidden}}$ and $b_1 \in \mathbb{R}$.

##### Head 2: Categorical Error-Type Router
Head 2 parameterizes a linear projection into a 2-dimensional failure-mode manifold distinguishing **Factual Knowledge Gaps** ($k = 0$) from **Compositional Reasoning Fallacies** ($k = 1$):
$$\mathbf{z} = \mathbf{W}_2 \mathbf{f} + \mathbf{b}_2 \in \mathbb{R}^2$$
$$\hat{\mathbf{y}}_{\text{type}} = \text{Softmax}(\mathbf{z}) = \left[ P(\text{knowledge} \mid h_t^{(L)}), \ P(\text{reasoning} \mid h_t^{(L)}) \right]$$
where:
$$P(\text{class} = k \mid h_t^{(L)}) = \frac{e^{z_k}}{\sum_{j \in \{0, 1\}} e^{z_j}}$$

##### Joint Multi-Task Training Objective
During offline probe calibration, the dual-head module is optimized via joint empirical risk minimization:
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{BCE}}\left(H_{sem}, y_{\text{entropy}}\right) + (1 - \alpha) \mathcal{L}_{\text{CE}}\left(\hat{\mathbf{y}}_{\text{type}}, y_{\text{type}}\right)$$
where $\mathcal{L}_{\text{BCE}}$ is the Binary Cross-Entropy loss over continuous uncertainty labels, $\mathcal{L}_{\text{CE}}$ is the Categorical Cross-Entropy loss over failure modes, and $\alpha \in [0, 1]$ is a Pareto weighting hyperparameter (default $\alpha = 0.5$).

##### Dynamic Routing Policy
Let $T_{safe}$ denote the calibrated risk threshold (default $0.55$). The dynamic dispatch policy $\Pi(x, y)$ is formulated as:
$$\Pi(x, y) = \begin{cases} 
\mathbf{FastPath} \ (\text{Deliver Draft } y), & \text{if } H_{sem} < T_{safe} \\ 
\mathbf{Layer\ 2\ (NLI\text{-}RAG)}, & \text{if } H_{sem} \ge T_{safe} \;\wedge\; \arg\max(\hat{\mathbf{y}}_{\text{type}}) = \text{knowledge} \\ 
\mathbf{Layer\ 3\ (HalluClean)}, & \text{if } H_{sem} \ge T_{safe} \;\wedge\; \arg\max(\hat{\mathbf{y}}_{\text{type}}) = \text{reasoning} 
\end{cases}$$

#### 2.3. Novelty 3: Conflict-Aware NLI Formulation
For queries routed to Layer 2, the draft response $y$ is decomposed into a set of independent atomic propositions $\mathcal{P} = \{p_1, p_2, \dots, p_n\}$.

For each proposition $p_i$, a multi-source evidence set $E_i = \{e_{i,1}, e_{i,2}, \dots, e_{i,m}\}$ is retrieved via external search. Rather than short-circuiting on the first passage, a cross-encoder model ($\text{DeBERTa-v3}$) evaluates all $m$ evidence passages exhaustively:
$$(P_{\text{ent}, j}, P_{\text{neu}, j}, P_{\text{con}, j}) = \text{DeBERTa}(p_i, e_{i,j})$$

##### Weighted Net Consensus Score
Let $w_j \in \mathbb{R}^+$ denote the relevance weight of passage $e_{i,j}$ (derived from dense similarity or retriever rank). We formulate the **Conflict-Aware Consensus Metric** $S(p_i)$ as:
$$S(p_i) = \frac{\sum_{j=1}^m w_j \left[ P(\text{entailment} \mid p_i, e_{i,j}) - P(\text{contradiction} \mid p_i, e_{i,j}) \right]}{\sum_{j=1}^m w_j} \in [-1, +1]$$

Under uniform passage weighting ($w_j = 1$):
$$S(p_i) = \frac{1}{|E_i|} \sum_{e \in E_i} \left[ P(\text{entailment} \mid p_i, e) - P(\text{contradiction} \mid p_i, e) \right]$$

##### Bounded Tri-State Decision Boundaries
Using calibrated acceptance and rejection thresholds $\tau_{\text{accept}} = +0.3$ and $\tau_{\text{reject}} = -0.3$:
$$\mathcal{D}(p_i) = \begin{cases} 
\mathbf{Entailed} \ (\text{Factual Grounding Confirmed}), & S(p_i) \ge \tau_{\text{accept}} \\ 
\mathbf{Contradicted} \ (\text{Hallucination Detected}), & S(p_i) \le \tau_{\text{reject}} \\ 
\mathbf{Neutral} \ (\text{Inconclusive Evidence}), & \tau_{\text{reject}} < S(p_i) < \tau_{\text{accept}} 
\end{cases}$$

*Resilience Mechanism:* If an irrelevant or adversarial search snippet exhibits $P(\text{con}) = 0.65$ but three corroborating sources exhibit $P(\text{ent}) \approx 0.85$, the net consensus score remains $S(p_i) = \frac{3(0.85) - 0.65}{4} = +0.475 \ge \tau_{\text{accept}}$, successfully preventing false-positive hallucination alerts.

---

### 3. Contrast Matrix: Phase 1 vs. Phase 2

| Dimension | Phase 1 (Baseline Prototype) | Phase 2 (Improvised Framework) | Academic Significance |
| :--- | :--- | :--- | :--- |
| **Probing Mechanism** | Assumed unquantized in-memory PyTorch generator; broke on Ollama | **Surrogate Proxy Probing:** White-box TinyLlama-1.1B hook capturing $h_t^{(L)} \in \mathbb{R}^{2048}$ | Solves the Quantization/Black-Box paradox for all Ollama/vLLM deployments |
| **Error Triage Policy** | Brittle prompt keyword matching (`"calculate"`, `"solve"`) | **Dual-Head Linear Classifier:** Latent manifold projecting $h_t^{(L)}$ to $H_{sem}$ and $\hat{\mathbf{y}}_{\text{type}} \in \mathbb{R}^2$ | Eliminates lexical heuristics; routes based on true latent representation geometry |
| **Evidence Aggregation** | Short-circuiting single-pair check (`break` on first contradiction) | **Conflict-Aware Consensus Metric:** Weighted net polarity $S(p_i) \in [-1, +1]$ across all passages | Robust to web search contradictions, retriever noise, and adversarial snippets |
| **Decision Boundaries** | Ad-hoc uncalibrated threshold ($P > 0.50$) | Calibrated dual-threshold bounds ($\tau_{\text{accept}} = +0.3, \tau_{\text{reject}} = -0.3$) | Minimizes false-positive hallucination flags and prevents destructive overwrites |
| **Data Integrity & Schemas** | Loose dictionary returns with partial fields | Strict **Pydantic v2** immutable models (`ConfigDict(frozen=True)`) | Ensures auditability, microsecond latency logging, and strict type safety |
| **Empirical Rigor** | Mocked ad-hoc test runs | Async evaluation harness on **TruthfulQA** ($N=30$) and **GSM8K** ($N=30$) | Produces reproducible camera-ready LaTeX tables and 300-DPI Pareto curves |
