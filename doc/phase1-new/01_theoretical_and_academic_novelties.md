# Theoretical & Academic Foundations: Phase 1 Baseline Architecture
## M.Tech Thesis Chapter 3: Proposed Methodology (Foundational Blueprint)
### Baseline Commit: `4187783ca0c74f98a71c956f9ae84a925b3552f9` (`feat(main): phase 1 implementaion`)

---

### 1. Research Motivation & The Multi-Tier Defense Premise

Large Language Models (LLMs) frequently generate plausible but ungrounded completions. When Phase 1 was formulated, existing research suffered from an operational dichotomy:
* **Passive Pre-generation Interventions:** Supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF) adjust broad statistical preferences but remain incapable of handling real-time epistemic distribution shifts.
* **Naive Retrieval-Augmented Generation (RAG):** Indiscriminately injects top-$k$ retrieved passages into the model's prompt for every query, regardless of whether the model already possesses factual conviction. This increases prompt tokens by $4\times\text{--}8\times$ and frequently injects irrelevant context into safe answers.
* **Multi-Agent Deliberation (Debate):** Employs multiple competing LLMs to critique each other's outputs, multiplying latency and GPU inference cost by orders of magnitude.

#### The Defense-in-Depth Conceptual Framework
Phase 1 posited that **hallucination mitigation should be adaptive and risk-proportional**:
1. **Layer 1 (Internal Triage):** Evaluate latent uncertainty within milliseconds using low-overhead internal probes.
2. **Layer 2 (Hypothesis-Testing RAG):** For factual knowledge uncertainty, break claims into atomic propositions and verify against retrieved evidence using Natural Language Inference (NLI).
3. **Layer 3 (Symbolic Reasoning):** For multi-step deductive errors, route to a 4-stage Directed Acyclic Graph (DAG) called HalluClean that isolates context boundaries to prevent error cascades.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Phase 1 Baseline Architectural Flow                             │
│                                                                                        │
│  User Query (x) ──► Autoregressive Generator G ──► Draft Response (y)                  │
│                                                             │                          │
│  ┌──────────────────────────────────────────────────────────┘                          │
│  ▼                                                                                     │
│  [Layer 1: Semantic Entropy Probe (SEP)]                                               │
│  Target Transformer Layer l in [2/3 L, L]                                              │
│  Hidden state extraction: h_t^(l) in R^d                                               │
│  P(Hallucination | h_t^(l)) = sigma(W_SEP . h_t^(l) + b_SEP)                           │
│  Sequence Entropy: H_sem = - (1/T) sum [P log P + (1-P) log(1-P)]                      │
│                                                                                        │
│  Decision Boundary:                                                                    │
│  ├── If H_sem < T_safe (0.55) ──────────────► FAST PATH (Deliver Draft y immediately)  │
│  └── If H_sem >= T_safe (0.55):                                                        │
│      ├── If prompt has arithmetic keywords ─► Layer 3: HalluClean Symbolic DAG         │
│      └── Otherwise ─────────────────────────► Layer 2: Hypothesis-Testing NLI-RAG      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Mathematical Formalisms of the Phase 1 Baseline

#### 2.1. Layer 1: Semantic Entropy Probes (SEPs)
Let the generative model be an autoregressive decoder parameterized by $\theta$. Given input prompt $x = (x_1, \dots, x_N)$ and draft generation $y = (y_1, \dots, y_T)$, the internal activation tensor at layer $l$ is denoted as:
$$\mathbf{H}^{(l)} = \left[ h_1^{(l)}, h_2^{(l)}, \dots, h_T^{(l)} \right], \quad h_t^{(l)} \in \mathbb{R}^d$$

Under the hypothesis of Azaria & Mitchell (2023) and Marks & Tegmark (2023), truthfulness is linearly decodable from representations in the final third of transformer layers ($l \in [\frac{2}{3}L, L]$).

Phase 1 defines a linear logistical estimator over the token representations:
$$P_t = P(\text{Hallucination} \mid h_t^{(l)}) = \sigma\left(\mathbf{W}_{\text{SEP}} \cdot h_t^{(l)} + b_{\text{SEP}}\right)$$
where $\sigma(z) = \frac{1}{1 + e^{-z}}$, $\mathbf{W}_{\text{SEP}} \in \mathbb{R}^{1 \times d}$, and $b_{\text{SEP}} \in \mathbb{R}$.

To compute the aggregate semantic uncertainty across the entire generated sequence, the token-level entropies are averaged:
$$H_{sem} = -\frac{1}{T} \sum_{t=1}^T \left[ P_t \log P_t + (1 - P_t) \log (1 - P_t) \right]$$

Applying safety threshold $T_{safe} = 0.55$:
$$\text{Decision}(y) = \begin{cases} 
\mathbf{Safe} \ (\text{Bypass External Layers}), & \text{if } H_{sem} < T_{safe} \\ 
\mathbf{Uncertain} \ (\text{Trigger Verification}), & \text{if } H_{sem} \ge T_{safe} 
\end{cases}$$

#### 2.2. Layer 2: Atomic Proposition NLI Hypothesis Testing
When $H_{sem} \ge T_{safe}$, Phase 1 rejects naive context stuffing and initiates hypothesis-driven verification:
1. **Atomic Decomposition:** The draft text $y$ is decomposed into a set of $n$ atomic claims $\mathcal{P} = \{p_1, \dots, p_n\}$.
2. **Targeted Retrieval:** For each $p_i$, a query $q(p_i)$ is dispatched to external search engines (DuckDuckGo API) or local vector collections (ChromaDB) to retrieve evidence passages $E_i = \{e_{i,1}, \dots, e_{i,m}\}$.
3. **Directional Cross-Encoder NLI:** A pre-trained cross-encoder ($\text{DeBERTa-v3}$) evaluates each (hypothesis, evidence) pair:
$$\mathbf{P}_{\text{NLI}}(p_i, e) = \left[ P(\text{contradiction}), P(\text{neutral}), P(\text{entailment}) \right]$$
4. **Short-Circuit Decision Rule:**
$$\text{Status}(p_i) = \begin{cases} 
\mathbf{Contradicted}, & \text{if } \exists e \in E_i \text{ such that } P(\text{contradiction} \mid p_i, e) \ge \tau_{\text{con}} \\ 
\mathbf{Entailed}, & \text{if } \exists e \in E_i \text{ such that } P(\text{entailment} \mid p_i, e) \ge \tau_{\text{ent}} \\ 
\mathbf{Neutral}, & \text{otherwise} 
\end{cases}$$
where $\tau_{\text{con}} = 0.50$ and $\tau_{\text{ent}} = 0.60$.

#### 2.3. Layer 3: HalluClean 4-Stage Symbolic DAG
When uncertainty is attributed to multi-step reasoning, Phase 1 routes the query to a 4-stage Directed Acyclic Graph (DAG) $\mathcal{G} = (V, E)$ to prevent error compounding:
* **Stage 1 (Planning):** $\mathcal{G} = \text{Planner}(x)$, generating sequential task nodes $\{v_1, \dots, v_k\}$.
* **Stage 2 (Plan-Guided Reasoning):** Each node $v_k$ is executed in topological order within an isolated context containing only prerequisite outputs.
* **Stage 3 (Constraint Judgment):** Synthesized conclusions are checked against original prompt constraints.
* **Stage 4 (Refinement):** A verified, polished answer is emitted.

---

### 3. Baseline Assumptions & Theoretical Strengths

1. **Theoretical Latency Advantage:** By introducing the Fast Path ($H_{sem} < T_{safe}$), Phase 1 established the theoretical principle that safe queries should exit early, bypassing heavy external retrieval.
2. **Decomposition Over Aggregation:** Breaking text into atomic propositions localized hallucination detection to individual factual claims rather than entire documents.
3. **Context Isolation in Reasoning:** HalluClean's topological context boundaries mathematically bounded error propagation in multi-step deductive chains.
