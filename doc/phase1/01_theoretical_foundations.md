# Theoretical Foundations: Baseline Defense-in-Depth Architecture (Phase 1)
## M.Tech Thesis Chapter 3: Proposed Methodology (Baseline Formulation)

---

### 1. Research Motivation & Theoretical Premise

The uncontrolled deployment of Large Language Models (LLMs) in high-stakes domains (healthcare, law, finance, engineering) is impeded by **hallucinations**—statistically fluent completions that diverge from established empirical facts or violate formal deductive logic.

In academic literature, mitigation strategies have historically operated in isolated silos:
1. **Pre-Training & Fine-Tuning Interventions:** Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) align global behavioral distributions but fail to prevent stochastic confabulations under novel prompt conditions.
2. **Naive Retrieval-Augmented Generation (RAG):** Blindly prepends retrieved documents to prompts, multiplying context length, introducing retrieval noise, and failing to address non-factual logical fallacies.
3. **Multi-Agent Deliberation & Debate Protocols:** Incur prohibitive runtime latency by generating multiple autoregressive token streams, multiplying GPU energy and compute costs by $5\times\text{--}10\times$.

#### The Defense-in-Depth Paradigm
Phase 1 introduced a **hierarchical, defense-in-depth model** based on the principle of **computational risk proportionality**:
* **Lightweight Internal Triage (Layer 1):** Fast, non-autoregressive probes evaluate model uncertainty from internal activations in milliseconds.
* **External Hypothesis Verification (Layer 2):** Claims flagged with high factual uncertainty are broken into atomic propositions and verified via external Natural Language Inference (NLI).
* **Symbolic Multi-Stage Reasoning (Layer 3):** Complex compositional queries are routed to a decoupled planning DAG (HalluClean) that prevents reasoning collapse through step-wise context isolation.

```
       Incoming User Query (x)
                 │
                 ▼
      [ Generator LLM (Draft y) ]
                 │
                 ▼
       Layer 1: Internal Triage
      Semantic Entropy Probe (SEP)
                 │
        ┌────────┴────────┐
   H_sem < T_safe    H_sem >= T_safe
        │                 │
        ▼                 ▼
   [ Fast Path ]     Is Knowledge or Reasoning?
  (Deliver Draft)         │
               ┌──────────┴──────────┐
          Knowledge              Reasoning
               │                     │
               ▼                     ▼
       Layer 2: NLI-RAG       Layer 3: HalluClean
      Hypothesis Testing     4-Stage Symbolic DAG
               │                     │
               └──────────┬──────────┘
                          ▼
               Verified Output (y*)
```

---

### 2. Layer 1 Formulation: Semantic Entropy Probes (SEPs)

#### Internal Representation Geometry
Modern transformer architectures construct high-dimensional contextual embeddings across sequential layers. Research demonstrates that earlier layers capture surface syntax, intermediate layers encode lexical semantics, and the final one-third of layers ($l \in [\frac{2}{3}L, L]$) encode factual conviction and truth-value representations.

Let $h_t^{(l)} \in \mathbb{R}^d$ denote the hidden-state activation vector at token position $t$ extracted from layer $l$ of the transformer generator.

#### Logistic Estimator Formulation
Phase 1 parameterizes a lightweight linear probe over the extracted activations:
$$P(\text{Hallucination} \mid h_t^{(l)}) = \sigma\left(\mathbf{W}_{\text{SEP}} \cdot h_t^{(l)} + b_{\text{SEP}}\right)$$
where:
* $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid function,
* $\mathbf{W}_{\text{SEP}} \in \mathbb{R}^{1 \times d}$ is the trained projection weight vector,
* $b_{\text{SEP}} \in \mathbb{R}$ is the learned scalar bias.

#### Semantic Entropy Metric ($H_{sem}$)
To quantify overall response uncertainty, the token-level probabilities are aggregated over the generated sequence of length $T$:
$$H_{sem} = -\frac{1}{T} \sum_{t=1}^T \left[ P_t \log P_t + (1 - P_t) \log (1 - P_t) \right]$$
where $P_t = P(\text{Hallucination} \mid h_t^{(l)})$.

#### Triage Decision Boundary
A pre-calibrated safety threshold $T_{safe}$ (default $0.55$) is applied:
$$\text{Decision}(y) = \begin{cases} 
\mathbf{Safe} \ (\text{Deliver immediately via Fast Path}), & \text{if } H_{sem} < T_{safe} \\ 
\mathbf{Uncertain} \ (\text{Trigger external mitigation}), & \text{if } H_{sem} \ge T_{safe} 
\end{cases}$$

---

### 3. Layer 2 Formulation: Hypothesis-Testing RAG with NLI Verification

When a draft is flagged as uncertain due to factual knowledge gaps, Phase 1 bypasses naive prompt stuffing and executes **Hypothesis-Testing RAG**.

#### Atomic Proposition (AP) Decomposition
The model draft $y$ is decomposed into a set of $n$ atomic propositions $\mathcal{P} = \{p_1, p_2, \dots, p_n\}$:
$$y \xrightarrow{\text{Decomposition}} \mathcal{P}$$
Each proposition $p_i$ is a minimal declarative assertion containing exactly one subject-predicate relationship.

#### External Evidence Retrieval
For each proposition $p_i$, a targeted retrieval query is formulated and dispatched to external search engines (DuckDuckGo API) or local vector collections (ChromaDB):
$$E_i = \text{Retrieve}(p_i) = \{e_{i,1}, e_{i,2}, \dots, e_{i,m}\}$$

#### Directional NLI Verification
A cross-encoder model (DeBERTa-v3) computes the directional semantic relationship between proposition $p_i$ (hypothesis) and retrieved snippet $e$ (premise):
$$\mathbf{P}_{\text{NLI}}(p_i, e) = \left[ P(\text{Entailment}), P(\text{Neutral}), P(\text{Contradiction}) \right]$$

The classification policy operates as:
$$\text{Status}(p_i) = \begin{cases} 
\mathbf{Entailed}, & \text{if } P(\text{Entailment}) \ge \tau_{\text{ent}} \\ 
\mathbf{Contradicted}, & \text{if } P(\text{Contradiction}) \ge \tau_{\text{con}} \\ 
\mathbf{Neutral}, & \text{otherwise} 
\end{cases}$$

#### Constrained Refinement Loop
* **Entailed:** The proposition is verified and preserved.
* **Contradicted:** The proposition is flagged as false, and an explicit negative constraint $\neg p_i$ is injected into the refinement prompt to instruct the generator to omit or correct the factual error.
* **Neutral:** The evidence is inconclusive, triggering an expanded search query.

---

### 4. Layer 3 Formulation: Structured Reasoning via HalluClean

When uncertainty stems from multi-step deduction, arithmetic, or logical consistency, external search cannot resolve the hallucination. Phase 1 introduces the **HalluClean 4-Stage Symbolic DAG**.

#### Stage 1: Task-Oriented Planning
The unstructured query $x$ is parsed into a directed acyclic graph $\mathcal{G} = (V, E)$, where each vertex $v_k \in V$ represents an atomic reasoning step, and directed edges $(v_j, v_k) \in E$ define causal dependencies:
$$\mathcal{G} = \text{Planner}(x)$$

#### Stage 2: Plan-Guided Sequential Execution
Reasoning steps are executed in topological order. To prevent the **accumulation of compounding hallucinations**, each step $v_k$ is executed within a strict local context boundary containing only its parent outputs:
$$\text{Output}(v_k) = \text{Executor}(v_k, \{\text{Output}(v_j) \mid (v_j, v_k) \in E\})$$

#### Stage 3: Constraint Judgment & Validation
The synthesized reasoning chain is formally evaluated against the original prompt constraints:
$$\text{Valid} = \text{Judge}(\mathcal{G}, x)$$
If any step violates logical consistency or numerical bounds, the judge generates a localized revision prompt targeting only the faulty sub-graph node.

#### Stage 4: Content Polish & Synthesis
The verified reasoning steps are synthesized into a coherent, natural language completion $y^*$.

---

### 5. Theoretical Cost & Latency Model

Let $C(\cdot)$ denote computational cost (measured in FLOPs or token count) and $L(\cdot)$ denote wall-clock latency.

In a baseline zero-shot LLM:
$$\mathbb{E}[L_{\text{baseline}}] = L_{\text{gen}}$$

In Naive RAG, external retrieval and prompt expansion are applied indiscriminately:
$$\mathbb{E}[L_{\text{naive}}] = L_{\text{retrieval}} + L_{\text{gen}}(\text{context\_size} \times 4)$$

In Phase 1 Defense-in-Depth, the expected latency is governed by the triage pass rate $\alpha = P(H_{sem} < T_{safe})$:
$$\mathbb{E}[L_{\text{DiD}}] = L_{\text{gen}} + L_{\text{probe}} + (1 - \alpha) \cdot \left[ \beta L_{\text{L2}} + (1 - \beta) L_{\text{L3}} \right]$$
where $\beta$ is the proportion of knowledge queries among uncertain prompts.

When $\alpha \approx 0.60\text{--}0.70$ (60% to 70% of routine domain queries are safe), the Defense-in-Depth model provides substantial amortized latency savings over unconditional verification.
