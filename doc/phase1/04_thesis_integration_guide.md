# M.Tech Thesis Integration Guide: Phase 1 Baseline & Limitations
## Direct Insertion Reference for `main.tex` (Chapters 3 & 4)

This document provides drop-in LaTeX code blocks, mathematical equations, system flowcharts, and research gap critiques detailing the **Phase 1 Baseline** for direct inclusion in the M.Tech thesis document (`main.tex`).

---

### 1. LaTeX Baseline Architecture Text & Equations (Chapter 3)

Insert the following text under Section 2 ("Proposed Methodology & Pure Python Architecture") to formally describe the initial baseline:

```latex
\section{Baseline Defense-in-Depth Architecture (Phase 1)}
To address the latency-cost-precision trade-offs inherent in existing hallucination mitigation strategies, we initially constructed an asynchronous, multi-layer defense framework operating within a pure-Python microservice architecture (FastAPI).

\subsection{Internal Triage via Semantic Entropy Probes (SEPs)}
When an input prompt $x$ is dispatched to the generator $G$, an initial draft $y$ is produced autoregressively. Rather than treating the generator as an impenetrable black box, Layer 1 intercepts the intermediate hidden states from the final third of transformer layers:
\begin{equation}
h_t^{(l)} \in \mathbb{R}^d, \quad l \in \left[\frac{2}{3}L, L\right]
\end{equation}

A linear logistic estimator parameterizes token-level hallucination probability:
\begin{equation}
P(\text{Hallucination} \mid h_t^{(l)}) = \sigma\left(\mathbf{W}_{\text{SEP}} \cdot h_t^{(l)} + b_{\text{SEP}}\right)
\end{equation}
where $\sigma$ denotes the standard sigmoid activation function, $\mathbf{W}_{\text{SEP}} \in \mathbb{R}^{1 \times d}$ is the learned projection weight vector, and $b_{\text{SEP}} \in \mathbb{R}$ is the scalar bias.

The aggregated semantic entropy $H_{sem}$ across sequence length $T$ is computed as:
\begin{equation}
H_{sem} = -\frac{1}{T} \sum_{t=1}^T \left[ P_t \log P_t + (1 - P_t) \log (1 - P_t) \right]
\end{equation}

If $H_{sem} < T_{safe}$ ($T_{safe} = 0.55$), the sequence is classified as low-risk and returned immediately via the \textit{Fast Path}, bypassing downstream external verification.

\subsection{Hypothesis-Testing RAG with NLI Verification}
If $H_{sem} \ge T_{safe}$ and the failure mode is identified as an epistemic knowledge gap, Layer 2 executes hypothesis-testing verification. The draft completion $y$ is decomposed into a set of discrete atomic propositions $\mathcal{P} = \{p_1, \dots, p_n\}$.

For each proposition $p_i$, external documents $E_i = \{e_{i,1}, \dots, e_{i,m}\}$ are retrieved via search APIs. A cross-encoder model ($\text{DeBERTa-v3}$) evaluates directional semantic entailment:
\begin{equation}
\mathbf{P}_{\text{NLI}}(p_i, e) = \left[ P(\text{Entailment}), P(\text{Neutral}), P(\text{Contradiction}) \right]
\end{equation}

If $P(\text{Contradiction}) \ge \tau_{con}$, negative constraints are synthesized to prompt the generator to correct the erroneous claim.

\subsection{Structured Reasoning via the HalluClean Paradigm}
When high uncertainty is attributed to multi-step reasoning failures, the query is routed to Layer 3 (HalluClean), which decouples generation into a 4-stage Directed Acyclic Graph (DAG):
\begin{enumerate}
    \item \textbf{Stage 1 (Task-Oriented Planning):} Compiles the prompt into a dependency graph $\mathcal{G} = (V, E)$.
    \item \textbf{Stage 2 (Plan-Guided Reasoning):} Sequentially executes steps with isolated local context boundaries.
    \item \textbf{Stage 3 (Final Judgment):} Compares intermediate conclusions against global prompt constraints.
    \item \textbf{Stage 4 (Content Refinement):} Formulates the verified reasoning path into natural language.
\end{enumerate}
```

---

### 2. LaTeX Critical Gap Analysis (Chapter 4: Motivation for Phase 2)

Insert the following subsection into Chapter 4 to establish the theoretical justification for the Phase 2 improvisations:

```latex
\section{Limitations of the Baseline Architecture and Motivation for Phase 2}
Empirical profiling of the Phase 1 prototype revealed three fundamental structural deficiencies that impaired its operational validity:

\subsection{The Opaque Quantized Generator Dilemma}
In practical, resource-constrained deployments, the primary generator (e.g., Llama-3.1-8B) is served in quantized 4-bit representation (AWQ/GGUF) via local inference engines such as Ollama. These runtimes discard intermediate activations and expose only high-level HTTP endpoints. Consequently, the baseline PyTorch forward hooks could not intercept $h_t^{(l)}$ from the quantized generator, reducing the internal triage layer to mocked heuristics.

\subsection{Surface Lexical Routing vs. Manifold Geometry}
The Phase 1 router differentiated between knowledge gaps (Layer 2) and reasoning fallacies (Layer 3) using static substring matching (e.g., checking for keywords such as \texttt{"calculate"}, \texttt{"solve"}, \texttt{"step"}). This heuristic frequently misrouted factual queries containing math terms to Layer 3, and routed subtle compositional logic puzzles lacking explicit keywords to external search.

\subsection{Variance in Single-Pair NLI Verification}
Phase 1 evaluated proposition-evidence pairs individually. Due to the inherent noise and topical variance of external search snippets, a single spurious contradiction score ($P(\text{con}) > 0.50$) triggered false-positive hallucination flags, causing destructive overwriting of factually sound completions.

These three bottlenecks directly motivated the Phase 2 research novelties: \textit{Surrogate Proxy Probing}, the \textit{Dual-Head Uncertainty Classifier}, and the \textit{Consensus-Driven NLI Decision Rule}.
```

---

### 3. LaTeX Comparative Table Template

```latex
\begin{table}[htbp]
\centering
\small
\caption{Architectural Evolution: Baseline Prototype (Phase 1) vs. Proposed Improvised Framework (Phase 2)}
\label{tab:phase1_vs_phase2}
\begin{tabularx}{\textwidth}{lXX}
\toprule
\textbf{Architectural Dimension} & \textbf{Phase 1: Baseline Prototype} & \textbf{Phase 2: Proposed Improvised Framework} \\
\midrule
\textbf{Layer 1 Activation Probing} & Assumed unquantized in-memory PyTorch generator; failed on quantized Ollama APIs. & \textbf{Surrogate Proxy Probing:} TinyLlama-1.1B forward hook capturing terminal hidden vector $h^* \in \mathbb{R}^{2048}$. \\
\addlinespace
\textbf{Error-Type Routing Policy} & Brittle prompt keyword matching (\texttt{"calculate"}, \texttt{"solve"}). & \textbf{Dual-Head Classifier:} Multi-task linear manifold projecting $h^*$ to entropy $H_{sem}$ and error logits $\mathbf{z} \in \mathbb{R}^2$. \\
\addlinespace
\textbf{Layer 2 NLI Verification} & Binary single-pair thresholding; highly sensitive to retriever noise. & \textbf{Consensus Aggregation:} Bounded net polarity metric $S_{consensus}(p_i) \in [-1, 1]$ across multi-evidence sets. \\
\addlinespace
\textbf{Schema Validation} & Legacy Pydantic models with minimal audit metadata. & Strict \textbf{Pydantic v2} immutable models with comprehensive timing and layer traces. \\
\addlinespace
\textbf{Evaluation Rigor} & Ad-hoc manual verification scripts. & Automated async benchmarking on \textbf{TruthfulQA} ($N=30$) and \textbf{GSM8K} ($N=30$) with LaTeX tables. \\
\bottomrule
\end{tabularx}
\end{table}
```
