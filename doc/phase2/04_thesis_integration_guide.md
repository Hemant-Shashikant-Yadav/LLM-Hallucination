# M.Tech Thesis Integration Guide: LaTeX Templates & Chapters
## Direct Insertion Reference for `main.tex` and Thesis Chapters

This document provides drop-in LaTeX code blocks, mathematical equations, algorithm blocks, table environments, figure inclusions, and BibTeX citations for direct use in the M.Tech thesis document (`main.tex`).

---

### 1. LaTeX Methodology Text & Equations (Chapter 3)

Copy and paste the following subsection directly into `main.tex` under Section 2 ("Proposed Methodology & Pure Python Architecture"):

```latex
\subsection{Surrogate Proxy Probing for Quantized LLMs}
Direct extraction of intermediate representations from quantized local generators ($G$) is precluded by inference abstraction layers. To preserve white-box probing capabilities without full-precision model deployment, we introduce a \textit{Surrogate Proxy Probing} architecture. 

Given user query $x$ and generator draft $y = G(x)$, we construct the unified evaluation sequence:
\begin{equation}
\mathbf{u} = [x \parallel y] = (u_1, u_2, \dots, u_M)
\end{equation}

The sequence $\mathbf{u}$ is fed to a white-box surrogate model $S_\phi$ ($\text{TinyLlama-1.1B}$, dimension $d = 2048$, $L = 22$ layers). A forward-pass hook registered on the terminal transformer layer $L$ intercepts the hidden state tensor:
\begin{equation}
\mathbf{H}^{(L)} = S_\phi^{(L)}(\mathbf{u}) \in \mathbb{R}^{M \times d}
\end{equation}

The final-token activation vector $h^* \in \mathbb{R}^d$ is pooled as the latent uncertainty signature:
\begin{equation}
h^* = \mathbf{H}^{(L)}[M, :] = h_M^{(L)}
\end{equation}

\subsection{Dual-Head Uncertainty Classifier and Dynamic Triage}
Rather than relying on lexical pattern heuristics, error diagnosis is framed as a multi-task learning problem over $h^*$. A shared representation layer feeds two dedicated heads:

\textbf{Head 1 (Semantic Entropy Estimator):} Parameterizes continuous epistemic uncertainty $H_{sem} \in [0, 1]$ via a logistic sigmoid transformation:
\begin{equation}
H_{sem}(h^*) = \sigma\left(\mathbf{W}_1 h^* + b_1\right) = \frac{1}{1 + e^{-(\mathbf{W}_1 h^* + b_1)}}
\end{equation}
where $\mathbf{W}_1 \in \mathbb{R}^{1 \times d}$ and $b_1 \in \mathbb{R}$.

\textbf{Head 2 (Categorical Logit Router):} Projects $h^*$ into a 2-dimensional failure-mode manifold distinguishing factual knowledge gaps ($k=0$) from compositional reasoning fallacies ($k=1$):
\begin{equation}
\mathbf{z} = \mathbf{W}_2 h^* + \mathbf{b}_2 \in \mathbb{R}^2, \quad P(\text{class} = k \mid h^*) = \frac{e^{z_k}}{\sum_{j \in \{0, 1\}} e^{z_j}}
\end{equation}

Let $T_{safe}$ represent the calibrated safety margin ($T_{safe} = 0.55$). The dynamic dispatch policy $\Pi(x, y)$ is formulated as:
\begin{equation}
\Pi(x, y) = \begin{cases} 
\mathbf{FastPath} \ (\text{Deliver Draft}), & \text{if } H_{sem}(h^*) < T_{safe} \\ 
\mathbf{Layer\ 2\ (NLI\text{-}RAG)}, & \text{if } H_{sem}(h^*) \ge T_{safe} \;\wedge\; c^* = \text{knowledge} \\ 
\mathbf{Layer\ 3\ (HalluClean)}, & \text{if } H_{sem}(h^*) \ge T_{safe} \;\wedge\; c^* = \text{reasoning} 
\end{cases}
\end{equation}

\subsection{Multi-Evidence Consensus NLI Verification}
For queries routed to Layer 2, the draft $y$ is decomposed into a set of atomic propositions $\mathcal{P} = \{p_1, \dots, p_n\}$. For each claim $p_i$ and retrieved evidence documents $E_i = \{e_{i,1}, \dots, e_{i,m}\}$, a cross-encoder $\text{DeBERTa-v3}$ computes directional entailment and contradiction probabilities. The net polarity is defined as:
\begin{equation}
\Delta(p_i, e_{i,j}) = P(\text{entailment} \mid p_i, e_{i,j}) - P(\text{contradiction} \mid p_i, e_{i,j}) \in [-1, 1]
\end{equation}

The multi-source consensus score $S_{consensus}(p_i)$ aggregates over all retrieved evidence:
\begin{equation}
S_{consensus}(p_i) = \frac{1}{|E_i|} \sum_{e \in E_i} \left[ P(\text{entailment} \mid p_i, e) - P(\text{contradiction} \mid p_i, e) \right]
\end{equation}

The tri-state decision rule is defined with acceptance threshold $T_{accept} = +0.3$ and rejection threshold $T_{reject} = -0.3$:
\begin{equation}
\mathcal{D}(p_i) = \begin{cases} 
\mathbf{Entailed} \ (\text{Verified}), & S_{consensus}(p_i) \ge T_{accept} \\ 
\mathbf{Contradicted} \ (\text{Hallucination}), & S_{consensus}(p_i) \le T_{reject} \\ 
\mathbf{Neutral} \ (\text{Inconclusive}), & T_{reject} < S_{consensus}(p_i) < T_{accept} 
\end{cases}
\end{equation}
```

---

### 2. Algorithm Pseudocode Block

Add this algorithm environment to the methodology section:

```latex
\begin{algorithm}[H]
\caption{Defense-in-Depth Latency-Optimal Hallucination Mitigation}
\label{alg:defense_in_depth}
\KwIn{Query $x$, Quantized Generator $G$, Surrogate Model $S_\phi$, Thresholds $T_{safe}, T_{accept}, T_{reject}$}
\KwOut{Verified Response $y^*$, Audit Metadata $\mathcal{M}$}

$y \leftarrow G(x)$ \tcp*{Generate initial draft from local LLM}
$\mathbf{u} \leftarrow [x \parallel y]$ \tcp*{Concatenate query and draft}
$h^* \leftarrow \text{ExtractTerminalHook}(S_\phi, \mathbf{u})$ \tcp*{Surrogate hidden vector}

$H_{sem} \leftarrow \sigma(\mathbf{W}_1 h^* + b_1)$\;
$c^* \leftarrow \arg\max \text{Softmax}(\mathbf{W}_2 h^* + \mathbf{b}_2)$\;

\If{$H_{sem} < T_{safe}$}{
    \Return{$y$, \text{Metadata}(path="FastPath", $H_{sem}$)} \tcp*{Bypass heavy verification}
}

\If{$c^* == \text{"knowledge"}$}{
    $\mathcal{P} \leftarrow \text{DecomposePropositions}(y)$\;
    \ForEach{$p_i \in \mathcal{P}$}{
        $E_i \leftarrow \text{RetrieveEvidence}(p_i)$\;
        $S_{consensus}(p_i) \leftarrow \frac{1}{|E_i|} \sum_{e \in E_i} [P(ent \mid p_i, e) - P(con \mid p_i, e)]$\;
        $\mathcal{D}(p_i) \leftarrow \text{TriStatePolicy}(S_{consensus}(p_i), T_{accept}, T_{reject})$\;
    }
    $y^* \leftarrow \text{RefineWithConstraints}(y, \mathcal{P}, \mathcal{D})$\;
    \Return{$y^*$, \text{Metadata}(path="Layer2\_NLIRAG", $H_{sem}$)}\;
}
\Else{
    $\mathcal{G} \leftarrow \text{GenerateTaskPlan}(x, y)$ \tcp*{Symbolic HalluClean DAG}
    $y^* \leftarrow \text{ExecutePlanAndJudge}(\mathcal{G})$\;
    \Return{$y^*$, \text{Metadata}(path="Layer3\_HalluClean", $H_{sem}$)}\;
}
\end{algorithm}
```

---

### 3. LaTeX Results Table Templates (Chapter 5)

```latex
\begin{table}[htbp]
\centering
\small
\caption{Empirical Performance Comparison across Evaluation Configurations}
\label{tab:evaluation_comparison}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
\textbf{Configuration} & \multicolumn{3}{c}{\textbf{TruthfulQA (N=30)}} & \multicolumn{3}{c}{\textbf{GSM8K (N=30)}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
 & \textbf{Acc (\%)} & \textbf{Lat (s)} & \textbf{Tokens} & \textbf{Acc (\%)} & \textbf{Lat (s)} & \textbf{Tokens} \\
\midrule
Raw LLM (Baseline) & 53.3 & 34.2 & 345 & 46.7 & 32.1 & 312 \\
Naive RAG (Layer 2) & 70.0 & 82.5 & 680 & 50.0 & 79.4 & 645 \\
\textbf{Proposed Defense-in-Depth} & \textbf{83.3} & \textbf{48.6} & \textbf{410} & \textbf{76.7} & \textbf{61.2} & \textbf{495} \\
\bottomrule
\end{tabular*}
\vspace{1mm}
\footnotesize{\textit{Note: Acc: Factual Accuracy (\%); Lat: Wall-Clock Latency in seconds; Tokens: Average tokens generated per query.}}
\end{table}
```

---

### 4. LaTeX Figure Inclusions

```latex
\begin{figure}[htbp]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{evaluation/plots/pareto_efficiency.png}
    \caption{Pareto Efficiency Frontier: Latency vs. Accuracy trade-off across evaluation configurations.}
    \label{fig:pareto_efficiency}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{evaluation/plots/token_overhead.png}
    \caption{Token Expenditure Overhead comparison showing the efficiency of Fast Path triage.}
    \label{fig:token_overhead}
\end{minipage}
\end{figure}
```

---

### 5. Primary BibTeX References

Add the following entries to your BibTeX bibliography file (`references.bib`):

```bibtex
@article{kuhn2023semantic,
  title={Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Large Language Models},
  author={Kuhn, Lorenz and Gal, Yarin and Farquhar, Sebastian},
  journal={International Conference on Learning Representations (ICLR)},
  year={2023}
}

@article{lin2021truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  journal={Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2022}
}

@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}

@article{zhang2024tinyllama,
  title={TinyLlama: An Open-Source Small Language Model},
  author={Zhang, Peiyuan and Zeng, Guangtao and Wang, Tianduo and Lu, Wei},
  journal={arXiv preprint arXiv:2401.02385},
  year={2024}
}

@article{he2021debertav3,
  title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing},
  author={He, Pengcheng and Gao, Jianfeng and Chen, Weizhu},
  journal={International Conference on Learning Representations (ICLR)},
  year={2023}
}

@article{lewis2020rag,
  title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={33},
  pages={9459--9474},
  year={2020}
}
```
