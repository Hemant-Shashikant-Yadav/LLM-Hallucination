# Phase 1 Presentation: Narration Script and Formula Guide

Companion to [Phase1_Uncertainty_Study.pptx](Phase1_Uncertainty_Study.pptx) (20 slides). It has four parts:

- **Part A: slide-by-slide script.** What is on each slide, what to say, and short answers to likely panel questions.
- **Part B: formulas and methods in detail.** Every formula behind the slides, explained in plain words with a small worked example, and matched to the code in `uncertainty/`.
- **Part C: quick glossary.** One line per term, for when you get stuck.
- **Part D: where the expected values come from.** Every published number behind the result slides, with links.

> **About the result numbers.** Slides 14–17 show **expected values**, marked "EXPECTED VALUES" on the slide. They are not measured in this study yet. Accuracy comes from published benchmark scores where a model card or paper reports one, and is estimated from those otherwise. AUROC comes from published semantic-entropy results. Four cells are real, from earlier stored runs: the accuracy and AUROC of qwen2.5:3b and llama2:7b on GSM8K and TruthfulQA. When you show those slides, say "expected, from published results", never "we measured". Part D lists the sources, so you can answer "where do these numbers come from?".

**Timing.** About 20–25 minutes in total. Give the methodology slides (8–11) and the results slides (14–17) the most time. Give the title, contents and references slides the least.

---

## Part A: Slide-by-slide script

### Slide 1: Title (≈ 30 s)

**On the slide:** "Phase 1: Measuring and Classifying LLM Uncertainty"; your name, guide and academic year.

**Say:**
"Good morning. My thesis is on reducing hallucinations in large language models. Today I present Phase 1, which covers the first two objectives. First, we measure how uncertain a language model is when it answers. Second, we identify what kind of uncertainty it has. Phase 2 will use these results to reduce hallucinations."

---

### Slide 2: Contents (≈ 20 s)

**Say:**
"I will start with the problem and what current research does. Then I cover the gap we address, our objectives and the Phase 1 plan. After that comes the method: how we measure and classify uncertainty. Then what we built, the evaluation setup and results for every model, and finally the limitations and the Phase 2 plan."

---

### Slide 3: Introduction (≈ 1 min)

**On the slide:** LLMs write fluent but sometimes wrong answers; factuality vs faithfulness; knowledge errors need evidence, logic errors need reasoning checks.

**Say:**
"Language models write answers that sound confident and fluent, but they can still be wrong. We call such wrong answers hallucinations. Two ideas are important here. **Factuality** means the answer agrees with real-world knowledge. **Faithfulness** means the answer agrees with the context or evidence the model was given.

Errors also come in different kinds. If the model does not know a fact, it needs evidence. If it knows the facts but reasons badly, it needs a careful step-by-step check. Our framework first screens how unsure the model is, and only then verifies the answer in the right way."

**If asked:**
- *Why not verify every answer?* Verification costs time and compute. If the model is sure and usually right, checking every answer wastes effort. Uncertainty tells us where to spend it.

---

### Slide 4: Problem Statement (≈ 1 min)

**Say:**
"Today's detection and mitigation methods mostly work separately. Retrieved evidence can be irrelevant, or it can conflict with what the model already knows. Generating many answers and checking every one the same way makes systems slow and expensive. So our research problem is to reduce factual and reasoning errors with one integrated system that checks only the uncertain answers, and checks each one in the way that suits it."

---

### Slide 5: Literature Review (≈ 1.5 min)

**On the slide:** a table of six approaches (key idea and limitation) and a summary box.

**Say:**
"This table shows how uncertainty is measured today.

- **Semantic entropy**, by Kuhn and by Farquhar, published in Nature in 2024, is the best-known method. We ask the same question several times and check whether the answers agree in meaning. Its limits: it needs five to ten answers, and it misses errors the model makes while it is sure.
- **Graph methods**, by Lin and colleagues, treat the answers as a similarity graph. They have the same sampling cost.
- **Token probability** methods look at how likely each word was. Low probability means unsure. But they mix up wording with doubt: a rare word gets a low probability even when the fact is right.
- **Self-evaluation** asks the model to rate its own answer. Small models tend to be over-confident.
- **SelfCheckGPT** compares each sentence with the other answers. It is slow for long answers.
- Newer work, by Hou and by Abbasi-Yadkori, separates *'the model does not know'* from *'the question is unclear'*. But it does not link the kind of doubt to a fix.

In short, most work gives one uncertainty number, on a few models, and does not say what kind of doubt it is."

**If asked:**
- *Why semantic entropy as "the existing algorithm"?* It is peer-reviewed (Nature 2024), well known, and works with any model that can be sampled. It is also what our pipeline's Layer 1 already uses.

---

### Slide 6: Research Gap (≈ 1 min)

**Say:**
"From the literature we see three gaps.

1. Methods are compared on few models and tasks. We compare **17 methods on 11 datasets and up to 12 models**. That is Objective 1.
2. A score says *how* unsure the model is, not *why*. We classify the kind of doubt into knowledge, reasoning, aleatoric or temporal. That is Objective 2.
3. The score is not linked to a fix. In our design, each type names its own fix. That is the bridge to Phase 2."

---

### Slide 7: Project Objectives (≈ 45 s)

**Say:**
"These are the four thesis objectives. Phase 1, which I present today, covers objectives 1 and 2: measuring baseline uncertainty with an existing algorithm, and building a framework that identifies and classifies uncertainty. Phase 2 covers objective 3, an algorithm that reduces uncertainty using evidence and structured reasoning, and objective 4, comparing the full framework with existing LLMs on faithfulness, hallucination reduction and cost."

---

### Slide 8: Phase 1 Plan and Architecture (≈ 2 min)

**On the slide:** the architecture diagram. The numbers 1–8 on it are the eight steps of the Phase 1 plan.

**Say:**
"This slide shows the Phase 1 plan and the architecture in one picture. The numbers are the eight steps.

1. **Question sets.** We freeze 11 public datasets, 3,367 questions, with fixed versions so anyone can repeat the study.
2. **LLMs under test.** We use 9 to 12 open models from 5 families, from 3 to 14 billion parameters.
3. **Generate.** Each question gets one normal answer, called the **greedy** answer, at temperature 0. It also gets ten **sampled** answers at temperature 1, plus a few self-checks.
4. **Mark right or wrong.** Rules mark the maths answers. A judge model from a *different* family marks free text, so a model never grades itself.

Everything is **stored once**, and every method reads the same stored answers. That makes the comparison fair, and adding a method later needs no new model runs.

5. On the left, **Objective 1** scores uncertainty with 17 methods. It measures how well each one predicts mistakes, and builds a baseline profile per model and dataset.
6. On the right, **Objective 2** first re-asks the unsure answers to get labels.
7. It then trains a detector and a type classifier, and compares them with the pipeline's current routing rule.
8. **Run and report.** One command runs every step, resumes after a crash, and writes the tables and figures."

**If asked:**
- *Why temperature 1 for samples?* That is the setting in the semantic-entropy papers. It shows the model's real spread of answers. The greedy answer at temperature 0 is the one we grade.
- *Why ten samples?* The Nature paper uses ten. We also measure how AUROC changes with 2 to 10 samples, so we can tell whether fewer are enough. The pipeline uses 4.
- *How many calls per question?* 18 per model: 1 greedy answer, 10 samples, 2 confidence questions and 5 yes/no self-checks.

---

### Slide 9: Methodology: Semantic Entropy (≈ 2 min)

**On the slide:** the Canberra/Sydney example, the formula, and four bullets.

**Say:**
"Semantic entropy is the existing algorithm for Objective 1. We ask the same question ten times, here *'What is the capital of Australia?'*. Seven answers say Canberra and three say Sydney.

Next we group answers that **mean the same**. Two answers go in the same group only if each one implies the other. A language-inference model, DeBERTa, checks this in both directions. 'Canberra' and 'The capital is Canberra' go together; 'Sydney' goes in another group.

Then we compute entropy over the groups: H equals minus the sum, over the groups, of p times log p, where p is the share of answers in that group. Here that is minus 0.7 log 0.7 minus 0.3 log 0.3, which is **0.61**.

If all ten agree, H is 0 and the model is sure. If the answers scatter over many meanings, H is high, and the answer is likely a hallucination. For maths, multiple-choice and yes/no questions we do not need the language model: we compare the final answers directly."

**If asked:**
- *Why meaning and not exact words?* 'Canberra' and 'It is Canberra' are different words but the same answer. Counting words would overstate uncertainty.
- *What is the maximum H?* With 10 samples, when all 10 differ, H = ln 10 ≈ 2.30. We also use H divided by ln 10, which runs from 0 to 1 (see Part B3).

---

### Slide 10: Methodology: Seventeen Methods (≈ 1.5 min)

**Say:**
"We do not test semantic entropy alone. It is compared with 16 other published methods, in four families.

- **Meaning-level, 9 methods.** These measure how the meanings of the answers spread: semantic entropy in three forms, the number of meaning groups, three graph methods, disagreement with the main answer, and SelfCheckGPT. The formula shown, EigV, is a soft count of the meaning groups taken from a graph.
- **Token-level, 5 methods.** These use the model's own word probabilities, for example the negative log-likelihood, NLL, which is high when the model found its own words unlikely.
- **Surface, 1 method.** Plain word overlap between the answers, using ROUGE-L.
- **Self-evaluation, 2 methods.** The model judges itself: P(True), and a confidence it states from 0 to 100.

All 17 run on every answer of every model and are computed from the same stored answers. Each gives one score per question, and we compare them by how well those scores separate wrong answers from right ones."

**If asked:**
- *Which one is best?* The full run will tell. Literature suggests meaning-level methods do best on short factual answers, and token methods are cheap but noisy.
- *Is a method chosen per model or per question type?* No. All 17 run everywhere. The results are reported per model and per dataset, so we can see which method works best where.

---

### Slide 11: Methodology: Classifying Uncertainty (≈ 2 min)

**On the slide:** two blue bars (Identify, Classify), the re-ask diagram on the left, the type → fix table and the classifier box on the right.

**Say:**
"Objective 2 asks two questions about an answer.

**Identify**: how likely is it to be wrong? We combine the 17 signals into one calibrated probability of hallucination and report it as low, medium or high.

**Classify**: what kind of doubt is it? The table on the right lists the four types and what fixes each:
- **Knowledge.** The model lacks or misremembers a fact, so we look up evidence.
- **Reasoning.** The facts are the same but the conclusions differ, so we use careful step-by-step reasoning.
- **Aleatoric.** The question itself is unclear, unanswerable or based on a false premise, so we ask for clarification, refuse, or correct the premise.
- **Temporal.** The true answer changes over time, so we search for fresh information.

To train a classifier we need labelled examples. The diagram shows how we get them: **the type is whichever re-ask removes the doubt.** For each unsure or wrong answer we ask the same model again, five times with the **true evidence** and five times with a **method hint**. The hint is the first half of a worked solution and never gives the answer. If the evidence fixes it, the model lacked knowledge. If the hint fixes it, the reasoning was the problem. 'Fixed' means the majority answer becomes correct *and* the five answers agree.

Ambiguous, unanswerable and time-dependent questions get their type from their datasets. About 300 answers are also labelled by hand and checked with Cohen's kappa.

Finally, the classifier learns the type from the 17 signals, how the answers group, and 5 yes/no self-checks. It must beat the pipeline's current routing rule, meaning the 95% confidence interval of the improvement is above zero."

**If asked:**
- *What does aleatoric mean?* It is uncertainty in the data or question itself. More knowledge does not remove it. Epistemic uncertainty, by contrast, comes from missing knowledge; knowledge and reasoning are the two epistemic types here.
- *What if both re-asks fix it?* It is labelled 'mixed' and left out of training. Neither fixing it gives 'unresolved', which is also left out.
- *Why both conditions for 'fixed'?* A correct majority with scattered samples could be luck. Requiring agreement, with at most one dissenting sample out of five, means the doubt is really gone.
- *Which classifiers are compared?* The current routing rule, a depth-3 decision tree printed as rules, logistic regression, and gradient-boosted trees, all tested on unseen datasets and models.

---

### Slide 12: Phase 1: Plan, Execution and Result (≈ 1 min)

**Say:**
"This table compares what we planned with what we built. The question sets, answer generation, right/wrong labelling, Objective 1 and Objective 2 are all built. Only the full study is pending: up to 12 models on a Mac, about one week of running with one command. Even before that run, stored results from our pipeline give a first real finding: semantic entropy works well on maths, with AUROC 0.86, but not on myth questions, with 0.55."

---

### Slide 13: Implementation and Evaluation Setup (≈ 1.5 min)

**On the slide:** the datasets table, three number tiles, the tools table and the four metrics.

**Say:**
"We use 11 public datasets, chosen so that every kind of doubt appears:
- **Missing knowledge:** TriviaQA, PopQA and HaluEval, 900 questions.
- **Common myths:** TruthfulQA, 817 questions.
- **Reasoning:** GSM8K, MATH, StrategyQA and ARC, 950 questions.
- **Unclear or unanswerable:** AmbigQA and SelfAware, 500 questions.
- **Changing facts:** FreshQA, 200 questions.

That is 3,367 questions, 17 methods and up to 12 models.

For the implementation, Ollama runs the open models locally, DeBERTa checks whether two answers mean the same, scikit-learn gives the classifiers and metrics, and pandas and SciPy give the tables and statistics.

There are four metrics:
- **AUROC:** are wrong answers ranked as more unsure?
- **AURAC:** accuracy when unsure answers are refused.
- **ECE:** is a stated 70% right 70% of the time?
- **Macro-F1:** the quality of the type classifier."

**If asked:**
- *Why run locally?* Open models are free, the results can be repeated, and we can read token probabilities, which many paid APIs do not give.
- *How do you avoid tuning on the test data?* Each dataset is split once into 30% dev and 70% test. Every threshold and choice uses dev only, and all results are reported on test.
- *How do you know a difference is real?* We use 95% bootstrap confidence intervals, the DeLong test to compare two methods, grouped 5-fold cross-validation, and leave-one-dataset, model or family out.
- *Where are the datasets?* In `data/uq_sets/`: one file per dataset, plus `MANIFEST.json` with each source, version, licence and checksum.

---

### Slide 14: Results: Baseline Uncertainty per Model (1/3), facts and myths (≈ 1.5 min)

**On the slide:** a table with one row per model. Each cell lists values in the order of the Datasets column. Marked "EXPECTED VALUES".

**Say:**
"These three slides show the main Objective 1 table: one row for each of the 12 models. **These are expected values, not yet our measurements.** Accuracy comes from published benchmark scores where they exist, and AUROC from published semantic-entropy results. The full run on the Mac will replace them with measured values. The accuracy and AUROC of qwen2.5:3b and llama2:7b on TruthfulQA are already real, from our earlier runs.

How to read a row: the Datasets column gives the order, and every other cell follows that order.
- **Accuracy:** the share of answers that are right.
- **AUROC of semantic entropy:** how well entropy separates wrong answers from right ones. 0.5 is guessing and 1 is perfect.

This slide covers facts and myths. Fact questions are expected to give a useful AUROC, around 0.75 to 0.8, which is the level reported in the Nature paper. TruthfulQA stays near 0.5, and our real cells confirm it: 0.55 for qwen2.5:3b and 0.58 for llama2:7b. PopQA accuracy is low for every model because it asks about rare facts; published work shows small models answer only about one in five such questions."

**If asked:**
- *Where do these numbers come from?* Published model benchmarks and the semantic-entropy papers; Part D of my notes lists every source. The full run replaces them with our own measurements.
- *Where is the blind spot?* No paper reports it for these models, so it will be added from the full run.

---

### Slide 15: Results (2/3), reasoning (≈ 1 min)

**Say:**
"This is the reasoning group: GSM8K, MATH, StrategyQA and ARC. The values are expected from published results, except qwen2.5:3b and llama2:7b on GSM8K, which are real. For example, qwen2.5:3b answers 84% of GSM8K questions correctly in our run, and entropy separates its wrong answers well, with AUROC 0.86. Published maths scores show the same ordering we expect: newer and larger models do much better, from 24% for llama2:7b up to about 95% for qwen2.5:14b. Maths gives the clearest signal: when the model is wrong, its ten answers scatter over different numbers, so entropy is high."

---

### Slide 16: Results (3/3), unclear and changing questions (≈ 1 min)

**Say:**
"The last group is AmbigQA, SelfAware and FreshQA: ambiguous, unanswerable and time-dependent questions. All values here are expected values. These are the hardest questions: accuracy is lowest and entropy is weaker, because a model often gives the same wrong answer every time, for example an outdated fact. Its samples agree, so entropy says 'sure'. That is exactly why Objective 2 does not stop at *how* unsure the model is but asks *what kind* of doubt it is."

---

### Slide 17: Results: Maths vs Myths Across Models (≈ 1.5 min)

**On the slide:** a bar chart of AUROC on GSM8K and TruthfulQA for all 12 models, three summary boxes, and a green "Ready to run" strip. Marked "EXPECTED VALUES".

**Say:**
"This chart puts maths and myths side by side for every model. The bars are expected values from published results, except qwen2.5:3b and llama2:7b, which are real: 0.86 and 0.79 on maths, against 0.55 and 0.58 on myths.

On maths, wrong answers scatter across the ten samples, so entropy spots them. On myth questions, right and wrong answers look equally unsure, so entropy is close to guessing. The boxes summarise the chart: about 0.83 on maths against 0.57 on myths, and entropy is expected to work better on maths for every model.

The green strip shows what is already verified, using a dry run with a simulated model. Every one of the 6 stages finished for 12 models. After a simulated crash, only the 5 missing questions out of 30 were redone. Over 80% of the planted types were recovered. So the full study is ready to run."

**If asked:**
- *What is a simulated model?* A stand-in that answers instantly with known, planted kinds of uncertainty. It proves the code works end to end; its numbers say nothing about real models, so they are never reported as results.

---

### Slide 18: Discussion and Limitations (≈ 1 min)

**On the slide:** two columns of three short points.

**Say:**
"What we see:
1. Entropy spots maths errors well.
2. On myth questions it is close to guessing.
3. Some errors are made while the model is sure: the blind spot. Sampling cannot see them, so we also test self-checks and classify the type of doubt.

Limitations:
1. Most result numbers are expected values, not yet measured; the real numbers so far cover two models and two datasets.
2. A judge model marks free-text answers, so a second judge and 100 human checks test the judge.
3. We only test small open models, 3 to 14 billion parameters. Larger models may behave differently."

**If asked:**
- *How reliable are the type labels?* They come from re-asking, and about 300 hand labels check them with Cohen's kappa.

---

### Slide 19: Phase 2 Plan (≈ 1 min)

**Say:**
"Phase 2 uses what Phase 1 learns. First, each unsure answer is routed by its type. Knowledge doubts get **evidence-based verification**: the factual claims are checked against retrieved passages. Reasoning doubts get **structured reasoning**: a plan, worked samples and small programs, followed by a vote. Both are Objective 3. Then we compare the whole system with the base models on faithfulness, hallucination reduction and cost, which is Objective 4.

We already have a head start. A first pipeline prototype raised GSM8K accuracy from 84% to 88% on 100 test questions, but TruthfulQA did not improve, going from 66% to 63%. That is exactly where Phase 1's uncertainty types should help."

---

### Slide 20: References (≈ 15 s)

**Say:**
"These are the main papers behind the methods. Semantic entropy is from Farquhar and colleagues, in Nature 2024. Thank you. I am happy to take questions."

---

## Part B: Formulas and methods in detail

### B1. Setup and notation

For one question **x** and one model:

| Symbol | Meaning |
|---|---|
| `g` | the **greedy** answer, generated at temperature 0. This is the answer we grade. |
| `s₁ … s_K` | **K = 10 sampled** answers, generated at temperature 1 with fixed seeds. Cut-off or empty samples are dropped, so `m ≤ K` valid samples remain. |
| `n = m + 1` | greedy answer plus samples |
| `p(tᵢ)` | the probability the model gave to token *i* of an answer (read from Ollama's log-probs) |
| `wrong` | 1 if the greedy answer is wrong, 0 if right |
| `u` | an uncertainty score; higher means less sure |

Two extra calls give self-evaluation: **P(True)** and a **0–100 confidence**. Five yes/no **probes** feed Objective 2 (B12). Code: `uncertainty/generate/runner.py`, `uncertainty/generate/prompts.py`.

**Temperature** controls randomness. At 0 the model always picks its most likely next word. At 1 it samples words in proportion to their probability, so the samples show the range of answers the model considers plausible.

---

### B2. Grouping answers by meaning

Semantic methods need to know when two answers **mean the same**.

**Free-text answers.** We use the NLI model DeBERTa-v3 (`cross-encoder/nli-deberta-v3-base`). NLI stands for natural language inference. Given a *premise* A and a *hypothesis* B, it outputs three probabilities: **entailment** (A implies B), **neutral**, and **contradiction**. Each answer is prefixed with the question, so that "Canberra" is read as "What is the capital of Australia? Canberra".

```
A entails B   ⇔   P(entail | A → B) ≥ max(P(neutral | A → B), P(contradiction | A → B))
A ≡ B         ⇔   A entails B  AND  B entails A        (bidirectional entailment)
```

**Worked answers** (numbers, maths expressions, multiple-choice options, yes/no). The final answer is read from the text, and two answers are equal when the read answers match. This is exact, and no NLI model is needed.

**Clusters (meaning groups).** We go through the answers in order. Each answer joins the first group whose first member (the representative) it is equivalent to; if there is none, it starts a new group (Kuhn et al., 2023). Code: `methods/view.py` (`equivalent`, `clusters`) and `methods/nli.py`.

**Why both directions?** "Paris" entails "a city in France", but not the other way round. One-way entailment would wrongly merge a specific answer with a vague one.

---

### B3. Semantic entropy (Objective 1's main method)

#### (a) Discrete semantic entropy (Farquhar et al., 2024)

```
p(c) = (number of samples in group c) / m

SE = − Σ_c  p(c) · ln p(c)
```

**Example (slide 9).** 10 samples: 7 say "Canberra" and 3 say "Sydney".

```
SE = −(0.7 · ln 0.7 + 0.3 · ln 0.3)
   = −(0.7 · (−0.357) + 0.3 · (−1.204))
   = 0.250 + 0.361 = 0.611
```

| Situation | Groups | SE |
|---|---|---|
| all 10 agree | {10} | 0 |
| 7 / 3 | {7, 3} | 0.611 |
| 5 / 3 / 2 | {5, 3, 2} | 1.030 |
| all 10 differ | 10 groups of 1 | ln 10 = 2.303 (the maximum) |

**Normalised SE** (used for the thresholds and the blind spot):

```
SE_norm = SE / ln m          (0 = all agree, 1 = all different)
```

Example: 0.611 / ln 10 = 0.611 / 2.303 = **0.265**.

**Why "entropy"?** Entropy measures how unpredictable an outcome is. If you always get the same answer, there is no surprise (0). If every answer differs, the surprise is maximal.

#### (b) Likelihood-weighted semantic entropy (Kuhn et al., 2023)

Instead of counting samples, each sample is weighted by how likely the model found it:

```
w(s) = exp( (1/L) · Σ_i ln p(tᵢ) )       (length-normalised probability of sample s, L tokens)
P(c) = Σ_{s in c} w(s)  /  Σ_all w(s)
SE_weighted = − Σ_c P(c) · ln P(c)
```

**Example.** The 7 "Canberra" samples each have average log-prob −0.1, so w = e^−0.1 = 0.905. The 3 "Sydney" samples each have −0.7, so w = e^−0.7 = 0.497.
Mass: Canberra 7 × 0.905 = 6.33 and Sydney 3 × 0.497 = 1.49, so P = 0.81 and 0.19.
SE_weighted = −(0.81 ln 0.81 + 0.19 ln 0.19) ≈ **0.49**. This is lower than 0.611 because the model itself found "Sydney" unlikely.

#### (c) Pipeline Layer 1 semantic entropy

This is the variant our pipeline uses: the greedy answer is counted together with the samples, and the result is scaled to [0, 1]:

```
SE_pipeline = min(1,  − Σ_c (n_c / n) · ln(n_c / n)  /  ln n),     n = m + 1
```

Example: greedy "Canberra" plus 7 Canberra and 3 Sydney samples gives 8/11 and 3/11. The entropy is 0.586, and 0.586 / ln 11 = **0.244**.
The study also reports it with only 4 samples (`se_pipeline_k4`), which is the pipeline's real operating point.

Code: `methods/semantic_entropy.py`.

---

### B4. Number of meaning groups (Lin et al., 2024)

```
NumSets = number of distinct groups among the samples
```

It is simple and easy to explain (1 means all agree). But it is coarse: a {9, 1} split and a {5, 5} split both give 2.

---

### B5. Graph (spectral) methods (Lin, Trivedi & Sun, 2024)

Treat the m samples as points in a graph, joined by edges weighted by how similar they are.

```
W_jk = similarity of samples j and k  (in [0, 1])
       free text:      ( P(entail j→k) + P(entail k→j) ) / 2,   with W_jj = 1
       worked answers: 1 if the read answers match, else 0
D    = diagonal matrix of degrees,  D_jj = Σ_k W_jk
L    = I − D^(−1/2) · W · D^(−1/2)          (normalised graph Laplacian)
λ_1 ≤ … ≤ λ_m = eigenvalues of L
```

| Method | Formula | Meaning |
|---|---|---|
| **EigV** | `Σ_k max(0, 1 − λ_k)` | a soft count of meaning groups |
| **Degree** | `1 − mean_j(D_jj) / m` | the average dissimilarity between samples |
| **Eccentricity** | `‖V − mean(V)‖_F`, where V holds the eigenvectors with λ < 0.9 (one row per sample) | how spread out the samples are in the graph's "map" |

**Intuition with m = 10:**

| Situation | EigV | Degree |
|---|---|---|
| all agree (W all ones) | 1 | 0 |
| two groups of 5 | 2 | 0.5 |
| all differ (W = identity) | 10 | 0.9 |

EigV works like the number of groups, but because it uses soft similarities, near-duplicates count as "almost one group". `‖·‖_F` is the Frobenius norm: the square root of the sum of all squared entries. Code: `methods/graph.py`.

---

### B6. Consistency methods

**Disagreement with greedy** (self-consistency idea, Wang et al., 2023):

```
Disagree = 1 − (number of samples equivalent to the greedy answer) / m
```

Example: greedy "Canberra", 7 of 10 samples agree, so 1 − 0.7 = **0.3**. This is close to what the pipeline's majority vote sees.

**SelfCheckGPT-NLI** (Manakul et al., 2023), free text only:

```
SelfCheck = (1/m) · Σ_j  P(contradiction | premise = sample j, hypothesis = greedy)
```

Example: 7 samples contradict the greedy answer with probability 0.02 and 3 with 0.95, so (7 × 0.02 + 3 × 0.95) / 10 = **0.30**.

**Lexical similarity** (Lin et al., 2024): plain word overlap, with no meaning involved.

```
LexSim = 1 − mean over all sample pairs (a, b) of ROUGE-L-F1(a, b)       (first 64 words of each)
ROUGE-L: LCS = longest common subsequence of words
         P = LCS / |b|,  R = LCS / |a|,  F1 = 2PR / (P + R)
```

Example: "the capital is Canberra" vs "Canberra is the capital". The LCS is "the capital" (2 words), so P = R = 0.5, F1 = 0.5 and LexSim = 0.5, although the meanings are identical. This is why surface overlap is a weak baseline.

Code: `methods/consistency.py`.

---

### B7. Token-level methods (Malinin & Gales, 2021; Kuhn et al., 2023)

These read the model's own token probabilities for the greedy answer `t₁ … t_L`.

```
NLL       = − Σ_i ln p(tᵢ)                       (sequence negative log-likelihood)
NLL_norm  = − (1/L) · Σ_i ln p(tᵢ)               (per token, so long answers are not punished)
```

Example: the answer "Canberra." has 3 tokens with probabilities 0.90, 0.95 and 0.99.
NLL = −(ln 0.90 + ln 0.95 + ln 0.99) = 0.105 + 0.051 + 0.010 = **0.167**; NLL_norm = 0.167 / 3 = **0.056**.

**Token entropy.** At each position Ollama reports the top 5 alternative tokens. We renormalise them to sum to 1 (q) and take the entropy:

```
Hᵢ = − Σ_v q_v · ln q_v           (over the top-5 alternatives at position i)
TokenEntropy_mean = mean_i Hᵢ ,   TokenEntropy_max = max_i Hᵢ
```

Example: alternatives 0.6, 0.2, 0.1, 0.05 and 0.05 give Hᵢ = 0.307 + 0.322 + 0.230 + 0.150 + 0.150 = **1.16**. Because only the top 5 are seen, this is a lower bound on the true entropy.

**Predictive entropy** (a Monte-Carlo estimate over the samples, length-normalised):

```
PE = − (1/m) · Σ_s  (1/L_s) · Σ_i ln p(t_i^s)
```

It is high when the sampled answers themselves were unlikely.

**Weakness:** token methods measure uncertainty about *wording*. "Canberra" and "The capital city is Canberra" have different probabilities but mean the same thing. Code: `methods/token_level.py`.

---

### B8. Self-evaluation (Kadavath et al., 2022; Tian et al., 2023)

**P(True).** The model sees the question, its own samples as "brainstormed ideas" and its proposed answer, and is asked: *"Is the possible answer: (A) True (B) False"*. We read the probabilities of "A" and "B" from the first token:

```
P(True) = P(A) / (P(A) + P(B))
uncertainty = 1 − P(True)
```

**Verbalised confidence.** The model is asked: *"How confident are you that the proposed answer is correct? Reply with a number from 0 to 100."*

```
confidence = number / 100 ,    uncertainty = 1 − confidence
```

These two are probabilities, so they are also checked for **calibration** (ECE and Brier, B10). Code: `methods/self_eval.py`, `generate/runner.py` (`binary_prob`).

---

### B9. Marking answers right or wrong

- **Maths, options and yes/no:** rules. The final answer is read and compared with the gold answer (numbers, expressions, letters).
- **Free text:** a **judge model from another family**. llama3.1:8b judges qwen, gemma, phi and mistral; qwen2.5:7b (or 14b) judges the llama models. A **second judge** from a third family re-grades about 300 answers to measure agreement, Cohen's κ (B12).
- The **100-answer human audit** checks the judge itself.

Code: `label/correctness.py`, `label/judge.py`, `roster.py`.

---

### B10. Evaluation metrics (how good is an uncertainty score?)

#### AUROC (the main metric)

```
AUROC = P( u(random wrong answer) > u(random right answer) )  +  ½ · P(tie)
```

In words: pick one wrong and one right answer at random. AUROC is the chance that the wrong one got the higher uncertainty. **0.5 = guessing, 1.0 = perfect.**

**Worked example.** Wrong answers have u = 0.9, 0.6 and 0.3; right answers have u = 0.0, 0.2 and 0.6. There are 3 × 3 = 9 pairs:

| wrong u | beats | ties | points |
|---|---|---|---|
| 0.9 | 0.0, 0.2, 0.6 | – | 3 |
| 0.6 | 0.0, 0.2 | 0.6 | 2.5 |
| 0.3 | 0.0, 0.2 | – | 2 |

AUROC = 7.5 / 9 = **0.83**.

**Rank formula** (what the code uses): with m wrong and k right answers, `AUROC = (Σ ranks of wrong − m(m+1)/2) / (m·k)`. Here the ranks are 6, 4.5 and 3, so (13.5 − 6) / 9 = 0.83.

AUROC does not depend on the score's scale, so SE, NLL and P(True) can be compared fairly.

#### AUPRC

The area under the precision–recall curve with "wrong" as the positive class (average precision). It is more informative than AUROC when wrong answers are rare.

#### Accuracy–rejection curve and AURAC (Farquhar et al., 2024)

Sort the answers from most sure to least sure. If we answer only the r most sure and refuse the rest, the accuracy is:

```
Acc(r) = (number right among the r most sure) / r
AURAC  = mean over r = 1..N of Acc(r)
```

With the same 6 answers, sorted by u, the correctness is 1, 1, 0, ½, ½, 0 (the tied pair counts ½ each). The cumulative accuracies are 1, 1, 0.667, 0.625, 0.6 and 0.5, so AURAC = **0.73**. A random ordering gives 0.5, and a perfect ordering gives 0.81.
**Accuracy at coverage c** is Acc(⌈c·N⌉). For example, the accuracy if we answer only the 80% most sure questions.

#### PRR (prediction rejection ratio)

```
risk(score) = mean over r of (1 − Acc(r))
PRR = (risk_random − risk_method) / (risk_random − risk_oracle)
```

1 means as good as an oracle and 0 means as good as random. Example: (0.5 − 0.268) / (0.5 − 0.192) = **0.75**.

#### Calibration: ECE and Brier (for probabilities: P(True), verbal confidence, the detector)

**ECE** (expected calibration error), with 15 equal-width bins:

```
ECE = Σ_b (n_b / N) · | accuracy_b − mean confidence_b |
```

Example: 5 answers at confidence 0.9 with 4 right (accuracy 0.8), and 5 answers at 0.5 with 3 right (0.6).
ECE = 0.5 × |0.8 − 0.9| + 0.5 × |0.6 − 0.5| = **0.10**. 0 means perfectly calibrated: "70% sure" is right 70% of the time.

**Brier score:**

```
Brier = (1/N) · Σ (confidence − correct)²
```

Same example: (4 × 0.1² + 1 × 0.9² + 3 × 0.5² + 2 × 0.5²) / 10 = **0.21**. Lower is better.

Code: `uncertainty/metrics.py`.

---

### B11. Statistics, thresholds and the blind spot

**Dev/test split.** Each set is split once, with a fixed seed, into **30% dev** and **70% test**. Everything that is chosen (thresholds, the best single method, detector training) uses dev only. Every reported number is on test.

**95% bootstrap confidence interval.** Resample the test questions with replacement 1,000 times and recompute the metric each time. The 2.5th and 97.5th percentiles form the interval. A narrow interval means a stable result.

**DeLong test** (comparing two methods on the *same* questions):

```
z = (AUROC_A − AUROC_B) / sqrt( Var_A + Var_B − 2·Cov_AB )
p = 2 · (1 − Φ(|z|))
```

Because both methods score the same answers, their AUROCs are correlated, and the covariance term accounts for that. DeLong estimates the variances from each answer's contribution to the AUROC. p < 0.05 means the difference is unlikely to be chance.

**Holm correction** (many comparisons). We compare semantic entropy with 16 methods, so some differences would look significant by luck. Sort the p-values in ascending order, p₍₁₎ ≤ … ≤ p₍ₘ₎:

```
adjusted p₍ᵢ₎ = max_{j ≤ i}  min(1, (m − j + 1) · p₍ⱼ₎)
```

Example with 3 tests, p = 0.01, 0.02 and 0.04: the adjusted values are 0.03, 0.04 and 0.04.

**Thresholds on SE_norm** (tuned per model on dev; `thresholds.py`):
- **low ("sure")**: the largest t such that answers with SE_norm ≤ t are at least **90% right** (at least 10 answers).
- **high**: the smallest t such that answers with SE_norm ≥ t are at least **50% wrong**.

**Blind spot** (`report/obj1.py`):

```
Blind spot = (number of wrong answers with SE_norm ≤ low) / (number of wrong answers)
Confident-error rate = (number of wrong answers with SE_norm ≤ low) / (all answers)
```

These are confident hallucinations: the samples agree, so no sampling method can flag them. Self-evaluation and Objective 2 are tested on exactly these.

**AUROC vs K.** The same analysis is rerun using only the first k = 2 … 10 samples (`View.subset`). This shows how many samples are worth their cost, compared with the pipeline's K = 4.

---

### B12. Objective 2: the framework

#### Inputs (features), all available without knowing the right answer

- the **17 method scores** and SE_norm;
- **group structure**: share of the largest group (`top_share`), share of the second group, whether the greedy answer is in the majority, average log-prob of the top two groups (`lp_top1`, `lp_top2`), `confident_split` (both groups likely: suggests aleatoric), and the spread of log-probs;
- **premise vs conclusion**: `premise_agreement` is the mean Jaccard overlap of the facts or numbers the samples use, `|A ∩ B| / |A ∪ B|`. `premise_minus_conclusion = premise_agreement − top_share`. A high value means the samples share their premises but reach different conclusions, which points to a reasoning problem;
- **text cues**: a year or time words in the question; hedges, refusals or premise corrections in the answer; question length;
- **question kind** (short, number, expression, choice, yes/no) and the pipeline's task type;
- **5 self-diagnosis probes**: P(Yes) for "Is this question ambiguous?", "Does the answer change over time?", "Does it need a calculation?", "Does it rest on a false premise?" and "Is it impossible to answer with certainty?".

Code: `framework/features.py`.

#### Identify: the hallucination detector

**Logistic regression** over the 17 scores:

```
z = w · x + b ,       p = σ(z) = 1 / (1 + e^(−z))
```

Before fitting, missing scores are filled with the median plus a "was missing" flag, and every feature is standardised, `(x − mean) / sd`.
**Platt calibration** then maps the raw output through a second sigmoid, `p_cal = σ(A·z + B)`, fitted on held-out folds (5-fold), so that "p = 0.7" really means about 70% wrong.

**Levels**, from out-of-fold predictions on dev:
- **low**: the largest p below which answers are at least 90% right;
- **high**: the smallest p above which they are at least 50% wrong;
- **medium**: everything in between.

It is compared with the best single method (chosen on dev) and with semantic entropy, using DeLong. A **lite** version uses K = 4 and no extra calls, which matches the pipeline's budget. Code: `framework/detector.py`.

#### Labels: the type of doubt (PLAN §9.3; `label/interventions.py`, `framework/taxonomy.py`)

Precedence: **human label > dataset metadata > intervention**.

- **Metadata**: AmbigQA gives *aleatoric/ambiguous*; SelfAware unanswerable gives *aleatoric/unanswerable*; FreshQA gives *aleatoric/false premise* or *temporal*.
- **Interventions**: re-ask 5 times with the oracle evidence, and 5 times with the reasoning hint. A re-ask **fixes** the question when

```
majority answer is correct   AND   ( SE_norm ≤ low   OR   majority share ≥ 0.8 )
```

| Evidence fixed? | Hint fixed? | Label |
|---|---|---|
| yes | no | knowledge |
| no | yes | reasoning |
| yes | yes | mixed (not used for training) |
| no | no | unresolved (not used) |

Only answers that **have** uncertainty are labelled: wrong answers, or answers whose SE_norm is above the "sure" threshold.

#### Classify: four classifiers compared

1. **Baseline (routing rule)**: maths becomes *reasoning*, time cues become *temporal*, everything else *knowledge*. It never predicts aleatoric.
2. **Decision tree** of depth 3, on readable features, printed as if-then rules. Each split is chosen to reduce **Gini impurity**, `G = 1 − Σ_k p_k²` (0 = pure node).
3. **Multinomial logistic regression**: `P(type k | x) = e^(w_k·x) / Σ_j e^(w_j·x)` (softmax).
4. **Gradient-boosted trees** (HistGradientBoosting): 200 small trees, each correcting the errors of the ones before it.

All four use `class_weight = balanced`, which gives each type a weight of `N / (number of types × N_type)` so that rare types are not ignored. Code: `framework/classifier.py`.

#### Classifier metrics

```
precision_k = TP_k / (TP_k + FP_k)
recall_k    = TP_k / (TP_k + FN_k)
F1_k        = 2 · precision_k · recall_k / (precision_k + recall_k)
macro-F1    = (1/K) · Σ_k F1_k            (every type counts equally)
```

Example: F1 of 0.80 (knowledge), 0.60 (reasoning), 0.70 (aleatoric) and 0.50 (temporal) gives macro-F1 = **0.65**.

**Beats the baseline** when the paired-bootstrap 95% interval of `macro-F1(best) − macro-F1(rule)` lies entirely above 0. "Paired" means the same resampled answers are used for both.

#### Validation, so the classifier learns uncertainty, not dataset names

- **Grouped stratified 5-fold cross-validation** (`StratifiedGroupKFold`): the same question, answered by different models, never appears in both training and testing, and each fold keeps the type proportions.
- **Leave-one-dataset-out**, **leave-one-model-out** and **leave-one-family-out**: train without one dataset, model or family, then test on it.
- **Permutation importance**: shuffle one feature's column and measure how much macro-F1 drops (5 repeats). A big drop means an important feature.

#### Agreement: Cohen's κ (judges and human labels)

```
κ = (p_o − p_e) / (1 − p_e)
p_o = observed agreement ;  p_e = agreement expected by chance
```

Example: two judges agree on 85 of 100 answers (p_o = 0.85). Judge A says "correct" 60% of the time and judge B 55%, so p_e = 0.6 × 0.55 + 0.4 × 0.45 = 0.51.
κ = (0.85 − 0.51) / 0.49 = **0.69**, which is substantial agreement. We flag κ below 0.4.

#### Recommended action

| Level / type | Action |
|---|---|
| low and not over-confident | accept |
| knowledge | retrieve evidence (pipeline Layer 2) |
| reasoning | structured reasoning (pipeline Layer 3) |
| aleatoric: ambiguous / unanswerable / false premise | ask to clarify / abstain / correct the premise |
| temporal | fresh retrieval |

Try it on one question: `python -m uncertainty analyze "question" --model qwen2.5:3b`.

---

### B13. The models and datasets

**Models (roster; 12 on a 24 GB+ Mac, 9 on 16 GB):**

| Family | Models |
|---|---|
| Qwen | qwen2.5:3b (the pipeline model), qwen2.5:7b, qwen2.5:14b\*, qwen3:4b |
| Llama | llama3.2:3b, llama3.1:8b, llama2:7b (an older generation) |
| Gemma | gemma3:4b, gemma3:12b\* |
| Phi | phi4-mini (3.8B), phi4 (14B)\* |
| Mistral | mistral:7b |

\* needs 24 GB of memory. **Focus models** (larger question set plus interventions): qwen2.5:3b, llama3.1:8b and gemma3:4b.

**Datasets (11):** TriviaQA (popular facts), PopQA (long-tail facts), HaluEval (multi-hop facts), TruthfulQA (misconceptions), GSM8K (arithmetic word problems), MATH (competition maths), StrategyQA (multi-step yes/no), ARC (science multiple choice), AmbigQA (ambiguous), SelfAware (unanswerable), FreshQA (changing facts and false premises).

---

## Part C: Quick glossary

| Term | One line |
|---|---|
| Hallucination | a fluent answer that is wrong or unsupported |
| Greedy answer | the single most likely answer (temperature 0); the one we grade |
| Sample | an answer drawn at temperature 1, so it varies from run to run |
| Semantic entropy | entropy over meaning groups of the samples; 0 = all agree |
| NLI / entailment | a model that decides whether sentence A implies sentence B |
| Bidirectional entailment | A implies B and B implies A, so they mean the same |
| Epistemic uncertainty | doubt from missing knowledge (knowledge, reasoning types) |
| Aleatoric uncertainty | doubt in the question itself (ambiguous, unanswerable, false premise) |
| Temporal uncertainty | the right answer changes over time |
| AUROC | chance that a wrong answer is scored more unsure than a right one (0.5 = guessing) |
| AURAC | average accuracy when refusing the least sure answers first |
| PRR | how much of the ideal refusal gain a method achieves (0 = random, 1 = ideal) |
| ECE | gap between stated confidence and actual accuracy |
| Brier score | mean squared error of a confidence |
| Blind spot | share of errors made while the model was sure |
| Bootstrap CI | range of a metric over 1,000 resamples of the questions |
| DeLong test | tests whether two AUROCs on the same questions differ |
| Holm correction | adjusts p-values when many tests are run |
| Dev / test split | 30% to tune, 70% to report |
| Platt calibration | a sigmoid that turns a score into an honest probability |
| Macro-F1 | average F1 over the types, each type counted equally |
| Cohen's κ | agreement between two raters beyond chance |
| Intervention | re-asking with evidence or a hint to find what fixes the doubt |
| Cross-family judge | a grader from a different model family, so a model never grades itself |

---

## Part D: Where the expected values come from

The result slides (14–17) use these values until the full run replaces them. **Published** means the value on the slide is the reported score, rounded. **Real** means our own stored runs. Every other cell is an **estimate**, built from the model's size and family and the published scores around it. Benchmarks use different settings (number of examples in the prompt, chain of thought, base vs instruct model), so published scores are a guide to what our run should find, not a promise.

### Real (our stored runs, 100 test questions each, pipeline Layer 1)

| Model | Dataset | Accuracy | AUROC |
|---|---|---|---|
| qwen2.5:3b | GSM8K | 84% | 0.86 |
| qwen2.5:3b | TruthfulQA | 66% | 0.55 |
| llama2:7b | GSM8K | 24% | 0.79 |
| llama2:7b | TruthfulQA | 61% | 0.58 |

### Published accuracy used

| Model | Dataset | On the slide | Published | Source |
|---|---|---|---|---|
| qwen2.5:7b | GSM8K | 92% | 91.6 (instruct) | [llm-stats comparison](https://llm-stats.com/models/compare/qwen-2.5-14b-instruct-vs-qwen-2.5-7b-instruct) (secondary) |
| qwen2.5:7b | MATH | 76% | 75.5 (instruct) | [Qwen2.5 blog](https://qwenlm.github.io/blog/qwen2.5-llm), [Qwen2.5 technical report](https://arxiv.org/pdf/2412.15115) |
| qwen2.5:14b | GSM8K | 95% | 94.8 (instruct) | [llm-stats comparison](https://llm-stats.com/models/compare/qwen-2.5-14b-instruct-vs-qwen-2.5-7b-instruct) (secondary) |
| qwen2.5:14b | MATH | 80% | 80.0 (instruct); 75.6 under simple-evals | llm-stats; [Phi-4 technical report](https://arxiv.org/html/2412.08905v1) |
| llama3.2:3b | GSM8K / MATH / ARC | 78% / 48% / 79% | 77.7 / 48.0 / 78.6 | [Llama 3.2 3B Instruct model card (NVIDIA NIM)](https://docs.api.nvidia.com/nim/reference/meta-llama-3_2-3b-instruct) |
| llama3.1:8b | GSM8K / MATH / ARC | 85% / 52% / 83% | 84.5 / 51.9 / 83.4 | same model-card table |
| gemma3:4b | GSM8K | 89% | 89.2 (instruct, 0-shot) | [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3) |
| gemma3:4b | TriviaQA | 66% | 65.8 (pre-trained, 5-shot) | [Gemma 3 listing](https://lmstudio.ai/models/gemma-3) |
| gemma3:12b | GSM8K / TriviaQA | 94% / 78% | 94.4 / 78.2 | Gemma 3 model card and listing |
| phi4-mini | GSM8K / ARC | 89% / 84% | 88.6 / 83.7 | [DataLearner](https://www.datalearner.com/en/ai-models/pretrained-models/Phi-4-mini-instruct) (secondary) |
| phi4 | MATH | 80% | 80.4 | [Phi-4 technical report](https://arxiv.org/html/2412.08905v1) |
| mistral:7b | TriviaQA | 70% | 69.9 (base) | [Mistral 7B paper](https://arxiv.org/pdf/2310.06825) |
| llama2:7b | TriviaQA | 64% | 63.8 (base) | Mistral 7B paper |

Close estimates anchored on published numbers: mistral:7b GSM8K 48% (published 52.2 for the base model with a vote of 8 answers, so a single greedy answer will be a bit lower); gemma3 MATH 76% / 84% and phi4-mini MATH 64% (the instruct-model scores in their technical reports; not re-checked here, so treat them as estimates).

### Published basis for the other columns

- **AUROC of semantic entropy.** Farquhar et al. (Nature 2024) report an average AUROC of about 0.79 over 30 task–model pairs (TriviaQA, SQuAD, BioASQ, NQ-Open, SVAMP; LLaMA 2, Falcon, Mistral) ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750)). The slides use about 0.80 for popular facts, 0.74–0.77 for long-tail and multi-hop facts, 0.80–0.83 for maths, and lower values for yes/no, ambiguous and time-dependent questions, where only a few distinct answers exist or the model repeats the same wrong answer. TruthfulQA (about 0.57) and GSM8K (about 0.83) follow our own real runs.
- **PopQA.** Mallen et al. (ACL 2023) show that models answer few long-tail questions: about 16–19% for GPT-J 6B and GPT-3 on the least popular facts ([paper](https://arxiv.org/abs/2212.10511)). The slides use 15–28%.
- **Everything else** (HaluEval, StrategyQA, AmbigQA, SelfAware, FreshQA, and the remaining TruthfulQA, TriviaQA and ARC cells) is estimated. These datasets have no published scores for these exact models under our setup.

### After the full run

In `build_phase1_deck.py`, put the measured numbers from `uncertainty/results/reports/obj1/` into `EXP`, set `EXPECTED = False`, and rebuild. The tag disappears, and the blind-spot column can be added back from the same reports.
