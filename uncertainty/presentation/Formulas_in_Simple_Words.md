# Uncertainty Formulas in Simple Words

This guide explains every formula and calculation in the uncertainty study (thesis objectives 1 and 2) in plain words. For each one you get:

- **what it asks**, in one line;
- **the formula**;
- **what each letter means**;
- **a worked example**, step by step;
- **how to read the result**, and where it is in the code.

All examples use **one question**, so you can compare the methods with each other. The example numbers are made up for teaching, but every result was computed with the study's own code (`uncertainty/methods/`, one file per method, and `uncertainty/metrics.py`). So this guide and the code agree.

For the longer, more technical version, see Part B of [Phase1_Presentation_Script.md](Phase1_Presentation_Script.md).

---

## Part 0: Basic ideas you need first

### Probability or share

A number from 0 to 1. **0.7** means "70 out of 100", or "7 out of 10".

### Σ (sigma): "add them all up"

`Σ p(c)` means: take `p(c)` for every group `c`, and add them together.

### ln: the natural logarithm

`ln` is a button on every scientific calculator. You only need four facts:

- `ln 1 = 0`
- `ln` of a number **below 1** is **negative**; the smaller the number, the more negative.
- `ln` of a number **above 1** is **positive**.
- It turns multiplying into adding: `ln(a × b) = ln a + ln b`.

| x | 0.01 | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1 | 2 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|
| ln x | −4.605 | −2.303 | −1.204 | −0.693 | −0.357 | −0.105 | 0 | 0.693 | 2.303 | 2.398 |

### e^x: the opposite of ln

`e^x` undoes `ln`: if `ln 0.905 = −0.1`, then `e^−0.1 = 0.905`. We use it to turn a log-probability back into a probability.

| x | −0.1 | −0.7 | −1.2 | 1.2 |
|---|---|---|---|---|
| e^x | 0.905 | 0.497 | 0.301 | 3.320 |

### Entropy: "how unpredictable is it?"

```
H = − Σ  p × ln p        (over every possible outcome)
```

- A coin that **always** lands heads: p = 1, so H = −(1 × ln 1) = **0**. No surprise.
- A **fair** coin: p = 0.5 and 0.5, so H = −(0.5 × (−0.693) + 0.5 × (−0.693)) = **0.693**. More surprise.

The **minus sign** is there because `ln p` is negative for p below 1; the minus makes H positive. Many methods below are this same idea applied to answers.

### Tokens and log-probabilities

A **token** is a piece of a word ("Canberra" might be "Can" + "berra"). For every token, the model knows how likely it was. Ollama gives us `ln` of that probability, called the **log-probability**. Example: probability 0.90 → log-probability ln 0.90 = −0.105.

### Greedy answer and samples

- **Greedy answer:** the model's single most likely answer (temperature 0). **This is the answer we mark right or wrong.**
- **Samples:** 10 more answers with randomness switched on (temperature 1). They show what else the model thinks is possible.

### Uncertainty score

Every method gives one number per question. **Higher always means "less sure".** If a method gives a confidence (like "80% sure"), we use `uncertainty = 1 − confidence`.

---

## The running example

> **Question:** What is the capital of Australia? **Right answer:** Canberra.
>
> - **Greedy answer:** "Canberra." It is right.
> - **10 samples:** 7 say "Canberra" and 3 say "Sydney".
> - **Word probabilities of the greedy answer:** 3 tokens, "Can" 0.90, "berra" 0.95, "." 0.99.
> - **Average log-probability of each sample:** −0.1 for each "Canberra" sample, −0.7 for each "Sydney" sample.
> - **NLI model** (checks if two answers mean the same): same-meaning pairs get entailment 0.95; different-meaning pairs get contradiction 0.95.
> - **P(True):** 0.80. **Stated confidence:** 85 out of 100.

**Grouping by meaning.** Two answers are in the same group only if each one implies the other. The NLI model DeBERTa checks this in both directions. For maths, multiple choice and yes/no, we simply compare the final answers. Here we get **2 groups: Canberra (7) and Sydney (3)**.

Symbols used everywhere below:

| Symbol | In simple words |
|---|---|
| `m` | number of samples (here 10) |
| `n` | samples plus the greedy answer (here 11) |
| `c` | one meaning group (for example, all the "Canberra" answers) |
| `L` | number of tokens in an answer |

---

## Part 1: The 17 methods

The 17 methods come in four families:

| Family | Methods | The idea |
|---|---|---|
| Meaning of the samples | 1–9 | Do the 10 answers mean the same thing? |
| Word probabilities | 10–14 | How likely did the model find its own words? |
| Word overlap | 15 | Do the answers use the same words? |
| Self-evaluation | 16–17 | Ask the model how sure it is. |

### Family A: the meaning of the samples

#### 1. Semantic entropy (discrete), the main method

**What it asks:** if we ask 10 times, how spread out are the meanings?

```
p(c) = (number of samples in group c) ÷ m
SE   = − Σ  p(c) × ln p(c)            (over every group c)
```

| Symbol | Meaning |
|---|---|
| `p(c)` | the share of samples in group c |
| `m` | number of samples |
| `Σ` | add up over all groups |

**Example:**

1. Shares: Canberra 7 ÷ 10 = **0.7**, Sydney 3 ÷ 10 = **0.3**.
2. ln 0.7 = −0.357 and ln 0.3 = −1.204.
3. Multiply: 0.7 × (−0.357) = −0.250 and 0.3 × (−1.204) = −0.361.
4. Add: −0.611. Flip the sign: **SE = 0.611**.

**How to read it:** 0 means all answers agree (sure). Higher means the answers spread over more meanings (unsure).

| Situation (10 samples) | Groups | SE |
|---|---|---|
| all 10 agree | 10 | **0** |
| 7 and 3 | 7, 3 | **0.611** |
| 5 and 5 | 5, 5 | **0.693** |
| 5, 3 and 2 | 5, 3, 2 | **1.030** |
| all 10 different | 1 each | **2.303** (= ln 10, the maximum) |

**Normalised SE** (used for the "sure" and "unsure" thresholds):

```
SE_norm = SE ÷ ln m          (always between 0 and 1)
```

Example: 0.611 ÷ ln 10 = 0.611 ÷ 2.303 = **0.265**. Dividing by the maximum makes 0 = all agree and 1 = all different, whatever the number of samples.

Code: `methods/01_semantic_entropy.py` → `se_discrete`, `se_normalized`.

#### 2. Semantic entropy (weighted)

**What it asks:** the same as method 1, but answers the model found *more likely* count *more*.

```
w(s)  = e^( average log-probability of sample s )
P(c)  = (sum of w for the samples in group c) ÷ (sum of w for all samples)
SE_w  = − Σ  P(c) × ln P(c)
```

| Symbol | Meaning |
|---|---|
| `w(s)` | the weight of sample s: its average word probability |
| `P(c)` | the share of the total weight that falls in group c |

**Example:**

1. Each Canberra sample: w = e^−0.1 = **0.905**. Each Sydney sample: w = e^−0.7 = **0.497**.
2. Group weights: Canberra 7 × 0.905 = **6.334**; Sydney 3 × 0.497 = **1.490**. Total 7.824.
3. Shares: Canberra 6.334 ÷ 7.824 = **0.810**; Sydney 1.490 ÷ 7.824 = **0.190**.
4. SE_w = −(0.810 × ln 0.810 + 0.190 × ln 0.190) = 0.171 + 0.316 = **0.487**.

**How to read it:** lower than the 0.611 of method 1, because the model itself found "Sydney" unlikely. So the doubt is smaller than plain counting suggests.

Code: `methods/02_semantic_entropy_weighted.py` → `se_weighted`.

#### 3. Semantic entropy (pipeline version)

**What it asks:** the version our pipeline's Layer 1 uses. The greedy answer is counted too, and the result is scaled to 0–1.

```
SE_pipe = [ − Σ (n_c ÷ n) × ln(n_c ÷ n) ]  ÷  ln n
```

| Symbol | Meaning |
|---|---|
| `n_c` | answers in group c, **including** the greedy answer |
| `n` | samples + greedy = 11 |

**Example:**

1. With the greedy "Canberra": Canberra 8, Sydney 3. Shares 8 ÷ 11 = 0.727 and 3 ÷ 11 = 0.273.
2. Entropy: −(0.727 × ln 0.727 + 0.273 × ln 0.273) = 0.232 + 0.354 = 0.586.
3. Scale: 0.586 ÷ ln 11 = 0.586 ÷ 2.398 = **0.244**.

The study also computes it with only **4 samples**, because that is what the pipeline really uses. Fewer samples give a rougher estimate.

Code: `methods/03_semantic_entropy_pipeline.py` → `se_pipeline`.

#### 4. Number of meaning groups

**What it asks:** how many different meanings are there?

```
Groups = number of different meaning groups among the samples
```

**Example:** Canberra and Sydney → **2**.

**How to read it:** 1 = all agree. Simple, but rough: a 9-and-1 split and a 5-and-5 split both give 2.

Code: `methods/04_number_of_meaning_groups.py` → `num_clusters`.

#### 5–7. The graph methods: EigV, Degree, Eccentricity

**The idea:** draw the 10 samples as dots. Join every two dots with a line whose strength is how similar they are. This is called a **similarity graph**.

```
W(j, k) = similarity of samples j and k, between 0 and 1
          text answers: the average of "j implies k" and "k implies j" from the NLI model
          maths, choices, yes/no: 1 if the final answers match, else 0
```

In our example: Canberra–Canberra = **0.95**, Canberra–Sydney = **0.02**, and every dot with itself = 1.

**5. EigV: a "soft" count of the groups**

```
EigV = Σ  max(0, 1 − λ)        (over every eigenvalue λ of the graph's Laplacian)
```

| Symbol | Meaning |
|---|---|
| Laplacian | a standard table (matrix) made from W that describes how the dots are connected |
| `λ` (lambda) | its **eigenvalues**: special numbers a computer calculates from the table. Each "island" of tightly joined dots gives one λ close to 0. |
| `max(0, 1 − λ)` | how close λ is to 0, never below 0 |

**Example:** the computer finds λ = 0, 0.055, and eight values around 0.98–0.99. Then:

1 − 0 = 1, 1 − 0.055 = 0.945, and the other eight add only about 0.07.

**EigV = 1 + 0.945 + 0.07 ≈ 2.02.**

**How to read it:** like the number of groups (2), but "soft". Since Canberra and Sydney are linked a tiny bit (0.02), it is 2.02, not exactly 2. All agree gives about 1; all different gives about 10 (8.47 in the code, because unrelated answers still get a small similarity).

**6. Degree: how unlike each other are the answers, on average?**

```
degree(j) = Σ W(j, k)        (how strongly dot j is joined to all dots)
Degree    = 1 − (average degree) ÷ m
```

**Example:**

1. A Canberra dot: itself 1 + six other Canberra × 0.95 + three Sydney × 0.02 = **6.76**.
2. A Sydney dot: itself 1 + two other Sydney × 0.95 + seven Canberra × 0.02 = **3.04**.
3. Average over all 10 dots: (7 × 6.76 + 3 × 3.04) ÷ 10 = **5.644**.
4. Degree = 1 − 5.644 ÷ 10 = **0.436**.

**How to read it:** 0 = all answers alike (sure), close to 1 = all different (unsure).

**7. Eccentricity: how far apart are the answers on a "map"?**

```
Ecc = total distance of every dot from the centre of the map
      (the map is built from the eigenvectors whose λ is below 0.9)
```

The computer places each answer as a point on a small map, so that similar answers land close together. Eccentricity adds up how far the points are from the centre. (Technically: the Frobenius norm, the square root of the sum of all squared distances.)

**Example:** **1.00** for our 7-and-3 split.

**How to read it:** with clean groups it is about √(groups − 1): 1 group → 0, 2 groups → 1.0, 3 groups → 1.41, 10 groups → 3.0.

| Situation | Groups | EigV | Degree | Ecc |
|---|---|---|---|---|
| all 10 agree | 1 | 1.05 | 0.05 | 0 |
| 7 and 3 | 2 | 2.02 | 0.44 | 1.00 |
| 5 and 5 | 2 | 2.04 | 0.51 | 1.00 |
| 5, 3 and 2 | 3 | 2.96 | 0.62 | 1.41 |
| all different | 10 | 8.47 | 0.88 | 3.00 |

Code: `methods/05_spectral_eigv.py`, `06_spectral_degree.py`, `07_spectral_eccentricity.py`.

#### 8. Disagreement with the main answer

**What it asks:** what share of the samples disagree with the greedy answer?

```
Disagree = 1 − (samples that mean the same as the greedy answer) ÷ m
```

**Example:** the greedy answer is "Canberra" and 7 of 10 samples agree. Disagree = 1 − 7 ÷ 10 = **0.30**.

**How to read it:** 0 = every sample backs the main answer; 1 = none does.

Code: `methods/08_disagreement_with_greedy.py` → `disagree_greedy`.

#### 9. SelfCheckGPT-NLI

**What it asks:** on average, how much does each sample **contradict** the main answer?

```
SelfCheck = average over the samples of  P(contradiction: sample → greedy answer)
```

| Symbol | Meaning |
|---|---|
| `P(contradiction: sample → greedy)` | the NLI model's probability that the sample contradicts the greedy answer |

**Example:** 7 Canberra samples contradict "Canberra" with 0.02; 3 Sydney samples with 0.95.

SelfCheck = (7 × 0.02 + 3 × 0.95) ÷ 10 = (0.14 + 2.85) ÷ 10 = **0.299**.

**How to read it:** close to 0 = the samples support the answer; close to 1 = they contradict it. Only for text answers (it needs the NLI model).

Code: `methods/09_selfcheck_nli.py` → `selfcheck_nli`.

### Family B: the model's own word probabilities

These read the probabilities of the **greedy answer's tokens**: "Can" 0.90, "berra" 0.95, "." 0.99.

#### 10. Sequence NLL (negative log-likelihood)

**What it asks:** how unlikely was the whole answer, for the model itself?

```
NLL = − Σ  ln p(token)          (over every token of the greedy answer)
```

**Example:**

1. ln 0.90 = −0.105, ln 0.95 = −0.051, ln 0.99 = −0.010.
2. Add: −0.167. Flip the sign: **NLL = 0.167**.

**How to read it:** 0 = the model was 100% sure of every token. Higher = less sure. **Problem:** long answers always get a higher NLL, just because they have more tokens.

#### 11. Length-normalised NLL

**What it asks:** the same as NLL, **per token**, so long answers are not punished.

```
NLL_norm = NLL ÷ L          (L = number of tokens)
```

**Example:** 0.167 ÷ 3 = **0.056**.

Code: `methods/10_sequence_nll.py` → `nll`, `methods/11_length_normalised_nll.py` → `nll_norm`.

#### 12–13. Token entropy (mean and max)

**What it asks:** at each word, how torn was the model between its top choices?

```
H(token) = − Σ  q × ln q        (q = the probabilities of the top choices at that spot, scaled to add up to 1)
Mean token entropy = average of H over all tokens
Max token entropy  = the largest H (the single most "torn" spot)
```

**Example:** the top choices Ollama reports at each spot:

| Spot | Choices | H |
|---|---|---|
| 1 | "Can" 0.90, "Syd" 0.06, "Mel" 0.04 | −(0.90 ln 0.90 + 0.06 ln 0.06 + 0.04 ln 0.04) = **0.392** |
| 2 | "berra" 0.95, "bera" 0.05 | **0.199** |
| 3 | "." 0.99, "!" 0.01 | **0.056** |

Mean = (0.392 + 0.199 + 0.056) ÷ 3 = **0.216**. Max = **0.392** (spot 1, where "Sydney" was a real option).

**How to read it:** 0 = no doubt at any word. The max points to the most doubtful word. Because Ollama shows only the top few choices, this is a slight under-estimate.

Code: `methods/12_token_entropy_mean.py`, `methods/13_token_entropy_max.py`.

#### 14. Predictive entropy

**What it asks:** on average, how unlikely were the **sampled** answers?

```
PE = − (average over the samples of each sample's average log-probability)
```

**Example:** (7 × (−0.1) + 3 × (−0.7)) ÷ 10 = (−0.7 − 2.1) ÷ 10 = −0.28. Flip the sign: **PE = 0.28**.

**How to read it:** higher = the samples were less likely. **Problem:** "Canberra" and "The capital is Canberra" have different probabilities but the same meaning. So this mixes up *wording* with *doubt*. Semantic entropy fixes exactly this.

Code: `methods/14_predictive_entropy.py` → `predictive_entropy`.

### Family C: word overlap

#### 15. Lexical similarity (ROUGE-L)

**What it asks:** do the samples use the same words, in the same order?

```
ROUGE-L(a, b): LCS = the longest run of words that appears in both a and b, in the same order (gaps allowed)
               P = LCS ÷ (words in b),   R = LCS ÷ (words in a),   F1 = 2 × P × R ÷ (P + R)
LexSim = 1 − (average F1 over every pair of samples)
```

| Symbol | Meaning |
|---|---|
| LCS | longest common subsequence: words shared in the same order |
| `F1` | one overlap score from 0 (nothing shared) to 1 (identical) |

**Example:**

1. There are 45 pairs of samples (10 × 9 ÷ 2).
2. Same-word pairs: Canberra–Canberra 21 pairs + Sydney–Sydney 3 pairs = 24, each F1 = 1. All other pairs F1 = 0.
3. Average F1 = 24 ÷ 45 = 0.533. LexSim = 1 − 0.533 = **0.467**.

**The trap:** "the capital is Canberra" and "Canberra is the capital" share only "the capital" in the same order, so F1 = **0.5**, although the meaning is identical. Word overlap misses paraphrases.

Code: `methods/15_lexical_similarity.py` → `lexical_sim`.

### Family D: self-evaluation (ask the model)

#### 16. P(True)

**What it asks:** shown its own samples and answer, does the model say "True"?

The model sees the question, its samples as ideas, and its answer, and must reply **(A) True** or **(B) False**. We read the probability of the first letter.

```
P(True)     = P("A") ÷ (P("A") + P("B"))
uncertainty = 1 − P(True)
```

**Example:** P("A") = 0.72 and P("B") = 0.18. P(True) = 0.72 ÷ 0.90 = **0.80**. Uncertainty = 1 − 0.80 = **0.20**.

#### 17. Verbalised (stated) confidence

**What it asks:** "How confident are you, from 0 to 100?"

```
confidence  = the number ÷ 100
uncertainty = 1 − confidence
```

**Example:** the model says 85. Confidence = **0.85**, uncertainty = **0.15**.

**How to read them:** easy to understand, but small models often say "90" even when they are wrong. That is why we also check calibration (Part 2).

Code: `methods/16_p_true.py`, `methods/17_verbalised_confidence.py`.

### All 17 on the running example

| # | Method | Value | 0 means | Higher means |
|---|---|---|---|---|
| 1 | Semantic entropy | 0.611 (norm. 0.265) | all agree | meanings spread |
| 2 | Semantic entropy (weighted) | 0.487 | all agree | likely answers disagree |
| 3 | Semantic entropy (pipeline) | 0.244 | all agree | spread, 0–1 scale |
| 4 | Number of groups | 2 | (1 = all agree) | more meanings |
| 5 | EigV | 2.02 | (≈1 = all agree) | more groups |
| 6 | Degree | 0.436 | all alike | answers unlike each other |
| 7 | Eccentricity | 1.00 | all agree | answers far apart |
| 8 | Disagreement | 0.30 | all back the answer | samples disagree |
| 9 | SelfCheckGPT-NLI | 0.299 | samples support it | samples contradict it |
| 10 | Sequence NLL | 0.167 | fully sure of every word | less likely words |
| 11 | Normalised NLL | 0.056 | fully sure | less likely per word |
| 12 | Mean token entropy | 0.216 | no doubt at any word | torn between words |
| 13 | Max token entropy | 0.392 | no doubt anywhere | one very doubtful word |
| 14 | Predictive entropy | 0.28 | samples very likely | samples unlikely |
| 15 | Lexical similarity | 0.467 | identical wording | different wording |
| 16 | P(True) uncertainty | 0.20 | model says surely true | model doubts it |
| 17 | Stated-confidence uncertainty | 0.15 | "100% sure" | "less sure" |

The values are on **different scales**, so you cannot compare 0.611 with 0.30 directly. That is why every method is judged with AUROC (Part 2), which only looks at the **order** of the scores.

---

## Part 2: How we judge a method

We run a method on many questions, then check: **did the wrong answers get the higher uncertainty?**

Example data for this part, 6 answers:

| Answer | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Right or wrong | wrong | wrong | wrong | right | right | right |
| Uncertainty u | 0.9 | 0.6 | 0.3 | 0.0 | 0.2 | 0.6 |

### AUROC (the main measure)

**What it asks:** pick one wrong and one right answer at random. How often does the **wrong** one have the **higher** uncertainty?

```
AUROC = (pairs where the wrong answer is higher  +  ½ × pairs that are tied) ÷ (all pairs)
```

AUROC stands for **Area Under the Receiver Operating Characteristic curve**; the formula above gives the same number.

**Example:** 3 wrong × 3 right = 9 pairs.

| Wrong answer's u | Higher than right answers | Tied | Points |
|---|---|---|---|
| 0.9 | 0.0, 0.2, 0.6 | – | 3 |
| 0.6 | 0.0, 0.2 | 0.6 | 2 + ½ = 2.5 |
| 0.3 | 0.0, 0.2 | – | 2 |

AUROC = (3 + 2.5 + 2) ÷ 9 = 7.5 ÷ 9 = **0.83**.

**How to read it:** **0.5 = guessing, 1.0 = perfect.** 0.83 means: in 83 of 100 random (wrong, right) pairs, the wrong answer looked less sure. Above about 0.7 is a useful warning signal.

Code: `metrics.py` → `auroc`.

### AUPRC (average precision)

**What it asks:** if we flag the least-sure answers first, how many of the flagged ones are really wrong?

Go down the list from the highest u. At each step where we catch new wrong answers, note the **precision** (share of flagged answers that are wrong):

| Flag everything with u ≥ | Flagged | Wrong among them | Precision | Wrong caught |
|---|---|---|---|---|
| 0.9 | 1 | 1 | 1.00 | 1 of 3 |
| 0.6 | 3 | 2 | 0.67 | 2 of 3 |
| 0.3 | 4 | 3 | 0.75 | 3 of 3 |

AUPRC = average of the precisions, each weighted by the share of wrong answers newly caught at that step (one third each here): (1.00 + 0.67 + 0.75) ÷ 3 = **0.81**.

**How to read it:** higher is better. It is more useful than AUROC when wrong answers are rare.

### AURAC (accuracy when refusing unsure answers)

**What it asks:** if the system answers only its most-sure questions and refuses the rest, how accurate is it?

```
Acc(r) = (right answers among the r most sure) ÷ r
AURAC  = average of Acc(r) for r = 1, 2, …, N
```

**Example:** sort from most sure to least sure: u = 0.0 (right), 0.2 (right), 0.3 (wrong), then the two tied at 0.6 (one wrong, one right: each counts as ½ right), then 0.9 (wrong).

| r (answer this many) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Right so far | 1 | 2 | 2 | 2.5 | 3 | 3 |
| Acc(r) | 1.00 | 1.00 | 0.667 | 0.625 | 0.60 | 0.50 |

AURAC = (1 + 1 + 0.667 + 0.625 + 0.6 + 0.5) ÷ 6 = **0.73**. Random order gives 0.50; a perfect order gives 0.81.

Code: `metrics.py` → `aurac`.

### PRR (prediction rejection ratio)

**What it asks:** of the best possible gain from refusing, how much did our method get?

```
risk     = average of (1 − Acc(r))
PRR      = (risk of random order − risk of our method) ÷ (risk of random order − risk of perfect order)
```

**Example:** random risk = 0.50, our risk = 1 − 0.732 = 0.268, perfect risk = 1 − 0.808 = 0.192.

PRR = (0.50 − 0.268) ÷ (0.50 − 0.192) = 0.232 ÷ 0.308 = **0.75**.

**How to read it:** 1 = as good as perfect, 0 = no better than random.

Code: `metrics.py` → `prr`.

### Calibration: ECE and Brier (only for methods that give a probability)

**What it asks:** when the model says "90% sure", is it right 90% of the time?

```
ECE   = Σ  (share of answers in the bin) × | accuracy in the bin − average confidence in the bin |
Brier = average of (confidence − correct)²          (correct = 1 if right, 0 if wrong)
```

A **bin** is a range of confidence, for example 0.85 to 0.90. The study uses 15 equal bins.

**Example:** 10 answers. 5 said 0.9 and 4 of them were right (accuracy 0.8). 5 said 0.5 and 3 were right (accuracy 0.6).

- ECE = 0.5 × |0.8 − 0.9| + 0.5 × |0.6 − 0.5| = 0.05 + 0.05 = **0.10**.
- Brier = (4 × (0.9 − 1)² + 1 × (0.9 − 0)² + 3 × (0.5 − 1)² + 2 × (0.5 − 0)²) ÷ 10 = (0.04 + 0.81 + 0.75 + 0.50) ÷ 10 = **0.21**.

**How to read them:** **0 is perfect** for both. ECE = 0.10 means the stated confidence is off by 10 points on average.

Code: `metrics.py` → `ece`, `brier`.

---

## Part 3: Thresholds, the blind spot and the statistics

### "Sure" and "unsure" thresholds

Each model's thresholds on SE_norm are chosen on the **dev** questions only (30% of each dataset). Every reported number uses the other 70%, the **test** questions.

- **Low ("sure") threshold:** the largest SE_norm such that the answers at or below it are at least **90% right**.
- **High threshold:** the smallest SE_norm such that the answers at or above it are at least **50% wrong**.

Code: `thresholds.py`.

### The blind spot

**What it asks:** what share of the wrong answers did the model give while it looked **sure**?

```
Blind spot = (wrong answers with SE_norm ≤ low threshold) ÷ (all wrong answers)
```

**Example:** 40 wrong answers; 10 of them had SE_norm at or below the "sure" threshold. Blind spot = 10 ÷ 40 = **0.25**. A quarter of the mistakes were confident mistakes: all 10 samples agreed on the same wrong answer, so no sampling method can catch them.

Code: `report/obj1.py`.

### 95% confidence interval (bootstrap)

**What it asks:** how much could the AUROC change with a different set of questions?

1. Draw the same number of questions at random from the test questions, **with replacement** (a question can be drawn twice).
2. Compute the AUROC again.
3. Repeat 1,000 times. The middle 95% of the results is the interval.

**Example:** "AUROC 0.83, CI 0.76–0.89" means the true value is very likely between 0.76 and 0.89. A narrow interval means a stable result.

### DeLong test (is method A really better than method B?)

```
z = (AUROC_A − AUROC_B) ÷ (how much that difference would wobble by chance)
p = the chance of a difference this big if the two methods were really equal
```

Both methods score the **same** answers, so their AUROCs move together. DeLong's formula for the "wobble" takes that into account. **p below 0.05** means the difference is unlikely to be luck.

### Holm correction (many tests at once)

We compare semantic entropy with 16 methods, so some tests would pass by luck. Holm makes each p-value stricter:

1. Sort the p-values from small to large.
2. Multiply the smallest by the number of tests, the next by one less, and so on.
3. A later value may never be smaller than an earlier one.

**Example:** 3 tests, p = 0.01, 0.02 and 0.04.

- 0.01 × 3 = **0.03**
- 0.02 × 2 = **0.04**
- 0.04 × 1 = 0.04, kept at **0.04**

All three stay below 0.05, so all three still count.

Code: `metrics.py` → `bootstrap_auroc`, `delong`, `holm`.

---

## Part 4: Objective 2 calculations

### The detector: probability of a hallucination (logistic regression)

**What it asks:** combine many uncertainty scores into one probability that the answer is wrong.

```
z = w₁ × x₁ + w₂ × x₂ + … + b        (a weighted sum of the scores)
p = 1 ÷ (1 + e^(−z))                  (the sigmoid squeezes any z into 0–1)
```

| Symbol | Meaning |
|---|---|
| `x₁, x₂, …` | the method scores for this answer (standardised first) |
| `w₁, w₂, …` | weights the computer learns from the dev questions |
| `b` | a starting value (bias), also learned |
| `p` | probability that the answer is a hallucination |

**Example** (made-up weights): SE_norm = 0.265 with weight 3.0, P(True) uncertainty = 0.20 with weight 2.0, bias −1.5.

1. z = 3.0 × 0.265 + 2.0 × 0.20 − 1.5 = 0.795 + 0.40 − 1.5 = **−0.305**.
2. p = 1 ÷ (1 + e^0.305) = 1 ÷ (1 + 1.357) = **0.42**.

A second step (**Platt calibration**) adjusts p so that "0.42" really means about 42% wrong. The level is then **low**, **medium** or **high**, using thresholds found the same way as for SE_norm.

Code: `framework/detector.py`.

### When does a re-ask "fix" the answer?

We re-ask an unsure answer 5 times with the true evidence, and 5 times with a method hint.

```
fixed  =  the most common answer is right
          AND ( the 5 answers agree: SE_norm ≤ low threshold,
                OR at least 4 of the 5 give the most common answer )
```

**Example:** with the evidence, 5 of 5 say "Canberra" → right and agreed → **fixed**. With the hint, 3 say Canberra and 2 say Sydney → majority right, but only 3 of 5 agree → **not fixed**. So the label is **knowledge**: evidence fixed it, the hint did not.

| Evidence fixed it? | Hint fixed it? | Label |
|---|---|---|
| yes | no | knowledge |
| no | yes | reasoning |
| yes | yes | mixed (not used for training) |
| no | no | unresolved (not used) |

Code: `label/interventions.py`.

### Premise agreement (Jaccard)

**What it asks:** do the samples use the same facts or numbers, even when their final answers differ?

```
Jaccard(A, B) = (items in both) ÷ (items in either)
```

**Example:** sample 1 uses the numbers {3, 4, 12}; sample 2 uses {3, 4, 7}. In both: {3, 4} = 2. In either: {3, 4, 7, 12} = 4. Jaccard = 2 ÷ 4 = **0.5**.

If the samples share their facts but reach different answers, the problem is likely the **reasoning**, not missing knowledge.

Code: `framework/features.py`.

### Macro-F1 (how good is the type classifier?)

```
precision = (correct "knowledge" predictions) ÷ (all "knowledge" predictions)
recall    = (correct "knowledge" predictions) ÷ (all answers that really are "knowledge")
F1        = 2 × precision × recall ÷ (precision + recall)
macro-F1  = average F1 over the four types
```

**Example:** the classifier says "knowledge" 40 times and 32 are right, so precision = 32 ÷ 40 = 0.80. There are 40 real knowledge answers and it found 32, so recall = 0.80. F1 = 2 × 0.8 × 0.8 ÷ 1.6 = **0.80**.

With F1 = 0.80 (knowledge), 0.60 (reasoning), 0.70 (aleatoric) and 0.50 (temporal), macro-F1 = (0.80 + 0.60 + 0.70 + 0.50) ÷ 4 = **0.65**.

**Why macro?** Every type counts equally, so a classifier cannot score well by ignoring rare types. It must beat the pipeline's current routing rule.

Code: `framework/classifier.py`.

### Cohen's kappa (κ): do two judges really agree?

```
κ = (agreement seen − agreement expected by chance) ÷ (1 − agreement expected by chance)
```

**Example:** two judges agree on 85 of 100 answers, so agreement seen = 0.85. Judge A says "correct" 60% of the time, judge B 55%. By chance they would agree 0.6 × 0.55 + 0.4 × 0.45 = 0.33 + 0.18 = **0.51**.

κ = (0.85 − 0.51) ÷ (1 − 0.51) = 0.34 ÷ 0.49 = **0.69**.

**How to read it:** 0 = no better than chance, 1 = perfect. 0.6–0.8 is "substantial"; we flag anything below 0.4.

Code: `report/obj1.py` (judges) and `report/obj2.py` (human labels).

---

## One-page summary

| What | Formula in one line | Good value |
|---|---|---|
| Semantic entropy | −Σ p ln p over meaning groups | low = sure |
| SE_norm | SE ÷ ln m | 0 to 1 |
| Number of groups | count the meanings | 1 = sure |
| EigV, Degree, Ecc. | from the similarity graph | low = sure |
| Disagreement | 1 − share agreeing with the main answer | low = sure |
| SelfCheckGPT | average contradiction with the main answer | low = sure |
| NLL, NLL per token | −Σ ln p(token), then ÷ L | low = sure |
| Token entropy | −Σ q ln q at each word | low = sure |
| Predictive entropy | −average log-probability of the samples | low = sure |
| Lexical similarity | 1 − average ROUGE-L F1 | low = sure |
| P(True), stated confidence | 1 − confidence | low = sure |
| AUROC | share of (wrong, right) pairs ranked correctly | 1 = perfect, 0.5 = guessing |
| AURAC, PRR | accuracy when refusing unsure answers | higher is better |
| ECE, Brier | gap between confidence and accuracy | 0 = perfect |
| Blind spot | wrong answers made while sure ÷ all wrong | lower is better |
| Macro-F1 | average F1 over the four types | higher is better |
| Cohen's κ | agreement beyond chance | above 0.6 is good |
