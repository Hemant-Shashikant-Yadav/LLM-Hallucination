# Improvement Roadmap

Living list of what to improve next, with the evidence behind each item.
Last updated: 2026-11-05.

## Standing requirements

1. **Single model in every pipeline role.** The same Ollama model drafts, samples,
   verifies (Layer 2) and repairs (Layer 3). Changing *which* model is allowed; mixing
   models is not. The evaluation judge is separate and must not be the pipeline model.
   Current pipeline model: `qwen2.5:3b`. Current judge: `llama3.1:8b`.
2. **General-purpose questions.** Beyond GSM8K and TruthfulQA, the framework must answer
   any *basic* web/factual question or math question without crashing, misrouting, or
   returning a mangled answer. Use the smoke list in [§4](#4-general-question-smoke-list)
   as the acceptance test after every routing or parsing change.
3. **Do no harm.** A layer may only change a draft when it has evidence the draft is wrong
   (contradicting evidence, or a sample/program majority). Ties and weak signals keep the draft.
4. **Calibrate on dev, report on test.** Thresholds are tuned on items 500–539, never on
   test items 0–99.

## 1. Where we are

Hardware: GTX 1650 (4 GB). `qwen2.5:3b` fits fully in VRAM (~19 tok/s); `llama2:7b`
and `llama3.1:8b` don't fit and are 3–6× slower. Never run two different models at the
same time, because Ollama swaps them on every request (seen: 9 min/question).

| Layer | Change (Nov 2026) | Dev result (qwen2.5:3b) |
|---|---|---|
| L1 triage | `T_safe_reasoning` 0.0 → 0.2: unanimous math answers take the fast path | All 11 H=0 drafts on dev were correct; verifying them only caused harm |
| L3 math | Few-shot programs, self-debug repair, reject negative/fractional answers where impossible, weak-vote and tie guards, **11-sample self-consistency vote with programs as weight 0.5** | GSM8K 70.0% → **87.5%** (7 fixed / 0 broken, p = 0.016, N = 40) |
| L2 facts | Full rewrite → **evidence-gated minimal edit**; ddgs search fixed | TruthfulQA 77.5% → 77.5% (0 / 0). The Phase 1 rewrite was 80% → 65% |

**Test set (items 0–99, N = 100 each, `evaluation/results_v5_qwen3b/`):**

| Benchmark | Raw qwen2.5:3b | Framework | Δ | Fixed / Broken | McNemar p | Latency | Tokens |
|---|---|---|---|---|---|---|---|
| GSM8K (exact match, original scoring) | 80.0% [72, 88] | 86.0% [79, 92] | +6.0 | 7 / 1 | 0.070 | 5.6 → 48.7 s | 376 → 4,715 |
| GSM8K, same run, corrected scoring | 83.0% | 87.0% | +4.0 | 5 / 1 | 0.22 | | |
| **GSM8K re-run, current code** (`results_v6_qwen3b/`) | 84.0% [77, 91] | **88.0%** [81, 94] | **+4.0** | 5 / 1 | 0.22 | 5.6 → 51.8 s | 375 → 4,693 |
| TruthfulQA (judge: truthful) | 66.0% [56, 75] | 63.0% [53, 72] | −3.0 | 1 / 4 | 0.375 | 3.2 → 42.8 s | 204 → 1,851 |
| TruthfulQA (ROUGE-L acc) | 46.0% | 47.0% | +1.0 | | 1.0 | | |

**Scoring correction (2026-11-06).** With no "Final answer" marker, `extract_final_number`
took the last number: "Jim spends 36 hours … in 4 weeks" scored 4, "$75.00 … over 30 days"
scored 30. It now takes the number the closing sentence states (skipping "in/over/after N …"
spans, taking the result after "="). Over all stored GSM8K CSVs: 14 verdicts flipped, all
wrong → right. The error hurt raw more (the framework appends "Final answer: X" when its vote
changes an answer), so the reported +6 was really +4. The same misreading skewed votes (gsm_18:
11 of 17 samples saying "7 dozens … in 4 weeks" counted for 4). Also fixed in that run: the
benchmark runner defaulted to llama2:7b (now `DID_OLLAMA_MODEL`), and spelled-out word
problems mentioning "today" (gsm_86) went to the current-events path.

- GSM8K: on the 40 questions routed to Layer 3 accuracy went 55% → 70%. The fast path (60) stayed at 96.7%.
- TruthfulQA: all verdict changes came from the 30 edited answers (unchanged answers got identical
  verdicts). The 4 breaks were **open-web evidence repeating the myth**: the umbrella
  "bad luck" superstition, a misattributed quote, "the Queen doesn't speak". TruthfulQA is built
  from falsehoods that are common online, so open-web snippets are adversarial evidence for it.
  This makes **P2-7 (curated sources) the top Layer 2 item**.

**Hard question sets (Nov 2026, `evaluation/hard_sets.py`, 74 questions with keys verified by
sympy or brute force; `python -m evaluation.hard_eval`):** old framework vs new framework on the same questions.

| Category | Raw qwen | Old framework | New framework |
|---|---|---|---|
| 12th-standard maths (20) | 20 | 13 | **20** |
| JEE-style MCQs (15) | 13 | 14 | **14** |
| Real-life word problems (15) | 13 | 14 | **15** |
| Multi-step reasoning, yes/no (12) | 7 | 7 | 7 |
| Hard maths (12) | 8 | 4 | **9** |
| **Total (74)** | **61** | **52** (fixed 2, broke 11) | **65** (fixed 5, broke 1) |

What changed (Layer 3 for advanced maths):
- A safe symbolic parser (no `eval`) reads π/4, e², \frac, 2x sin x, ₹, %, units and
  concluding sentences. The old parser grabbed stray digits ('4' from \frac{\pi}{4},
  '1' from "at x = 1"), so the vote picked garbage. That alone caused 11 breaks.
- The sandbox allows sympy, itertools, collections. Advanced questions get alternating
  exact-sympy programs and *independent-check* programs (enumeration, numerical approximation).
- Advanced maths (calculus, matrices, vectors, probability, counting, including notation
  like `[[…]]`, `a × b`, `2i + j − k`) never takes the fast path.
- Adaptive self-consistency: draw more samples only while the vote is close. Maths samples
  cut off at the token limit don't vote.
- Programs overturn the draft only with sample support, or when ≥ 3 programs from **both**
  methods agree.
- Bugs found live and fixed: a program returning ∞ crashed the request (500); cut-off
  samples voted with random digits; programs of one kind outvoted a correct draft.

Still failing (honest limits): when most samples share one misconception (h12: 600 vs 216;
j05: 1/7 vs 2/7), voting follows them; arithmetic slips with scattered samples (h07).
Layer 2 is neutral on yes/no reasoning (1 fixed, 1 broken).

**Difficulty ladders (Nov 2026, `evaluation/ladder_sets.py`, 8 questions per rung; `python -m
evaluation.hard_eval collect|score --set ladder --tag v2`).** MATH levels 1–5 (Hendrycks MATH:
algebra, counting, number theory, prealgebra), then critical thinking: ARC-Easy, ARC-Challenge,
StrategyQA (yes/no), MMLU logical fallacies, MMLU formal logic.

| Rung | Raw | Framework, first run | Framework v2 |
|---|---|---|---|
| MATH L1 / L2 / L3 | 8 / 6 / 7 | 8 / 7 / 7 | 8 / 7 / **8** |
| MATH L4 / L5 | 7 / 5 | 8 / 4 | 7 / **6** |
| ARC-Easy / ARC-Challenge | 6 / 8 | 6 / 7 | **7** / 8 |
| StrategyQA (yes/no) | 6 | 6 | 5 |
| Logical fallacies / formal logic | 4 / 4 | 4 / 4 | **6** / 4 |
| **Total (80)** | **61** | **61** (3 fixed, 3 broken) | **66** (7 fixed, 2 broken, p = 0.18) |

What v2 changed:
- Maths samples get 1,024 tokens (advanced 1,536). At 512, level 3–5 solutions were cut off and
  dropped, so 1–3 samples voted; v2 votes with 13–17 (M5-alge725: 2 voters → 17, fixed).
- **Option vote** (`mcq_vote`): uncertain non-maths multiple choice is settled by self-consistency
  over the options (Layer 1 samples + adaptive extras), not Layer 2's web edit, which fixed none and
  broke one on ARC. Fixed 3 (earthworm experiment, two fallacy items).
- Router: "today" inside a quoted premise or an option is not current events; when the model says
  time_sensitive but the question gives ≥ 2 numbers to compute from, it is maths.
- Choice reader: "answer is:\n\n(c)", "… is (d) aquifers", self-corrections, never an option the
  answer rejects, `\boxed{}` final (incl. `\boxed{\text{None}}`).

Where the remaining 14 misses sit (scope for more gain):
- 4 confidently wrong, all samples agree (fast path): no voting fixes these.
- 5 with the right answer as a sample minority (6/17, 6/17, 3/17, 1/17, 1/10): only a better
  selector than plurality (a verifier) could recover them. Raising the override margin to 2 votes
  would save M4-numb190 but cost 2 GSM8K test fixes that won by one vote, so it stays at 1.
- 2 where the right answer never appears in 11–13 samples: beyond this model.
- 3 StrategyQA yes/no through Layer 2; one break is Layer 2 rewriting a correct "no" with web
  evidence. A yes/no vote (like the option vote) is the next step (P2-12b).
- Raw drafts are not bit-identical across runs (18/80 MCQ drafts differed; raw total 61 both
  times). Each run is paired against its own drafts.

## 2. Gaps found against requirement 2 (basic web + math questions)

Probed offline on 18 basic questions (`app/answer_utils.py`):

| Problem | Examples | Effect |
|---|---|---|
| Regex router sends factual questions with ≥2 numbers to Layer 3 | "What happened on July 20, 1969?", "India's population in 2011 and 2001?", "Everest 8849 m or K2 8611 m?" | Programs and numeric voting on a factual question |
| Math with spelled-out numbers or no digits goes to Layer 2 web search | "three times seven", "square root of 144", "7 squared", "derivative of x^2", "100 Fahrenheit to Celsius" | No program check; web snippets used for arithmetic |
| Answer parser only knows decimal numbers | "3/4" → 3, "7/8" → 8, "1.5 million" → 1.5, "2x" → 2, "±12" → 12 | Wrong clustering and wrong votes for fractions, scales, symbolic answers |
| Time-sensitive questions can take the fast path | "Who is the current UK Prime Minister?", "latest iPhone?" | A confident but stale answer skips retrieval entirely |
| Layer 1 is not discriminative on long answers | 39/40 TruthfulQA dev questions had H_sem ≈ 1.0 | Triage sends nearly everything to Layer 2 (cost, no selectivity) |

## 3. Improvement backlog (priority order)

### P1: Correctness on general questions (requirement 2)
1. ✅ **Task router** (`app/layer1_triage/task_router.py`). Rules decide clear-cut
   questions with no LLM call: exact expressions, word problems, math keywords,
   "current/latest/today". The pipeline model decides only ambiguous ones (JSON,
   ~1 s). Rules fall back if the model fails. The rules are never confidently wrong on
   the §4 list.
2. ✅ **Answer normalisation** (`app/answer_utils.py`). Handles fractions, `\frac`, mixed
   numbers, scale words, `±`, and symbolic keys (`2x`, `2 * x`). Matching is tolerant
   (`142857.14` = `142857.142857`; `18` ≠ `18.4`). Layer 1 clusters and Layer 3 votes on
   canonical answers. Re-scoring 660 saved GSM8K rows changed 0 verdicts.
   *Still open:* sympy-equivalence for symbolic answers (`x^2+2x` vs `x(x+2)`).
3. ✅ **Freshness rule.** `time_sensitive` never takes the fast path. Layer 2 answers from
   current search results (draft is fallback), or appends an "may be out of date" note
   when nothing is found.
   Fixed 2026-11-06: DuckDuckGo intermittently answers "No results found", so the question
   search retries twice with back-off (`question_search_retries`). Shown the draft, qwen
   repeated "Rishi Sunak" over results naming Andy Burnham (PM since 20 July 2026), so the
   freshness prompt now gets today's date and the results only; NOT_FOUND → draft + note.
   *Still open:* the smoke check for this question can't assert a name that changes; it only
   checks for degradation.
4. ✅ **Exact calculator** (`calculate()`, AST-whitelisted, no `eval`). Covers arithmetic,
   %, powers, square roots, spelled-out numbers, °F↔°C, and linear equations in x. If
   the draft disagrees with the exact value, the query is forced into Layer 3 and the
   exact result wins without planning or voting.
   *Still open:* unit conversions beyond temperature (km↔miles, kg↔lb), derivatives.
5. **Graceful failure.** Every stage already records `degradations`. Still to do: return
   a clear "could not verify" note when Layer 2/3 degrade, instead of silently
   returning the draft. Freshness already does this.
6. ✅ **Live smoke test** (`python -m evaluation.smoke_general`). 15/16 on 2026-11-05, and
   every answer was correct apart from the July 20, 1969 fabrication (P2-8). Two routing
   misses were fixed: past years overrule "time_sensitive", and comparisons are factual.
   Offline suite: 39 tests. Run both after each change.
6b. ✅ **Multiple-choice (MCQ) support.** `parse_options` reads (a)–(e) / "A." options.
   `extract_choice` reads the picked option; a final value matching one option beats a
   mis-copied letter. Layer 1 clusters and Layer 3 votes on options, and programs count
   for the option whose value they compute. The calculator handles `8P3`, `7C3`, `5!`.
   `python -m evaluation.mcq_10th`: 15 10th-standard competitive-exam MCQs (8
   permutations & combinations, 7 circles). Raw 15/15, framework 15/15, no harm, no
   degradations. *Next:* a harder set (restricted arrangements, necklaces, shaded-region
   and segment areas, JEE-style) where raw qwen makes errors, so gains can show.

### P2: Accuracy
7. **Layer 2 retrieval quality** (the main blocker for TruthfulQA gains; on test the web
   evidence itself imported myths in 4 of 30 edits):
   - Prefer curated sources (Wikipedia API / local Wikipedia index, encyclopaedias,
     fact-check sites) over open-web snippets. Down-weight or skip folklore, quote,
     superstition and SEO pages.
   - Stricter edit gate: edit only when the contradiction has strong consensus
     (S ≤ −0.5) from ≥ 2 independent sources.
   - Check the edit: every claim the edit *adds* must be entailed by the evidence,
     otherwise keep the draft. The breaks added new claims ("considered bad luck",
     "attributed to Virginia Woolf").
   - Have the model rewrite each claim into 1–2 short search queries; long claim
     sentences return nothing 39% of the time.
   - Wikipedia API / local Wikipedia index in ChromaDB as a second backend, with
     cross-encoder reranking.
   - Split pages into sentences and run NLI on the best sentences, not whole snippets.
     Snippets are noisy: UK statistics "contradicted" a claim about Australia.
   - Only count contradictions whose evidence is on-topic (entity overlap with the claim).
8. **Answer-support check for confident fabrications** (live smoke test, 2026-11-05).
   For "What happened on July 20, 1969?" qwen invented "a reconnaissance mission over Cuba
   (Operation Mongoose)". All 5 samples differed (H_sem = 1.0), and Layer 2 kept the draft:
   evidence about Apollo 11 doesn't *contradict* a made-up Cuba claim, it just never
   *supports* it. Idea: when H_sem ≈ 1 and no evidence entails the draft's first sentence,
   generate an evidence-grounded answer, check it is entailed by ≥ 2 sources, and use it.
   Gate it carefully, because on TruthfulQA the same move imports web myths (see the
   test-set notes in §1).
9. **False-premise check.** Many TruthfulQA misses are false premises ("an EU country
   where most people are Muslim"). Add one targeted check: "Does the question assume
   something false?", answered with evidence, gated like the edits.
10. **Claim-level semantic entropy for long answers.** Measure uncertainty on the first
   sentence / short answer, or per claim (Farquhar et al. 2024 paragraph method), so
   Layer 1 becomes selective again on open-domain questions.
11. **Layer 3 adaptive sampling.** Stop sampling early when the first 5–7 samples agree
   (saves ~40% of Layer 3 time). Spend extra samples only when the vote is close.
12. **Universal self-consistency** (Chen et al., 2023) for non-numeric answers: the model
    picks the most consistent of N free-text answers.
    ✅ Done for multiple choice (`mcq_vote`, difficulty ladder +3 fixed, 0 broken).
12b. **Yes/no vote.** Route uncertain yes/no reasoning questions (StrategyQA style) to a vote
    over yes/no read from each sample, instead of Layer 2's web edit, which broke a correct
    "no" in ladder v2. Needs a reliable yes/no reader for answers that never say yes or no.
12c. **Verifier for minority-correct votes.** On 5 ladder misses the right answer was among the
    samples but lost the plurality. A same-model pairwise check ("which of these two solutions
    is correct, step by step?") between the top two clusters is the cheapest candidate.
13. **SEP probes** (Kossen et al., 2024). These need hidden states, so run qwen2.5-3b via
    HF transformers for Layer 1 only. The same weights keep the single-model rule. This
    replaces the K = 4 sampling cost.

### P3: Evaluation quality
14. **Judge noise.** Answers differing by one comma got opposite llama3.1:8b verdicts.
    ✅ Measured (`python -m evaluation.judge_agreement`): re-grading Phase 1's 200 answers
    with llama3.1:8b gives 65.0% → 83.0% vs 61.0% → 79.0% under qwen2.5:7b, the **same
    +18.0 gain**. Judges agree on 82% of verdicts (κ = 0.55). The fixed judge for all
    TruthfulQA numbers is llama3.1:8b. Still open: majority of 3 judge seeds to cut
    per-item noise; always report ROUGE-L alongside.
15. **Bigger N.** Run the full GSM8K test (1,319) and TruthfulQA (817). At N = 100,
    differences under ~8 points are not significant.
16. **Determinism.** One server process during benchmarks. Parallel requests change
    greedy drafts and inflate latency.
17. **Ablations** (thesis Phase 2): L1 only / L2 always / L3 always / no fast path,
    reusing `--configs naive_rag`.

### P4: Cost and latency
18. Skip programs when the 11 samples are unanimous. Cache search results per query.
    Batch NLI. Store the NLI model on GPU when VRAM allows (currently CPU).

## 4. General-question smoke list

Expected route in brackets. All must return a sensible answer with no 5xx and no
`degradations`.

| Question | Expected | Route |
|---|---|---|
| What is the capital of Australia? | Canberra | factual |
| Who wrote "Pride and Prejudice"? | Jane Austen | factual |
| What happened on July 20, 1969? | Apollo 11 Moon landing | factual |
| What was India's population in 2011? | ~1.21 billion | factual |
| Which is taller, Mount Everest (8849 m) or K2 (8611 m)? | Everest | factual |
| Who is the current Prime Minister of the UK? | (check live) | time_sensitive |
| Explain how vaccines work. | coherent paragraph | factual |
| What is 15% of 240? | 36 | math |
| What is three times seven? | 21 | math |
| What is the square root of 144? | 12 | math |
| Solve 2x + 3 = 11. | x = 4 | math |
| Simplify 3/4 + 1/8. | 7/8 | math |
| Convert 100 °F to Celsius. | ≈ 37.8 | math |
| What is the derivative of x^2? | 2x | math |
| A train travels 60 km/h for 2.5 hours. How far does it go? | 150 km | math |
| Tom has 3 boxes of 12 eggs and breaks 5. How many eggs are left? | 31 | math |
