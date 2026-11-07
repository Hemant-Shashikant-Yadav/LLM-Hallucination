# 6. Roadmap

Ordered by expected value. Measure every change on dev first, then on test, and re-run the
smoke test (below) after any routing or reading change.

## Next (new hardware)

1. **Stronger pipeline model.** The biggest lever. Compare candidates with
   `python -m evaluation.model_compare collect --model <tag>` then `score` (194 questions
   against qwen2.5:3b). Keep the one-model rule; the judge must stay a different model.
   Set `DID_OLLAMA_THINK=false` for thinking models.
2. **Full-size runs** for significance: GSM8K 1,319, TruthfulQA 817, MATH-500, JEEBench
   (multi-correct MCQs need the vote to return several letters first).
3. **Measure the unmeasured**: the yes/no vote (ladder StrategyQA, hard-set reasoning) and
   Wikipedia evidence (TruthfulQA, `DID_WIKIPEDIA_EVIDENCE=true`).

## Accuracy

4. **Verifier for outvoted right answers.** In 5 ladder misses the right answer was among
   the samples but lost; a same-model pairwise check of the top two answers is the
   cheapest candidate.
5. **Layer 2 evidence quality** (main TruthfulQA blocker): curated sources; edit only on
   contradiction from 2+ sources; reject edits that add claims the evidence doesn't support;
   short search queries per claim; NLI on the best sentences, not whole snippets.
6. **Confident fabrications.** "What happened on July 20, 1969?" still gets an invented
   answer: when entropy is high and no evidence supports the draft, answer from evidence
   checked against 2+ sources.
7. **False-premise check** for questions that assume something untrue.
8. **Claim-level uncertainty** for long answers (Layer 1 sends 96% of TruthfulQA to Layer 2).

## Evaluation

9. Three judge seeds or two judges, majority vote; always report ROUGE-L too.
10. Ablations: Layer 1 only / always Layer 2 / always Layer 3 / no fast path.
11. Bigger ladder (more per rung) so per-level differences are meaningful.

## Cost

12. Skip programs when samples are unanimous; cache searches; batch NLI; NLI on GPU.

## Smoke test (must pass after every routing or reading change)

`python -m evaluation.smoke_general` — 16 questions; all need a sensible answer, no
server error and no degradation:

| Question | Expected | Route |
|---|---|---|
| What is the capital of Australia? | Canberra | factual |
| Who wrote "Pride and Prejudice"? | Jane Austen | factual |
| What happened on July 20, 1969? | Apollo 11 Moon landing (still fails) | factual |
| What was India's population in 2011? | ~1.21 billion | factual |
| Which is taller, Mount Everest (8849 m) or K2 (8611 m)? | Everest | factual |
| Who is the current Prime Minister of the UK? | check live | time-sensitive |
| Explain how vaccines work. | coherent paragraph | factual |
| What is 15% of 240? | 36 | maths |
| What is three times seven? | 21 | maths |
| What is the square root of 144? | 12 | maths |
| Solve 2x + 3 = 11. | x = 4 | maths |
| Simplify 3/4 + 1/8. | 7/8 | maths |
| Convert 100 °F to Celsius. | ≈ 37.8 | maths |
| What is the derivative of x^2? | 2x | maths |
| A train travels 60 km/h for 2.5 hours. How far does it go? | 150 km | maths |
| Tom has 3 boxes of 12 eggs and breaks 5. How many eggs are left? | 31 | maths |
