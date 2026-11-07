# 2. The pipeline, step by step

Entry point: `POST /api/v1/query` (`app/main.py`) → `DefenseInDepthRouter.process_query`
(`app/router.py`). Each step below names its file; that file's `.README.md` explains it by
line number.

## Step 0 — trace and draft

- A trace ID is attached to every log line of the query, and a list collects any
  **degradation** (a step that fell back instead of doing its job) — `app/telemetry.py`.
- A token meter counts every LLM call of the query — `app/llm_clients/ollama_client.py`.
- **Draft**: one greedy answer (temperature 0) to the bare question. This is also the
  "raw model" answer in every evaluation.

## Step 1 — what kind of question? (`answer_utils.classify_task_heuristic`, `task_router.py`)

Rules decide clear cases with no model call; only unclear questions ask the model (JSON).

| Rule (in order) | Type |
|---|---|
| Exactly computable (`15% of 240`, `Solve 2x + 3 = 11`, `8P3`, `100 °F to Celsius`) | maths (exact value kept) |
| Word problem: 2+ quantities (digits or spelled out) and a quantity question (`how many`, `total`, `each`) | maths |
| A time word in the question itself (`current`, `latest`, `today`, `as of`), not in quotes or options | time-sensitive |
| Maths words (`area`, `probability`, `solve`) with numbers or x/y/z | maths |
| Otherwise | ask the model |

Two corrections to the model's answer: a question naming a past year is not
time-sensitive ("What happened on July 20, 1969?"); one that supplies 2+ numbers is maths.

## Step 2 — Layer 1: is the model sure? (`semantic_entropy.py`)

1. Draw K = 4 extra answers (temperature 0.7, fixed seeds). Maths and multiple-choice
   samples get 1,024 tokens (1,536 for advanced maths); a sample cut off at the limit is
   **dropped**, because its last digit is not an answer.
2. Group the draft + samples **by meaning**:
   maths → same canonical final answer (`0.5` = `1/2`); multiple choice → same option;
   yes/no → same verdict; anything else → mutual entailment by the NLI model.
3. Semantic entropy over the groups, scaled to 0–1:

   `H = −Σ p(c) log p(c) / log N`

4. Decide:

| Condition | Decision |
|---|---|
| Time-sensitive | always verify (agreeing answers can all be out of date) |
| Advanced maths (calculus, matrices, probability, counting) | always verify |
| H < threshold (0.2 maths, 0.35 otherwise), draft in the largest group, ≥ 2 samples succeeded | **fast path**: return the draft |
| Otherwise | maths → Layer 3; else Layer 2 (or a vote, below) |

If the exact calculator has a value and the draft disagrees, the question goes to Layer 3
regardless of agreement.

## Step 3a — votes for multiple choice and yes/no (`router._process_choice_vote`)

Uncertain non-maths multiple-choice questions, and yes/no questions whose draft says yes or
no, are settled by **self-consistency**: draw 6 more samples (3 more at a time while the top
answer has under 70% agreement, up to 12), read each answer's option or verdict, and take
the most common. **A tie keeps the draft.** No web search: evidence has nothing to check in
"which premise is valid?".

## Step 3b — Layer 2 for facts (`app/layer2_nli_rag/`)

1. **Claims**: the model splits the draft into up to 6 short, self-contained claims
   (`proposition_parser.py`).
2. **Evidence**: DuckDuckGo search for the question (retried when empty) and for each claim;
   optional Wikipedia intros (off by default) — `search_retriever.py`.
3. **NLI** (`nli_verifier.py`): DeBERTa-v3 scores every passage against the claim; passages
   vote, weighted by source relevance `w` (web 0.7, Wikipedia 0.9):

   `S = Σ w·(P(entail) − P(contradict)) / Σ w`; S ≥ 0.15 supported, S ≤ −0.15 contradicted.

4. **Revise** (`reviser.py`):
   - time-sensitive → answer from today's results (today's date given, the draft not shown);
     no answer in the results → draft + "may be out of date";
   - otherwise **evidence-gated edit**: change only contradicted claims; no contradiction →
     draft unchanged.

## Step 3c — Layer 3 for maths (`app/layer3_halluclean/`)

1. **Exact calculator** (plain arithmetic, percentages, nPr/nCr/n!, °F↔°C, linear
   equations): its value decides; done.
2. **Plan**: a short step list, used as a hint (`planner.py`).
3. **More samples**: 6 to 12, as in Step 3a.
4. **Programs** (`program_verifier.py`): 5 Python programs; for advanced maths they
   alternate exact **sympy** programs and **independent-check** programs (enumeration,
   numeric approximation). Each runs in an isolated subprocess (20 s, whitelisted imports,
   static safety check); crashed programs get one repair attempt; implausible results
   (fractional counts, unexplained negatives) are discarded.
5. **Vote**: score = samples + 0.5 × programs per answer; ties favour the draft. An answer
   no sample gave may win only if ≥ 3 programs from **both** methods agree.
6. The winning sample's text is returned with `Final answer: …`.

## Output and audit trail

`QueryResponse.answer` plus, with `include_audit`, a `PipelineResult`: path taken,
task type, entropy, Layer 2 claims and verdicts or Layer 3 votes and programs, tokens across
all calls, latency, trace ID and degradations (`app/models/schemas.py`).

## Settings that matter (`app/config.py`, override with `DID_*` in `.env`)

| Setting | Value | Effect |
|---|---|---|
| `DID_OLLAMA_MODEL` / `DID_OLLAMA_JUDGE_MODEL` | qwen2.5:3b / same | the one pipeline model |
| `DID_T_SAFE_REASONING` / `DID_T_SAFE` | 0.2 / 0.35 | fast-path thresholds |
| `DID_SE_NUM_SAMPLES` | 4 | Layer 1 samples |
| `DID_L3_EXTRA_SAMPLES`, `DID_L3_MAX_EXTRA_SAMPLES` | 6, 12 | vote samples |
| `DID_PROGRAM_VOTE_WEIGHT` | 0.5 | weight of one program |
| `DID_MCQ_VOTE`, `DID_YES_NO_VOTE` | on | votes instead of Layer 2 |
| `DID_WIKIPEDIA_EVIDENCE` | off | curated evidence (unmeasured) |
| `DID_OLLAMA_THINK` | unset | `false` for thinking models |
| `DID_OLLAMA_NUM_CTX` | 4096 | context window |
