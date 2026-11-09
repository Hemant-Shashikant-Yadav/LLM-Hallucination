# Plan: Objectives 1 and 2 — measuring and classifying uncertainty

Status: **built** (2026-11-09, revision 4). All code for both objectives is written and tested
on the Windows PC with the simulated model; the question sets are frozen in `data/uq_sets/`.
Next: review, then the Mac run (`run-all`). How to run: [README.md](README.md). Where the
build departs from earlier revisions of this plan, the text below says so ("Built:").

The thesis has four objectives. The current pipeline (`app/`, `evaluation/`) covers 3
(evidence-based verification and structured reasoning) and 4 (comparison on faithfulness,
hallucination and cost). This plan adds the other two as a **separate package,
`uncertainty/`**, that never changes or plugs into the pipeline:

| # | Objective | What we will deliver |
|---|---|---|
| 1 | Measure an AI model's baseline uncertainty using an existing algorithm | Semantic entropy (Farquhar et al., 2024) and the standard comparison methods, run on many raw models and a wide range of questions, with a **baseline uncertainty profile** per model: how unsure it is, how well that predicts its mistakes, and where it is wrong while sure |
| 2 | Develop a framework for identifying and classifying uncertainty in AI-generated responses | A tool that takes a question and a model and returns **how unsure** the answer is (a calibrated hallucination probability) and **what kind** of uncertainty it is (knowledge, reasoning, ambiguity/unanswerable, time-dependent), with the action each kind calls for |

---

## 1. Where things run

| | Windows PC (GTX 1650, 4 GB) | Mac |
|---|---|---|
| Role | Write the code, unit tests, dry runs, build the question sets | **Every model run**: generation, judging, interventions, self-diagnosis probes |
| Models loaded | None. The package never loads a model on the PC | One at a time (`OLLAMA_MAX_LOADED_MODELS=1`), with parallel slots |
| NLI (DeBERTa) | CPU, only for tests on small data | Apple GPU (MPS) |
| Analysis and reports | Yes (CPU only) | Yes |

Workflow: **build everything once, run everything once.**

1. **PC: build all the code for both objectives and test it without a model.** The package
   has two backends: `ollama` (real) and `fake`. The fake backend is a simulated model with
   **planted uncertainty types**: some questions it "doesn't know", some it "miscalculates",
   some it finds ambiguous. It produces realistic answers, samples, token log-probs,
   self-evaluation replies and intervention replies. Unit tests and a full dry run
   (`run-all --backend fake`) go through every stage of both objectives, down to the two
   final reports, and check that the Objective 2 classifier recovers the planted types.
2. **PC: freeze the question sets** (needs internet for Hugging Face and Wikipedia) and commit
   them. The Mac then only needs `git pull`, no downloads.
3. **You review once** (code and dry-run reports), then commit and push.
4. **Mac: one command.** `run-all` goes through every stage in §1.1 by itself, model after
   model, and stops early with a clear message if a gate fails.
5. **Results come back through git.** Each (model, set) generation file is gzipped, with
   log-probs rounded to 4 decimals (a few MB each), and committed with its labels, scores and
   reports.
6. **Afterwards, on either machine, no model needed:** the human labels (§9.3) are imported
   and the reports rebuilt in seconds.

### 1.1 The one command on the Mac

Setup (once):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

Run (in a second terminal; `caffeinate -i` keeps the Mac awake):

```bash
caffeinate -i .venv/bin/python -m uncertainty run-all
```

| Stage | What it does | Gate (stops the run if it fails) |
|---|---|---|
| 1. check | Reads the Mac's memory and picks the roster (§4); pulls missing models; records the Ollama version; tests token log-probs (present for greedy and sampled tokens; before or after temperature); runs the unit tests | Ollama reachable; unit tests pass. Missing log-probs do **not** stop the run: token-level methods are switched off and the report says so |
| 2. pilot | 5 questions per set on `qwen2.5:3b`, through every later stage (labels, scores, interventions, a mini Objective 2) | Final answers readable in ≥ 90% of maths, option and yes/no samples; ≤ 10% of samples cut off; the judge returns valid verdicts; NLI runs. Prints and saves the projected total hours. Built: judge failures above 2% are recorded, above 20% (a broken judge) fail the gate, because one unparseable verdict in a small batch already exceeds 2% |
| 3. focus models | For each focus model (§4), `qwen2.5:3b` first: generate (extended tier) → NLI → correctness labels → interventions. Reports rebuilt after each model | Per model: the same readability and cut-off checks on its first 20 questions of each set, and the judge check. Built: a per-model failure **skips that model** (recorded in the summary) instead of stopping the run, so one badly formatted model cannot waste a night; the next `run-all` tries it again. A pilot failure still stops the run |
| 4. first reports | Objective 1 and Objective 2 reports from the focus models | — |
| 5. other models | Every other model in the roster (core tier, smallest first, 12–14B last): generate → NLI → correctness labels. Reports rebuilt after each model | Same per-model checks |
| 6. final reports | Both reports with all models; `results/reports/SUMMARY.md` with the headline numbers of both objectives, timings, and every degradation | — |

- **Resume**: each stage and each question is recorded as done. Running `run-all` again skips
  finished work, so a crash, a reboot or a code fix costs only the question in progress.
- **Usable at any point**: because reports are rebuilt after every model, stopping the run
  early still leaves complete reports for the models finished so far, and the thesis-critical
  part (focus models, interventions, Objective 2) is done first.
- **Time limit** (optional): `--max-hours N`. After the pilot measures speed, if the
  projection exceeds N, the run trims itself in this order: K = 6 for the worked-solution sets,
  then core instead of extended for the third focus model, then the optional models. What was
  trimmed is written to the summary.
- Overrides: `--models`, `--tier`, `--until <stage>`, `--only <stage>`.

---

## 2. Ground rules: how the current code stays untouched

1. All new code lives in `uncertainty/`. Frozen question sets go in `data/uq_sets/`,
   outputs in `uncertainty/results/`.
2. No edits to `app/`, `evaluation/`, `tests/`, `.env`, `requirements.txt` or the docs in
   `doc/`. If a pipeline helper needs a change, it is copied into `uncertainty/`, not edited.
   No new dependencies: everything needed is already in `requirements.txt`.
3. **Read-only imports only**, where reusing is safer than rewriting:
   - `app.answer_utils` for reading final answers (numbers, expressions, options, yes/no).
     The project's worst errors came from answer reading; this code is already debugged.
   - `app.answer_utils.classify_task_heuristic`: the pipeline's current routing rule, used
     as the **baseline** the Objective 2 classifier must beat.
   - `evaluation.truthfulqa_judge.judge_answer`, so TruthfulQA is judged exactly as in
     Objectives 3–4.
4. The package talks to Ollama directly. It does not need the FastAPI server and nothing in
   the pipeline reads its output. The type classifier will output a *recommended action*
   (e.g. "retrieve evidence"), but wiring that into the router is out of scope.
5. Own settings with an `UQ_` prefix (`uncertainty/settings.py`, optional `uncertainty/uq.env`),
   so pipeline `DID_` settings are never read or changed for these runs.
6. Model runs happen on the Mac only, one model loaded at a time, never at the same time as a
   pipeline benchmark. The runner refuses to start if `ollama ps` shows a different model.
7. Work on its own branch (`feat/uncertainty`), committed separately from pipeline changes.
8. Portable code: `pathlib` paths, no OS-specific calls, UTF-8 everywhere, so the same code
   runs on Windows and macOS.

---

## 3. Folder layout

```
uncertainty/
├── PLAN.md                  this file
├── README.md                how to run, index of results
├── __main__.py              CLI: python -m uncertainty <command>  (run-all, check, build-sets,
│                            generate, label, score, intervene, train, report, annotate, analyze)
├── settings.py              UQ_ settings: K, temperatures, token limits, paths
├── roster.py                models: tag, family, size, memory needed, think flag, judge, tier
├── orchestrate/
│   ├── run_all.py           the Mac's one command: stages, done-markers, resume, --max-hours
│   ├── gates.py             pilot and per-model checks that stop a bad run early
│   └── system.py            memory → roster; ollama pull; ollama ps; Ollama version
├── sets/
│   ├── build.py             download pinned dataset versions → data/uq_sets/*.jsonl + MANIFEST.json
│   └── load.py              Item schema, stratified sampling, fixed dev/test split, tiers
├── generate/
│   ├── prompts.py           prompt templates per answer kind, self-evaluation and probe prompts
│   ├── backends.py          OllamaBackend (logprobs, top_logprobs, seeds, think=false)
│   ├── fake.py              FakeBackend: a simulated model with planted uncertainty types
│   └── runner.py            greedy answer + K samples + self-evaluation + probes; resumable .jsonl.gz
├── label/
│   ├── correctness.py       right/wrong for every greedy answer (rules first, judge for free text)
│   ├── judge.py             cross-family LLM judge with a disk cache
│   ├── interventions.py     Objective 2: re-ask with oracle evidence / a reasoning hint
│   └── annotate.py          export items for human labelling, import labels, agreement (κ)
├── methods/                 Objective 1: published uncertainty algorithms (pure functions)
│   ├── nli.py               DeBERTa-v3 NLI pair matrix, batched, cached on disk
│   ├── semantic_entropy.py  discrete SE, likelihood-weighted SE, number of meaning clusters
│   ├── token_level.py       NLL, length-normalised NLL, token entropy, predictive entropy
│   ├── graph.py             spectral methods: EigV, Degree, Eccentricity
│   ├── consistency.py       lexical similarity, agreement with the greedy answer, SelfCheck-NLI
│   └── self_eval.py         P(True), verbalised confidence
├── metrics.py               AUROC, AUPRC, AURAC, PRR, ECE, Brier, bootstrap CIs, DeLong test
├── framework/               Objective 2
│   ├── taxonomy.py          uncertainty types, subtypes, recommended actions
│   ├── features.py          features available at inference time (no gold data)
│   ├── detector.py          calibrated P(hallucination) from the Objective 1 signals
│   ├── classifier.py        rule-based and learned type classifiers
│   └── analyze.py           public API: analyze(question, model) → UncertaintyReport
├── report/
│   ├── obj1.py              tables and figures for Objective 1
│   └── obj2.py              tables and figures for Objective 2
├── tests/                   unit tests (run: python -m unittest discover -s uncertainty/tests -t .)
└── results/                 one folder per model: <set>.jsonl.gz, labels, scores; reports/
```

Each file gets a `*.README.md` note beside it, the same convention as `app/`.

---

## 4. Models

The Mac allows a proper design: **several families, a size ladder inside each family, and old
vs new generations**. `qwen2.5:3b` is required: it is the current pipeline model, so its
profile links Objectives 1–2 to the results of 3–4.

Memory guide (q4 weights): 3–4B ≈ 2–3.5 GB, 7–8B ≈ 5 GB, 12–14B ≈ 8–9 GB, plus cache for the
parallel slots. Ollama can use about 70% of unified memory: about 11 GB on a 16 GB Mac,
17 GB on 24 GB, 22 GB on 32 GB.

| Model (Ollama tag) | Family | Size | Role in the design | Mac memory |
|---|---|---|---|---|
| `qwen2.5:3b` | Qwen | 3B | current pipeline model (required) | any |
| `qwen2.5:7b` | Qwen | 7B | size ladder | any |
| `qwen2.5:14b` | Qwen | 14B | size ladder, top | 24 GB+ |
| `qwen3:4b-instruct-2507-q4_K_M` | Qwen | 4B | newer generation | any |
| `llama3.2:3b` | Llama | 3B | size ladder | any |
| `llama3.1:8b` | Llama | 8B | size ladder; also the Objective 3–4 judge | any |
| `llama2:7b` | Llama | 7B | Phase 1 model, older generation | any |
| `gemma3:4b` | Gemma | 4B | size ladder | any |
| `gemma3:12b` | Gemma | 12B | size ladder, top | tight on 16 GB; fine on 24 GB+ |
| `phi4-mini` | Phi | 3.8B | size ladder | any |
| `phi4` | Phi | 14B | size ladder, top | 24 GB+ |
| `mistral:7b` | Mistral | 7B | optional fifth family | any |

- **Picked automatically**: `run-all` reads the Mac's memory and chooses the roster, so no
  decision is needed in advance. `--models` overrides it.
- **16 GB Mac**: the nine models marked "any" (four families; Qwen 3→7B and Llama 3→8B ladders;
  old vs new generation in both). Built: `gemma3:12b` is picked automatically only from 24 GB;
  on 16 GB it can be added with `--models` if it fits.
- **24 GB or more**: all twelve, with three-step ladders in Qwen (3→7→14B) and two-step
  ladders in Llama, Gemma and Phi.
- If the pipeline moves to a new model on the Mac (HANDOFF step 3), that model joins the roster
  as a focus model.
- Tags are checked against ollama.com/library before pulling; thinking models run with
  `think=false`.

**Focus models** get the larger question set and the Objective 2 interventions:
`qwen2.5:3b`, `llama3.1:8b` (another family, the largest model on any Mac), and the pipeline's
new Mac model if there is one (otherwise `gemma3:4b`, a third family).

**Judges** (correctness of free-text answers): never the model under test, and from a
**different family**. `llama3.1:8b` judges Qwen, Gemma, Phi and Mistral models (the same judge
as Objectives 3–4). A Qwen model judges the Llama models: `qwen2.5:7b` on a 16 GB Mac,
`qwen2.5:14b` on 24 GB+. The agreement check (§7) adds a second judge from a third family:
`gemma3:12b` where it fits, else `mistral:7b` (and for Gemma or Mistral models under test,
the other of the two). Judging runs after generation, never in parallel with it.

---

## 5. Question sets

Each set is chosen for the kind of uncertainty it exercises, so Objective 2 has examples of
every type. All are frozen at pinned Hugging Face revisions with a manifest and SHA-256, the
same way `evaluation/export_eval_sets.py` does it. Each set is split once (seed 7) into
**dev 30% / test 70%**; anything tuned is tuned on dev only.

Two tiers. **Core** runs on every model. **Extended** (a superset of core) runs on the focus
models, for tighter confidence intervals where the thesis needs them most.

| Domain | Dataset | Core N | Extended N | Answer kind | Mainly probes | Oracle material for Obj 2 |
|---|---|---|---|---|---|---|
| Popular facts | TriviaQA (`rc.nocontext`) | 150 | 300 | short text | knowledge | evidence passages |
| Long-tail facts | PopQA (stratified by popularity) | 150 | 300 | short text | knowledge, by how famous the subject is | Wikipedia intro of the subject (fetched once, frozen) |
| Multi-hop facts | HaluEval-QA (HotpotQA questions) | 150 | 300 | short text | knowledge + composition | `knowledge` field |
| Misconceptions | TruthfulQA (already frozen) | 150 | 817 (all) | short text | confident myths | best correct answer |
| Arithmetic word problems | GSM8K (already frozen; re-export with rationales) | 120 | 250 | number | reasoning | reference method (numbers removed) |
| Competition maths | MATH, equal per level 1–5 (already frozen) | 120 | 250 | expression | reasoning, by difficulty | reference solution, final step removed |
| Multi-step commonsense | StrategyQA (facts and decompositions joined from the official release file) | 120 | 250 | yes/no | knowledge vs reasoning, separable | `facts` and `decomposition` fields |
| Science MCQ | ARC-Challenge | 100 | 200 | option | mixed | — (Objective 1 only) |
| Ambiguous questions | AmbigQA | 100 | 200 | short text | aleatoric: ambiguous | the list of readings |
| Unanswerable questions | SelfAware (2 unanswerable : 1 answerable) | 150 | 300 | short text | aleatoric: unanswerable (incl. subjective) | answerable flag |
| Changing facts, false premises | FreshQA (snapshot of 2025-09-30) | 100 | 200 | short text | temporal; false premise | fast/slow/never-changing and false-premise labels |
| **Total per model** | | **1,410** | **3,367** | | | |

Notes:
- Five of these are already frozen or cached in the repo (TruthfulQA, GSM8K, MATH, ARC,
  StrategyQA). The new ones are TriviaQA, PopQA, HaluEval, AmbigQA, SelfAware, FreshQA;
  their exact sources, licences and revisions are fixed in phase B.
- FreshQA's "correct" answers are tied to its snapshot date, so we use it mainly for its
  **type labels** (time-dependent, false premise), and score correctness against that snapshot.

---

## 6. Generation protocol (once per model; every method reads the stored file)

**Generate once, score many.** Every uncertainty method in Objective 1, and every Objective 2
feature, is computed offline from one stored file per model and set. Adding a method later
costs no model time, and the Mac is needed only for generation.

Per question:

1. **Greedy answer** (T = 0) with token log-probabilities and the top-5 alternatives per
   token. This is the answer whose correctness and uncertainty we measure.
2. **K = 10 samples** at T = 1.0 with fixed seeds, with the log-probability of every
   sampled token. (K and T follow Farquhar et al., 2024. Smaller K is studied by
   subsampling these 10.)
3. **P(True)** call (Kadavath et al., 2022): the question, the samples as "brainstormed
   ideas", the greedy answer, "Is the proposed answer (A) True (B) False?"; we read the
   probability of A from the first token's top log-probs.
4. **Verbalised confidence** call (Tian et al., 2023): "How confident are you, 0–100, that
   this answer is correct?"
5. **Self-diagnosis probes** for Objective 2 (one token each, probability read from
   log-probs): "Is this question ambiguous?", "Does the answer depend on today's date?",
   "Does it need a calculation?", "Does it rest on a false assumption?", and (built) "Is it
   impossible to answer with certainty?" Done in the same run so the Mac never needs a second
   pass for them.

Prompts per answer kind (template id stored with each record):

| Kind | Prompt |
|---|---|
| short text | "Answer the following question in a single brief but complete sentence." (the semantic-entropy papers' prompt; keeps NLI clustering and judging reliable) |
| number / expression | the question, then "Show your working, then end with 'Final answer: …'" |
| option | question with options, then "End with 'Answer: <letter>'" |
| yes/no | the question, then "End with 'Answer: yes' or 'Answer: no'" |

Samples cut off at the token limit are kept in the file but flagged, and do not count as an
answer (the pipeline's lesson: a cut-off maths sample has no final answer).

Stored record (one line per question in `results/<model>/<set>.jsonl.gz`):

```json
{"id": "tqa_12", "set": "truthfulqa", "split": "test", "tier": "core", "model": "qwen2.5:3b",
 "backend": "ollama", "ollama_version": "...", "template": "short_v1", "question": "...",
 "greedy": {"text": "...", "tokens": [...], "logprobs": [...], "top_logprobs": [[...]],
            "done_reason": "stop", "eval_count": 31, "seconds": 1.4},
 "samples": [{"seed": 1001, "text": "...", "logprobs": [...], "done_reason": "stop"}],
 "p_true": {"p_a": 0.81, "raw": "..."}, "verbal": {"value": 90, "raw": "..."},
 "probes": {"ambiguous": 0.12, "time_dependent": 0.03, "calculation": 0.01, "false_premise": 0.05},
 "cost": {"calls": 17, "prompt_tokens": 2210, "completion_tokens": 406, "seconds": 22.7}}
```

The runner appends as it goes and skips finished ids, so an interrupted run resumes.

---

## 7. Correctness labels (is the greedy answer a hallucination?)

| Answer kind | How it is scored |
|---|---|
| number, expression | `app.answer_utils` equivalence against the key (as `evaluation/hard_eval.py`) |
| option, yes/no | the option / verdict read by `app.answer_utils`; unreadable → the judge says which way it answers |
| short text | the cross-family judge, given the question, the gold answers (all aliases) and the answer; an alias-match check is run alongside and its agreement with the judge is reported |
| TruthfulQA | `evaluation.truthfulqa_judge.judge_answer` (same judge and prompt as Objectives 3–4) |
| unanswerable / false premise | correct = the answer says it cannot be answered or corrects the premise (judge) |

Quality checks: the second judge on a 300-item subset (agreement and Cohen's κ, as
`evaluation/judge_agreement.py` did for Phase 1), and a 100-item human audit of judge labels.

---

## 8. Objective 1: baseline uncertainty with existing algorithms

### 8.1 Methods

The **primary algorithm is semantic entropy**, the method the thesis builds on and the one
Layer 1 uses. The others are the standard comparison methods from the literature, so the
thesis can say how good it is relative to alternatives.

| Family | Method | Needs | Reference |
|---|---|---|---|
| Meaning-level sampling | **Discrete semantic entropy** (cluster frequencies) | samples + NLI | Farquhar et al., 2024 |
| | Likelihood-weighted semantic entropy | + sequence log-probs | Kuhn et al., 2023 |
| | Number of meaning clusters | samples + NLI | Lin et al., 2024 |
| | Spectral: EigV, Degree, Eccentricity | NLI similarity matrix | Lin et al., 2024 |
| | Agreement of samples with the greedy answer | samples | self-consistency (Wang et al., 2023) |
| Token-level (needs log-probs) | Sequence NLL and length-normalised NLL | greedy log-probs | Malinin & Gales, 2021 |
| | Mean and max token entropy (over top-5) | greedy top log-probs | |
| | Predictive entropy (Monte Carlo over samples) | sample log-probs | Kuhn et al., 2023 |
| Surface sampling | Lexical similarity (mean ROUGE-L between samples) | samples | Lin et al., 2024 |
| Self-evaluation | P(True) | one extra call | Kadavath et al., 2022 |
| | Verbalised confidence | one extra call | Tian et al., 2023; Xiong et al., 2024 |
| Long answers (optional) | SelfCheckGPT-NLI per sentence | samples + NLI | Manakul et al., 2023 |

Meaning equivalence follows Layer 1: for numbers, expressions, options and yes/no, two
answers mean the same if their read answers are equal; for free text, if they entail each
other both ways under DeBERTa-v3 NLI (`cross-encoder/nli-deberta-v3-base`, the model Layer 1
uses), with the question prepended. One full NLI matrix per free-text question (110 ordered
pairs at K = 10) is computed once, cached, and shared by every NLI-based method.

### 8.2 How each method is judged

The positive class is a **wrong greedy answer**. For each model × set × method:

| Metric | Question it answers |
|---|---|
| AUROC (primary) | Does the method give wrong answers higher uncertainty than right ones? |
| AUPRC | Same, robust when errors are rare |
| AURAC and accuracy at 80% / 90% coverage | If the model refuses its most uncertain answers, how accurate is the rest? (used by Farquhar et al.) |
| PRR | Share of the ideal refusal gain the method achieves |
| ECE (15 bins), Brier, reliability diagram | For the scores that claim to be probabilities (P(True), verbalised, the Obj 2 detector): is a "70%" answer right 70% of the time? |
| Cost | Calls, tokens and seconds per method (feeds Objective 4's cost axis) |

Statistics: 95% bootstrap CIs (1,000 resamples, paired across methods), DeLong test for
AUROC differences between methods, Holm correction across comparisons. Reported on the test
split.

### 8.3 Analyses and outputs

1. **Method table**: AUROC (with CI) of every method on every model × set; a heatmap.
2. **Baseline uncertainty profile** per model and domain: accuracy; mean semantic entropy;
   share of answers above the "high uncertainty" threshold; and the **blind spot**: errors
   made with low entropy (confident hallucinations, which no sampling method can catch).
3. **Size, family and generation effects**: along each size ladder (Qwen 3→7→14B, Llama 3→8B,
   Gemma 4→12B, Phi 3.8→14B), and old vs new generation (llama2 vs llama3.1, qwen2.5 vs
   qwen3): do bigger or newer models have lower uncertainty, better-ranked uncertainty and
   fewer confident errors?
4. **How many samples are needed**: AUROC against K = 2, 4, 6, 8, 10 by subsampling. This
   checks the pipeline's choice of K = 4.
5. **Pipeline operating point**: AUROC of Layer 1's stored H_sem on the existing GSM8K and
   TruthfulQA test runs (`evaluation/results/test_qwen3b_v5`, `test_qwen3b_v6`), computed
   from the stored CSVs with no model, set beside the new measurements for qwen2.5:3b.
6. Figures: ROC curves, risk–coverage curves, reliability diagrams, entropy histograms split
   by right/wrong, AUROC-vs-K, AUROC-vs-cost, AUROC-vs-model-size.

Deliverable: `uncertainty/results/reports/obj1/` (tables as CSV + LaTeX, figures as PNG,
`summary.json`) and a written results page.

---

## 9. Objective 2: a framework to identify and classify uncertainty

### 9.1 Two axes

The framework answers two separate questions about a response:

- **Identify — how unsure?** A calibrated probability that the answer is a hallucination,
  plus a level (low / medium / high) with thresholds set on dev.
- **Classify — what kind of uncertainty?** One of four types (below), with a subtype for
  aleatoric. The type says *why* the model is unsure, and so *what would fix it*.

### 9.2 Taxonomy

| Type | Meaning | Typical signature | Recommended action |
|---|---|---|---|
| **Knowledge** (epistemic, factual) | The model lacks or misremembers a fact | samples disagree on names, dates, entities; giving the evidence settles it | retrieve evidence (what Layer 2 does) |
| **Reasoning** (epistemic, procedural) | The facts are known or given, the derivation is unstable | samples share premises and intermediate quantities but reach different conclusions; a method hint settles it | structured reasoning / programs (what Layer 3 does) |
| **Aleatoric** — subtypes: ambiguous, unanswerable, false premise (built: subjective questions count as unanswerable; SelfAware's file has no category labels to separate them) | The question itself has no single true answer | several clusters, each confident and self-consistent; evidence does not settle it | clarify, list readings, abstain, or correct the premise |
| **Temporal** | The true answer changes over time | time cues; agreement can still be out of date | fresh retrieval with a date |

Plus a flag, **overconfidence risk**: low entropy but low P(True) or verbalised confidence,
the only handle on confident hallucinations. Grounded in the epistemic/aleatoric literature
for LLMs (Hou et al., 2024; Yadkori et al., 2024) and in the thesis's own knowledge/reasoning
split.

### 9.3 Where the type labels come from

Three sources, combined in this order of precedence:

1. **Human gold labels** on about 300 items, stratified by set, model and predicted type,
   labelled with a short decision-tree guide (`label/annotate.py` exports a CSV and reads it
   back). Two annotators if possible, reporting Cohen's κ. Used only for evaluation. This is
   the **one manual step**: it happens after the Mac run, on either machine, and
   `report --human-labels <file>` adds the agreement figures in seconds. The Objective 2
   report is complete without it; the human labels add the validation the thesis needs.
2. **Dataset metadata**: AmbigQA → ambiguous; SelfAware unanswerable → unanswerable;
   FreshQA → temporal (fast/slow-changing) or false premise.
3. **Interventions** (the core idea): *the type of uncertainty is whichever intervention
   removes it.* For each uncertain or wrong answer, re-ask the same model, with 5 samples:
   - **with oracle evidence** (TriviaQA passages, PopQA Wikipedia intro, HaluEval knowledge,
     StrategyQA facts, TruthfulQA best answer);
   - **with a reasoning hint**: built as the first half of the reference method, never its
     last step (GSM8K rationale, MATH solution, StrategyQA decomposition).
   Evidence fixes it and the hint does not → **knowledge**; the hint fixes it and evidence
   does not → **reasoning**; both → **mixed** (reported, not used for training); neither →
   **unresolved** (reported). When a set offers only one intervention, it decides alone.
   "Fixes it" = the majority answer becomes correct and the samples agree: entropy at or below
   the low-uncertainty threshold, or (built) at most one dissenting sample, since with 5
   samples a 4–1 split already gives entropy ≈ 0.31.

Interventions run on the Mac for the focus models (§4), which span at least two families,
right after each focus model's answers are labelled. The items are picked automatically: a
wrong answer, or semantic entropy above the threshold tuned on dev.
StrategyQA and HaluEval matter most here: they are the sets where knowledge and reasoning
failures occur side by side, so the classifier cannot just learn "maths → reasoning".

Answers that are both low-entropy and correct have no uncertainty to classify; they count for
the Identify axis only.

### 9.4 Features (available at inference, no gold data)

- All Objective 1 scores.
- Cluster structure: number of clusters, largest-cluster share, greedy-in-majority, **mean
  log-prob inside each cluster** (confident-but-different clusters suggest aleatoric;
  scattered low-likelihood answers suggest epistemic).
- **Premise vs conclusion agreement**: overlap of the entities and intermediate numbers
  across samples, compared with agreement on the final answer (high premise agreement + low
  conclusion agreement suggests reasoning).
- Answer kind and the pipeline's task type (`classify_task_heuristic`, read-only).
- Text cues: dates and "current/latest", hedges, refusals, "as of my knowledge".
- The four self-diagnosis probes from the generation run (§6, step 5).

### 9.5 Models and evaluation

- **Detector**: logistic regression over the Objective 1 scores, calibrated (Platt or
  isotonic) → P(hallucination). Compared with the best single method (expected: semantic
  entropy) by AUROC, AURAC and ECE.
- **Type classifier**, three versions compared:
  1. baseline: the pipeline's current rule (maths → reasoning, time cues → temporal, else
     knowledge);
  2. transparent rules: a small hand-readable decision tree with thresholds tuned on dev;
  3. learned: multinomial logistic regression and gradient-boosted trees, with feature
     importances.
- Metrics: macro-F1 and per-class precision/recall, confusion matrix; agreement with the human
  gold labels (κ).
- Generalisation: **leave-one-dataset-out**, **leave-one-model-out** and
  **leave-one-family-out**, so the classifier is shown to read uncertainty, not dataset or
  model identity.
- **Actionability check** (offline): on items where both interventions were run, does the
  intervention that matches the predicted type help more than the one that does not? This is
  the bridge to Objective 3 without touching the pipeline.

### 9.6 The tool

```python
from uncertainty.framework.analyze import analyze

report = await analyze("Who was the first person to walk on the Moon?", model="qwen2.5:3b", mode="full")
```

```json
{"question": "...", "model": "qwen2.5:3b", "answer": "Neil Armstrong ...",
 "identify": {"p_hallucination": 0.08, "level": "low", "semantic_entropy": 0.0},
 "classify": {"type": "knowledge", "subtype": null,
              "type_probs": {"knowledge": 0.71, "reasoning": 0.06, "aleatoric": 0.15, "temporal": 0.08},
              "overconfidence_risk": false},
 "signals": {"p_true": 0.93, "verbal": 95, "eigv": 1.0, "nll_norm": 0.12, "...": "..."},
 "recommended_action": "accept",
 "cost": {"calls": 19, "tokens": 680, "seconds": 18.2}}
```

Modes: `lite` (K = 4, no extra calls; the pipeline's budget) and `full` (K = 10, all
signals). CLI: `python -m uncertainty analyze "question" --model qwen2.5:3b`. It runs wherever
Ollama has the model, which in practice is the Mac.

Deliverable: `uncertainty/results/reports/obj2/` (confusion matrices, per-class scores,
feature importances, an example gallery of each type) and the tool itself.

---

## 10. Compute budget (Mac)

Generated tokens per model, K = 10:

| Part | Core questions | Core tokens | Extended questions | Extended tokens |
|---|---|---|---|---|
| Short-text sets (6) + TruthfulQA | 950 | 0.28 M | 2,417 | 0.78 M |
| GSM8K + MATH (worked solutions) | 240 | 0.79 M | 500 | 1.65 M |
| StrategyQA + ARC (worked answers) | 220 | 0.36 M | 450 | 0.74 M |
| **Total** | **1,410** | **≈1.4 M** | **3,367** | **≈3.2 M** |

Whole programme on a 16 GB Mac (9 models: 3 focus models extended, 6 core): about
**18 M tokens**, plus about 2 M for the interventions. A 24 GB+ Mac adds 3 larger models
(about 4 M more, at lower speed).

Time = tokens ÷ throughput. As a guide, 1 M tokens takes about 7 h at 40 tokens/s and 14 h at
20 tokens/s. Small models with 4 parallel slots should sit at the fast end, 12–14B models at
the slow end. That makes it roughly **a week of unattended running** in total. The pilot
(stage 2) measures the real speed and saves the projected hours before anything long starts.

**Order** (§1.1), so the most important results arrive first:
1. `qwen2.5:3b`, extended tier, with interventions (the pipeline model; headline results).
2. The other focus models, extended, with interventions. First complete reports here.
3. The remaining models, core tier, smallest first.
4. The 12–14B models (24 GB+ Macs).

If time runs short, `--max-hours` trims automatically: K = 6 for the worked-solution sets
first (they are over half the tokens), then the third focus model's extended tier, then the
optional models.

---

## 11. Phases

All the code for both objectives is written in one go on the PC (phases A–E). There is
**one review** before the Mac, and **no trip back** to the PC during the Mac run.

| Phase | Where | Work | Done when |
|---|---|---|---|
| **A. Scaffold** | PC | Package skeleton, settings, roster, CLI, Ollama backend, fake backend with planted types, unit-test setup | Tests pass; the fake backend writes valid records |
| **B. Question sets** | PC | `sets/build.py` for all 11 sets, pinned revisions, manifest, dev/test split, tiers; PopQA Wikipedia intros frozen | `data/uq_sets/` built and reproducible from the manifest |
| **C. Objective 1 code** | PC | Runner, NLI, all methods, correctness rules and judge client, metrics, Objective 1 report; Layer 1 AUROC from the stored pipeline CSVs | Unit tests: metrics match scikit-learn, entropy matches hand-worked examples, answer reading on known cases |
| **D. Objective 2 code** | PC | Taxonomy, self-diagnosis probes, interventions, features, detector, classifiers, evaluation, Objective 2 report, `analyze` tool, annotation export/import | On the fake model, the classifier recovers the planted types |
| **E. One command and dry run** | PC | `run-all`, gates, resume, `--max-hours`, roster from memory, summary; README and per-file notes | `run-all --backend fake` produces both reports and `SUMMARY.md`; stopping it mid-run and starting again resumes cleanly |
| **Review** | PC | You read the code and the dry-run reports; then commit and push | Your go-ahead |
| **F. Mac run** | Mac | `git pull`, setup, `run-all` (§1.1); commit the results | `SUMMARY.md` and both reports for every model |
| **G. Human labels and write-up** | either | ~300 Objective 2 labels and a 100-item audit of judge labels; `report --human-labels`; thesis text for both objectives | Objective 2 claim tested: the classifier beats the pipeline's rule on macro-F1 with a CI that excludes 0; human κ reported |

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| The code meets a real model for the first time on the Mac | The check and pilot stages run every later stage on a few questions within the first half hour, and stop with a clear message on any problem; stages resume, so a fix is `git pull` and run again with nothing lost; optional smoke test on the PC (decision 5) |
| Code that works on Windows fails on macOS | `pathlib`, no OS-specific calls; the fake backend runs the whole chain on the PC; the check stage runs the unit tests on the Mac first |
| Ollama on the Mac behaves differently (version, log-probs missing for samples, log-probs after temperature) | The check stage records the version and tests log-probs before any run; if they are missing, token-level methods switch off automatically and the sampling methods carry on |
| Runs take longer than a week | Focus models and Objective 2 come first; reports are rebuilt after every model; resumable; `--max-hours` trims automatically |
| Mac sleeps or reboots mid-run | `caffeinate -i`; append-only files; the queue resumes where it stopped |
| Generation files too big for git | gzip + rounded log-probs (a few MB per model and set); if the total passes ~1 GB, keep raw files on a drive and commit only labels, scores and reports |
| Judge noise (llama3.1:8b already seen flipping verdicts) | Rules wherever possible; second judge and human audit on subsets; cache verdicts |
| Classifier learns dataset or model identity instead of uncertainty | Mixed sets (StrategyQA, HaluEval); leave-one-dataset/model/family-out evaluation; dataset name never a feature |
| Interventions leak the answer, making every item "knowledge" | Hints carry the method without numbers; evidence is a passage, not the bare answer, where the set allows |
| FreshQA answers drift | One dated snapshot; used mainly for type labels |
| Answer-reading errors (the pipeline's biggest past bug) | Reuse `app.answer_utils`; explicit "Final answer:" prompts; unreadable answers logged and spot-checked |

---

## 13. Decisions needed from you

The Mac's memory is no longer a decision: `run-all` reads it and picks the roster.

1. **Tiers**: extended set for the three focus models and core for the rest, or extended for
   every model.
2. **Human labels**: can you (and ideally a second person) label about 300 items for
   Objective 2 after the Mac run, about 2–3 hours each with the guide?
3. **Primary algorithm**: semantic entropy as "the existing algorithm" for Objective 1, with
   the others as comparisons. (Recommended: it is the method Layer 1 already uses.)
4. **Moving results**: commit the gzipped generation files to git, or keep them on a drive and
   commit only labels, scores and reports.
5. **Smoke test on this PC**: before the review, run 3 questions through the real backend with
   `qwen2.5:3b` (already pulled; about 2 minutes; nothing else running). It is the only way to
   meet a real model before the Mac. You asked not to run the code here, so the default is no;
   the Mac's check and pilot stages catch the same problems in their first half hour.
6. **Time limit** for the Mac run (`--max-hours`), or none.

Defaults if you don't say otherwise: 1 = focus models extended, 2 = one annotator, 3 = yes,
4 = git, 5 = no, 6 = none.

---

## References

- Farquhar, Kossen, Kuhn, Gal (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*.
- Kuhn, Gal, Farquhar (2023). Semantic uncertainty: linguistic invariances for uncertainty estimation in natural language generation. *ICLR*.
- Lin, Trivedi, Sun (2024). Generating with confidence: uncertainty quantification for black-box large language models. *TMLR*.
- Malinin, Gales (2021). Uncertainty estimation in autoregressive structured prediction. *ICLR*.
- Kadavath et al. (2022). Language models (mostly) know what they know. *arXiv*.
- Tian et al. (2023). Just ask for calibration. *EMNLP*. · Xiong et al. (2024). Can LLMs express their uncertainty? *ICLR*.
- Manakul, Liusie, Gales (2023). SelfCheckGPT. *EMNLP*.
- Wang et al. (2023). Self-consistency improves chain of thought reasoning. *ICLR*.
- Hou et al. (2024). Decomposing uncertainty for LLMs through input clarification ensembling. *ICML*.
- Yadkori et al. (2024). To believe or not to believe your LLM. *arXiv*.
- Datasets: TriviaQA (Joshi et al., 2017), PopQA (Mallen et al., 2023), HaluEval (Li et al., 2023),
  TruthfulQA (Lin et al., 2022), GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021),
  StrategyQA (Geva et al., 2021), ARC (Clark et al., 2018), AmbigQA (Min et al., 2020),
  SelfAware (Yin et al., 2023), FreshQA (Vu et al., 2023).
