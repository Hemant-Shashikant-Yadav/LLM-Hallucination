# uncertainty/sets/build.py — freezing the question sets

Builds `data/uq_sets/*.jsonl` and `MANIFEST.json` (PLAN §5) from pinned sources: Hugging
Face revisions, a GitHub commit, or the repo's frozen `data/eval_sets/full/`. Seeded
sampling: a rebuild gives the same files (checked: identical SHA-256).

| Part | What it does |
|---|---|
| `SIZES`, `SOURCES` | Core and extended size per set; source, revision and licence. |
| `_stratified`, `_tiers` | Even spread across strata (popularity, MATH level, answer); core subset and dev/test split, both seeded per set. |
| `_first_half` | The reasoning hint: the first half of the reference steps, never the last. |
| `build_triviaqa` | Wikipedia-evidence questions; evidence = the sentence naming the answer, one sentence either side. |
| `build_popqa`, `_wikipedia_intros` | Five popularity bins; evidence = the subject's Wikipedia intro, 20 titles per API request (the robot policy needs a contact URL in the user agent). |
| `build_halueval`, `build_truthfulqa`, `build_gsm8k`, `build_math` | Evidence or hint per set (HaluEval knowledge; TruthfulQA best answer; GSM8K rationale; MATH solution, with diagrams and very long problems excluded). |
| `build_strategyqa` | ChilleD test questions joined to the official facts (evidence) and decompositions (hint). |
| `build_arc`, `build_ambigqa`, `build_selfaware`, `build_freshqa` | Options as (a)–(d); ambiguous questions only; 2 unanswerable : 1 answerable (TriviaQA/HotpotQA sources); FreshQA false premises and changing facts. |
| `build` | Writes the files and merges their entries into the manifest (read at the end, so builds of different sets never drop each other). |
