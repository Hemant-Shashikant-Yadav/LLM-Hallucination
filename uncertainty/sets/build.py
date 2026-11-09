"""
Freeze the 11 question sets (PLAN §5) into data/uq_sets/ with a manifest.

Every source is pinned: a Hugging Face revision, a GitHub commit, or one of the repo's
already-frozen files in data/eval_sets/full/. Sampling is seeded, so a rebuild gives the
same files; MANIFEST.json records sources, revisions, licences, counts and SHA-256.

Each file holds a set's extended tier; `core` marks the subset every model answers, and
`split` the fixed dev (30%) / test (70%) assignment. Strata (PopQA popularity, MATH level,
answer balance) are kept in both tiers.

Usage:
  python -m uncertainty build-sets                 # all sets
  python -m uncertainty build-sets --only popqa    # one set
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from loguru import logger

from uncertainty.sets.load import SETS, Item, write_set
from uncertainty.settings import PROJECT_ROOT, settings

FROZEN = PROJECT_ROOT / "data" / "eval_sets" / "full"
LETTERS = "abcdefgh"

# (core N, extended N) per set
SIZES = {
    "triviaqa": (150, 300), "popqa": (150, 300), "halueval": (150, 300), "truthfulqa": (150, 817),
    "gsm8k": (120, 250), "math": (120, 250), "strategyqa": (120, 250), "arc": (100, 200),
    "ambigqa": (100, 200), "selfaware": (150, 300), "freshqa": (100, 200),
}

SOURCES = {
    "triviaqa": {"source": "mandarjoshi/trivia_qa (rc.wikipedia, validation)",
                 "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f", "license": "Apache-2.0"},
    "popqa": {"source": "akariasai/PopQA (test.tsv); evidence: English Wikipedia page summaries",
              "revision": "098765c79ea10a2cb19c828324e33281b8336ec0",
              "license": "MIT (PopQA); CC BY-SA 4.0 (Wikipedia text)"},
    "halueval": {"source": "pminervini/HaluEval (qa)", "revision": "12a856119f03975a94509091e8cada3e6be6ead7",
                 "license": "MIT"},
    "truthfulqa": {"source": "data/eval_sets/full/truthfulqa.jsonl (truthfulqa/truthful_qa generation)",
                   "revision": "741b8276f2d1982aa3d5b832d3ee81ed3b896490", "license": "Apache-2.0"},
    "gsm8k": {"source": "openai/gsm8k (main, test)", "revision": "740312add88f781978c0658806c59bc2815b9866",
              "license": "MIT"},
    "math": {"source": "data/eval_sets/full/math_test.jsonl (EleutherAI/hendrycks_math)",
             "revision": "21a5633873b6a120296cce3e2df9d5550074f4a3", "license": "MIT"},
    "strategyqa": {"source": "ChilleD/StrategyQA (test) + official facts/decompositions from "
                             "voidful/StrategyQA strategyqa_train.json",
                   "revision": "705562638fe1d8ca6bb98c66fc8f94d45fda8c83 + 2279eaf9f2580aef77ed6fa0efd7846c381ab5a0",
                   "license": "MIT"},
    "arc": {"source": "allenai/ai2_arc (ARC-Challenge, test)",
            "revision": "210d026faf9955653af8916fad021475a3f00453", "license": "CC BY-SA 4.0"},
    "ambigqa": {"source": "sewon/ambig_qa (light, validation; multipleQAs only)",
                "revision": "e969d0132f4dd28c2939d55be34f1788c00ccfe7", "license": "CC BY-SA 3.0"},
    "selfaware": {"source": "github.com/yinzhangyue/SelfAware data/SelfAware.json",
                  "revision": "6b15164a3e11a2be714afde0615879dcde5c01f8", "license": "CC BY-SA 4.0"},
    "freshqa": {"source": "bojanbabic/freshqa_09302025 (FreshQA snapshot of 2025-09-30, test)",
                "revision": "ae293206abcacde7a86ee2251a1909e8b0e7edb8", "license": "Apache-2.0 (FreshLLMs)",
                "note": "Answers are true as of the snapshot date; used mainly for type labels"},
}


# ============================================================================ helpers


def _hub_file(repo: str, filename: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def _stratified(rows: list, n: int, key, rng: random.Random) -> list:
    """n rows spread as evenly as possible across strata (each stratum shuffled first)."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups)
    picked: list = []
    while len(picked) < n and any(groups[k] for k in order):
        for k in order:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    return picked


def _tiers(items: list[Item], name: str, strata=None) -> list[Item]:
    """Mark the core subset (stratified) and the dev/test split, both seeded per set."""
    core_n, _ = SIZES[name]
    rng = random.Random(f"{settings.split_seed}-{name}-core")
    core = {it.id for it in _stratified(items, core_n, strata or (lambda it: 0), rng)}
    rng = random.Random(f"{settings.split_seed}-{name}-split")
    ids = [it.id for it in items]
    rng.shuffle(ids)
    dev = set(ids[: round(settings.dev_fraction * len(ids))])
    for it in items:
        it.core = it.id in core
        it.split = "dev" if it.id in dev else "test"
    return items


def _first_half(steps: list[str]) -> str | None:
    """A partial method: the first half of the reference steps (never the last one)."""
    steps = [s.strip() for s in steps if s.strip()]
    if len(steps) < 2:
        return None
    return " ".join(steps[: max(1, len(steps) // 2)])


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


# ============================================================================ sets


def build_triviaqa() -> list[Item]:
    import pyarrow.parquet as pq

    src = SOURCES["triviaqa"]
    path = _hub_file("mandarjoshi/trivia_qa", "rc.wikipedia/validation-00000-of-00001.parquet", src["revision"])
    pf = pq.ParquetFile(path)
    qids = pf.read(columns=["question_id"]).column("question_id").to_pylist()
    rng = random.Random(f"{settings.split_seed}-triviaqa")
    wanted = set(rng.sample(qids, min(len(qids), SIZES["triviaqa"][1] * 2)))
    candidates = []
    for batch in pf.iter_batches(batch_size=256, columns=["question_id", "question", "answer", "entity_pages"]):
        for row in batch.to_pylist():
            if row["question_id"] not in wanted:
                continue
            aliases = [a for a in row["answer"]["aliases"] if a]
            evidence = _alias_passage(row["entity_pages"]["wiki_context"], aliases)
            if evidence:
                candidates.append(Item(
                    id=f"tqa-{row['question_id']}", set="triviaqa", kind="short", question=row["question"],
                    answers=list(dict.fromkeys([row["answer"]["value"], *aliases]))[:12],
                    evidence=evidence, meta={"answer_type": row["answer"]["type"]},
                ))
    candidates.sort(key=lambda it: it.id)
    rng.shuffle(candidates)
    return _tiers(candidates[: SIZES["triviaqa"][1]], "triviaqa")


def _alias_passage(contexts: list[str], aliases: list[str], max_chars: int = 600) -> str | None:
    """The sentence naming the answer, with one sentence either side: an oracle passage."""
    lowered = [a.lower() for a in aliases if len(a) >= 2]
    for ctx in contexts:
        sents = _sentences(ctx[:200_000])
        for i, s in enumerate(sents):
            low = s.lower()
            if any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", low) for a in lowered):
                return " ".join(sents[max(0, i - 1): i + 2])[:max_chars]
    return None


def build_popqa() -> list[Item]:
    import pandas as pd

    src = SOURCES["popqa"]
    df = pd.read_csv(_hub_file("akariasai/PopQA", "test.tsv", src["revision"]), sep="\t")
    df["pop_bin"] = pd.qcut(df["s_pop"].map(lambda p: math.log10(max(p, 1))), 5, labels=False, duplicates="drop")
    rng = random.Random(f"{settings.split_seed}-popqa")
    rows = _stratified(df.to_dict("records"), SIZES["popqa"][1], lambda r: int(r["pop_bin"]), rng)
    intros = _wikipedia_intros([r["s_wiki_title"] for r in rows])
    items = [Item(
        id=f"popqa-{r['id']}", set="popqa", kind="short", question=r["question"],
        answers=json.loads(r["possible_answers"]), evidence=intros.get(r["s_wiki_title"]),
        meta={"s_pop": int(r["s_pop"]), "pop_bin": int(r["pop_bin"]), "prop": r["prop"],
              "subject": r["subj"], "wiki_title": r["s_wiki_title"]},
    ) for r in rows]
    return _tiers(items, "popqa", strata=lambda it: it.meta["pop_bin"])


def _wikipedia_intros(titles: list[str], batch: int = 20) -> dict[str, str]:
    """Plain-text intro of each English Wikipedia article (20 titles per request, redirects followed)."""
    import httpx

    url = "https://en.wikipedia.org/w/api.php"
    # Wikimedia's robot policy asks for a contact URL in the user agent
    headers = {"User-Agent": "uncertainty-study/0.1 (https://github.com/Hemant-Shashikant-Yadav/LLM-Hallucination; "
                             "thesis research, one-off dataset build)"}
    out: dict[str, str] = {}
    unique = list(dict.fromkeys(t for t in titles if isinstance(t, str) and t))
    for start in range(0, len(unique), batch):
        chunk = unique[start:start + batch]
        params = {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "redirects": 1,
                  "format": "json", "formatversion": 2, "exlimit": batch, "titles": "|".join(chunk)}
        for attempt in range(3):
            try:
                r = httpx.get(url, params=params, headers=headers, timeout=60)
                r.raise_for_status()
                q = r.json().get("query", {})
                break
            except Exception as exc:
                logger.warning("Wikipedia batch {} failed ({}), attempt {}", start // batch, exc, attempt + 1)
                time.sleep(2.0 * (attempt + 1))
        else:
            continue
        # Map each requested title through normalisation and redirects to the page it landed on
        follow = {n["from"]: n["to"] for n in q.get("normalized", [])}
        follow.update({rd["from"]: rd["to"] for rd in q.get("redirects", [])})
        pages = {p["title"]: (p.get("extract") or "").strip() for p in q.get("pages", [])}
        for t in chunk:
            target = t
            for _ in range(3):
                target = follow.get(target, target)
            if pages.get(target):
                out[t] = pages[target][:800]
        logger.info("PopQA Wikipedia intros: {}/{}", min(start + batch, len(unique)), len(unique))
        time.sleep(0.2)
    return out


def build_halueval() -> list[Item]:
    import pandas as pd

    src = SOURCES["halueval"]
    df = pd.read_parquet(_hub_file("pminervini/HaluEval", "qa/data-00000-of-00001.parquet", src["revision"]))
    rng = random.Random(f"{settings.split_seed}-halueval")
    idx = sorted(rng.sample(range(len(df)), SIZES["halueval"][1]))
    items = [Item(id=f"halu-{i}", set="halueval", kind="short", question=df.at[i, "question"],
                  answers=[df.at[i, "right_answer"]], evidence=df.at[i, "knowledge"][:1200])
             for i in idx]
    return _tiers(items, "halueval")


def build_truthfulqa() -> list[Item]:
    rows = [json.loads(l) for l in (FROZEN / "truthfulqa.jsonl").read_text(encoding="utf-8").splitlines() if l]
    items = [Item(id=r["id"], set="truthfulqa", kind="short", question=r["prompt"], answers=r["correct_answers"],
                  incorrect_answers=r["incorrect_answers"], judge_mode="truthfulqa",
                  evidence=f"A reliable source states: {r['correct_answers'][0]}",
                  meta={"category": r.get("category", "")}) for r in rows]
    return _tiers(items[: SIZES["truthfulqa"][1]], "truthfulqa", strata=lambda it: it.meta["category"])


def build_gsm8k() -> list[Item]:
    from datasets import load_dataset

    from app.answer_utils import normalize_number

    ds = load_dataset("openai/gsm8k", "main", split="test", revision=SOURCES["gsm8k"]["revision"])
    rng = random.Random(f"{settings.split_seed}-gsm8k")
    idx = sorted(rng.sample(range(len(ds)), SIZES["gsm8k"][1]))
    items = []
    for i in idx:
        rationale, final = ds[i]["answer"].split("####")
        steps = [re.sub(r"<<[^>]*>>", "", s) for s in rationale.strip().splitlines()]
        items.append(Item(id=f"gsm_{i}", set="gsm8k", kind="number", question=ds[i]["question"],
                          answers=[normalize_number(final.strip()) or final.strip()], hint=_first_half(steps)))
    return _tiers(items, "gsm8k")


def build_math() -> list[Item]:
    rows = [json.loads(l) for l in (FROZEN / "math_test.jsonl").read_text(encoding="utf-8").splitlines() if l]
    # Diagrams cannot be read by a text model; very long problems crowd the context
    rows = [r for r in rows if "[asy]" not in r["question"] and len(r["question"]) <= 600 and r["key"]]
    rng = random.Random(f"{settings.split_seed}-math")
    picked = _stratified(rows, SIZES["math"][1], lambda r: r["level"], rng)
    items = []
    for r in picked:
        steps = [s for s in _sentences(r["solution"]) if "\\boxed" not in s]
        items.append(Item(id=r["id"], set="math", kind="expr", question=r["question"], answers=[r["key"]],
                          hint=_first_half(steps), meta={"level": r["level"], "subject": r["subject"]}))
    return _tiers(items, "math", strata=lambda it: it.meta["level"])


def build_strategyqa() -> list[Item]:
    import pandas as pd

    src = SOURCES["strategyqa"]
    rev_q, rev_full = [s.strip() for s in src["revision"].split("+")]
    df = pd.read_parquet(_hub_file("ChilleD/StrategyQA", "data/test-00000-of-00001-bae602f3ee37f4ca.parquet", rev_q))
    full = json.loads(_hub_file("voidful/StrategyQA", "strategyqa_train.json", rev_full).read_text(encoding="utf-8"))
    by_qid = {r["qid"]: r for r in full}
    rows = [r for r in df.to_dict("records") if r["qid"] in by_qid]
    rng = random.Random(f"{settings.split_seed}-strategyqa")
    rows = _stratified(rows, SIZES["strategyqa"][1], lambda r: bool(r["answer"]), rng)
    items = []
    for r in rows:
        official = by_qid[r["qid"]]
        steps = [s for s in official["decomposition"] if s]
        hint = "Break the question into steps: " + " ".join(f"({i}) {s}" for i, s in enumerate(steps, 1))
        items.append(Item(id=f"sqa-{r['qid']}", set="strategyqa", kind="yesno", question=r["question"],
                          answers=["yes" if bool(r["answer"]) else "no"],
                          evidence=" ".join(official["facts"]), hint=hint if len(steps) >= 2 else None,
                          meta={"term": r["term"]}))
    return _tiers(items, "strategyqa", strata=lambda it: it.answers[0])


def build_arc() -> list[Item]:
    import pandas as pd

    df = pd.read_parquet(_hub_file("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet",
                                   SOURCES["arc"]["revision"]))
    rows = [r for r in df.to_dict("records") if len(r["choices"]["label"]) == 4]
    rng = random.Random(f"{settings.split_seed}-arc")
    rows = sorted(rng.sample(rows, SIZES["arc"][1]), key=lambda r: r["id"])
    items = []
    for r in rows:
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        question = r["question"].strip() + "\n" + "\n".join(f"({LETTERS[i]}) {t}" for i, t in enumerate(texts))
        items.append(Item(id=f"arc-{r['id']}", set="arc", kind="choice", question=question,
                          answers=[LETTERS[labels.index(r["answerKey"])]]))
    return _tiers(items, "arc", strata=lambda it: it.answers[0])


def build_ambigqa() -> list[Item]:
    import pandas as pd

    df = pd.read_parquet(_hub_file("sewon/ambig_qa", "light/validation-00000-of-00001.parquet",
                                   SOURCES["ambigqa"]["revision"]))
    pool = []
    for r in df.to_dict("records"):
        ann = r["annotations"]
        if "multipleQAs" not in list(ann["type"]):
            continue
        readings, answers = [], []
        for pairs in ann["qaPairs"]:
            for q, a in zip(list(pairs["question"]), list(pairs["answer"])):
                alts = [str(x) for x in list(a)]
                readings.append({"question": str(q), "answers": alts})
                answers += alts
        answers = list(dict.fromkeys(a for a in answers if a))
        if len(readings) >= 2 and answers:
            pool.append(Item(id=f"ambig-{r['id']}", set="ambigqa", kind="short", question=r["question"],
                             answers=answers[:20], gold_type="ambiguous", meta={"readings": readings[:8]}))
    rng = random.Random(f"{settings.split_seed}-ambigqa")
    items = sorted(rng.sample(pool, SIZES["ambigqa"][1]), key=lambda it: it.id)
    return _tiers(items, "ambigqa")


def build_selfaware() -> list[Item]:
    import httpx

    rev = SOURCES["selfaware"]["revision"]
    url = f"https://raw.githubusercontent.com/yinzhangyue/SelfAware/{rev}/data/SelfAware.json"
    data = httpx.get(url, timeout=60, follow_redirects=True).json()["example"]
    unanswerable = [e for e in data if not e["answerable"]]
    # Answerable controls from standalone sources (SQuAD questions assume a passage)
    answerable = [e for e in data if e["answerable"] and e["source"].split("_")[0] in ("triviaqa", "hotpot")]
    rng = random.Random(f"{settings.split_seed}-selfaware")
    n = SIZES["selfaware"][1]
    picked = rng.sample(unanswerable, n * 2 // 3) + rng.sample(answerable, n - n * 2 // 3)
    items = []
    for e in sorted(picked, key=lambda e: e["question_id"]):
        if e["answerable"]:
            items.append(Item(id=f"sa-{e['question_id']}", set="selfaware", kind="short", question=e["question"],
                              answers=[str(a) for a in e["answer"]], meta={"answerable": True}))
        else:
            items.append(Item(id=f"sa-{e['question_id']}", set="selfaware", kind="short", question=e["question"],
                              answers=[], judge_mode="abstain", gold_type="unanswerable",
                              meta={"answerable": False}))
    return _tiers(items, "selfaware", strata=lambda it: it.meta["answerable"])


def build_freshqa() -> list[Item]:
    import pandas as pd

    df = pd.read_parquet(_hub_file("bojanbabic/freshqa_09302025", "data/test-00000-of-00001.parquet",
                                   SOURCES["freshqa"]["revision"]))
    pool = []
    for r in df.to_dict("records"):
        answers = [str(r[f"answer_{i}"]) for i in range(10)
                   if r.get(f"answer_{i}") is not None and str(r[f"answer_{i}"]) not in ("", "nan", "None")]
        false_premise = str(r["false_premise"]).lower() == "true"
        changing = r["fact_type"] in ("fast-changing", "slow-changing")
        pool.append(Item(
            id=f"fresh-{r['id']}", set="freshqa", kind="short", question=r["question"], answers=answers,
            judge_mode="premise" if false_premise else "match",
            gold_type="false_premise" if false_premise else ("temporal" if changing else None),
            meta={"fact_type": r["fact_type"], "false_premise": false_premise,
                  "effective_year": r["effective_year"], "num_hops": r["num_hops"], "snapshot": "2025-09-30"},
        ))
    rng = random.Random(f"{settings.split_seed}-freshqa")
    strata = lambda it: (it.meta["false_premise"], it.meta["fact_type"])
    items = sorted(_stratified(pool, SIZES["freshqa"][1], strata, rng), key=lambda it: int(it.id.split("-")[1]))
    return _tiers(items, "freshqa", strata=strata)


BUILDERS = {
    "triviaqa": build_triviaqa, "popqa": build_popqa, "halueval": build_halueval,
    "truthfulqa": build_truthfulqa, "gsm8k": build_gsm8k, "math": build_math,
    "strategyqa": build_strategyqa, "arc": build_arc, "ambigqa": build_ambigqa,
    "selfaware": build_selfaware, "freshqa": build_freshqa,
}


# ============================================================================ manifest


def build(only: list[str] | None = None, sets_dir: Path | None = None) -> dict:
    sets_dir = sets_dir or settings.sets_dir
    manifest_path = sets_dir / "MANIFEST.json"
    built: dict = {}
    for name in only or list(SETS):
        logger.info("Building {}", name)
        items = BUILDERS[name]()
        assert len({it.id for it in items}) == len(items), f"{name}: duplicate ids"
        path = write_set(name, items, sets_dir)
        built[name] = {
            **SOURCES[name],
            "domain": SETS[name][0],
            "kind": SETS[name][1],
            "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "extended": len(items),
            "core": sum(it.core for it in items),
            "dev": sum(it.split == "dev" for it in items),
            "test": sum(it.split == "test" for it in items),
            "with_evidence": sum(bool(it.evidence) for it in items),
            "with_hint": sum(bool(it.hint) for it in items),
            "gold_types": {t: sum(it.gold_type == t for it in items)
                           for t in sorted({it.gold_type for it in items if it.gold_type})},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "built": date.today().isoformat(),
        }
        logger.info("{}: {} items ({} core) → {}", name, len(items), built[name]["core"], path)
    # Read the manifest only now, so builds of different sets never drop each other's entries
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(built)
    manifest = {k: manifest[k] for k in SETS if k in manifest}
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest
