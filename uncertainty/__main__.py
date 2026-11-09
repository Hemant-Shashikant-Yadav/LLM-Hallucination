"""
Command line for the uncertainty study.

  python -m uncertainty run-all                      the whole study, end to end (the Mac command)
  python -m uncertainty run-all --backend fake       dry run with the simulated model (no Ollama)
  python -m uncertainty build-sets [--only SET ...]  freeze the question sets
  python -m uncertainty report [obj1|obj2|all]       rebuild reports from stored results
  python -m uncertainty train                        retrain the Objective 2 framework (= report obj2)
  python -m uncertainty annotate export|import|export-audit|import-audit
  python -m uncertainty analyze "question" --model qwen2.5:3b [--mode lite|full]
  python -m uncertainty generate|nli|label|score|intervene --model M    one step for one model
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from uncertainty.settings import settings


def _results_dir(args) -> Path:
    if getattr(args, "results_dir", None):
        return Path(args.results_dir)
    return settings.fake_results_dir if getattr(args, "backend", "ollama") == "fake" else settings.results_dir


def _items(limit: int | None = None):
    from uncertainty.sets.load import load_all

    return {it.id: it for its in load_all("extended", limit=limit).values() for it in its}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m uncertainty", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, backend=True):
        sp.add_argument("--results-dir")
        if backend:
            sp.add_argument("--backend", choices=["ollama", "fake"], default="ollama")

    r = sub.add_parser("run-all", help="the whole study, end to end")
    common(r)
    r.add_argument("--models", nargs="+", help="override the roster picked from memory")
    r.add_argument("--pipeline-model", help="the pipeline's model on this machine, if not qwen2.5:3b")
    r.add_argument("--tier-focus", default="extended", choices=["core", "extended"])
    r.add_argument("--tier-others", default="core", choices=["core", "extended"])
    r.add_argument("--max-hours", type=float)
    r.add_argument("--until")
    r.add_argument("--only")
    r.add_argument("--limit", type=int, help="first n questions per set (dry runs)")
    r.add_argument("--memory-gb", type=float)
    r.add_argument("--real-nli", action="store_true", help="fake backend: use the real DeBERTa NLI model")
    r.add_argument("--skip-tests", action="store_true")
    r.add_argument("--second-judge-n", type=int, default=300)

    c = sub.add_parser("check", help="run-all's first stage only")
    common(c)

    b = sub.add_parser("build-sets", help="freeze the question sets")
    b.add_argument("--only", nargs="+")

    for name in ("generate", "nli", "label", "score", "intervene"):
        s = sub.add_parser(name, help=f"{name} for one model")
        common(s)
        s.add_argument("--model", required=True)
        s.add_argument("--tier", default="core", choices=["core", "extended"])
        s.add_argument("--limit", type=int)
        s.add_argument("--judge")

    rep = sub.add_parser("report", help="rebuild reports from stored results")
    common(rep)
    rep.add_argument("which", nargs="?", default="all", choices=["obj1", "obj2", "all"])

    t = sub.add_parser("train", help="train and evaluate the Objective 2 framework")
    common(t)

    a = sub.add_parser("annotate", help="human labels for Objective 2 and the judge audit")
    common(a)
    a.add_argument("action", choices=["export", "import", "export-audit", "import-audit"])
    a.add_argument("--out")
    a.add_argument("--files", nargs="+")
    a.add_argument("--n", type=int)

    z = sub.add_parser("analyze", help="uncertainty report for one question")
    common(z)
    z.add_argument("question")
    z.add_argument("--model", default="qwen2.5:3b")
    z.add_argument("--mode", default="full", choices=["full", "lite"])

    args = p.parse_args(argv)
    rdir = _results_dir(args) if args.cmd != "build-sets" else None

    if args.cmd in ("run-all", "check"):
        from uncertainty.orchestrate.run_all import RunConfig, run_all

        cfg = RunConfig(backend=args.backend, results_dir=rdir)
        if args.cmd == "check":
            cfg.only = "check"
        else:
            cfg.models, cfg.pipeline_model = args.models, args.pipeline_model
            cfg.tier_focus, cfg.tier_others, cfg.max_hours = args.tier_focus, args.tier_others, args.max_hours
            cfg.until, cfg.only, cfg.limit, cfg.memory_gb = args.until, args.only, args.limit, args.memory_gb
            cfg.real_nli, cfg.skip_tests, cfg.second_judge_n = args.real_nli, args.skip_tests, args.second_judge_n
        run_all(cfg)
        print(f"Summary: {rdir / 'reports' / 'SUMMARY.md'}")
        return 0

    if args.cmd == "build-sets":
        from uncertainty.sets.build import build

        manifest = build(only=args.only)
        print(json.dumps({k: {x: v[x] for x in ("extended", "core", "dev", "test")} for k, v in manifest.items()},
                         indent=1))
        return 0

    if args.cmd in ("generate", "nli", "label", "score", "intervene"):
        return asyncio.run(_one_step(args, rdir))

    if args.cmd == "report" or args.cmd == "train":
        from uncertainty.report import obj1, obj2

        which = "obj2" if args.cmd == "train" else args.which
        if which in ("obj1", "all"):
            obj1.build(rdir)
        if which in ("obj2", "all"):
            obj2.build(rdir, _items())
        print(f"Reports in {rdir / 'reports'}")
        return 0

    if args.cmd == "annotate":
        from uncertainty.label import annotate

        out_dir = rdir / "annotation"
        if args.action == "export":
            path = annotate.export(rdir, _items(), Path(args.out or out_dir / "labels_to_fill.csv"), n=args.n or 300)
        elif args.action == "export-audit":
            path = annotate.export_audit(rdir, _items(), Path(args.out or out_dir / "judge_audit_to_fill.csv"),
                                         n=args.n or 100)
        elif args.action == "import":
            path = annotate.import_labels(rdir, [Path(f) for f in args.files])
        else:
            print(json.dumps(annotate.import_audit(rdir, Path(args.files[0])), indent=1))
            return 0
        print(f"Written: {path}")
        return 0

    if args.cmd == "analyze":
        from uncertainty.framework.analyze import analyze
        from uncertainty.generate.backends import make_backend

        backend = make_backend(args.backend, list(_items().values()))
        state = json.loads((rdir / "run_state.json").read_text(encoding="utf-8")) if (rdir / "run_state.json").exists() else {}
        mode = (state.get("logprobs") or {}).get("mode", "raw")
        report = asyncio.run(analyze(args.question, args.model, args.mode, backend=backend, results_dir=rdir,
                                     logprob_mode=mode))
        print(json.dumps(report, indent=1, ensure_ascii=False))
        return 0
    return 1


async def _one_step(args, rdir: Path) -> int:
    from uncertainty import thresholds
    from uncertainty.generate.backends import make_backend
    from uncertainty.generate.runner import generate_set
    from uncertainty.label.correctness import label_model
    from uncertainty.label.interventions import candidates, generate_interventions, resolve_interventions
    from uncertainty.methods.nli import FakeNLI, NLIScorer, build_set_matrices, nli_path
    from uncertainty.roster import judge_for, spec
    from uncertainty.score import score_model
    from uncertainty.sets.load import load_all

    sets = {s: [it for it in its if args.tier == "extended" or it.core]
            for s, its in load_all("extended", limit=args.limit).items()}
    items = {it.id: it for its in sets.values() for it in its}
    backend = make_backend(args.backend, list(items.values()))
    nli = FakeNLI() if args.backend == "fake" else NLIScorer()
    judge = args.judge or judge_for(args.model, 16.0)
    from uncertainty.store import load_generations

    if args.cmd == "generate":
        for its in sets.values():
            print(await generate_set(backend, args.model, its, rdir, k=settings.k_samples))
    elif args.cmd == "nli":
        for s, its in sets.items():
            if its and its[0].kind == "short":
                build_set_matrices(nli, load_generations(rdir, args.model, s), nli_path(rdir, spec(args.model).slug, s))
    elif args.cmd == "label":
        print(await label_model(backend, args.model, sets, rdir, judge))
    elif args.cmd == "score":
        thresholds.tune_model(rdir, args.model, score_model(rdir, args.model))
    elif args.cmd == "intervene":
        import pandas as pd

        scores = pd.read_csv(rdir / spec(args.model).slug / "scores.csv")
        todo = candidates(scores, items, thresholds.load(rdir, args.model)["low"])
        await generate_interventions(backend, args.model, todo, rdir)
        await resolve_interventions(backend, args.model, items, rdir, judge, nli)
    return 0


if __name__ == "__main__":
    sys.exit(main())
