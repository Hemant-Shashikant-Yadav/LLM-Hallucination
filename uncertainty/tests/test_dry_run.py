"""
End to end with the simulated model: every stage of `run-all`, both reports, resuming, and
whether the Objective 2 pipeline recovers the types the fake model plants.

One small run (a few questions per set, three models) is shared by every test here.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from uncertainty.generate.fake import PLANTED_TO_TYPE, FakeBackend
from uncertainty.orchestrate.run_all import RunConfig, run_all
from uncertainty.roster import spec
from uncertainty.settings import settings
from uncertainty.store import read_json

MODELS = ["qwen2.5:3b", "llama3.1:8b", "gemma3:4b", "phi4-mini"]
_RUN: dict = {}


def shared_run() -> Path:
    if "dir" not in _RUN:
        settings.bootstrap = 200  # confidence intervals are not what is tested here
        tmp = Path(tempfile.mkdtemp(prefix="uq_dry_"))
        run_all(RunConfig(backend="fake", results_dir=tmp, models=MODELS, limit=24, skip_tests=True,
                          second_judge_n=40))
        _RUN["dir"] = tmp
    return _RUN["dir"]


def tearDownModule():
    if "dir" in _RUN:
        shutil.rmtree(_RUN["dir"], ignore_errors=True)


class DryRunTest(unittest.TestCase):
    def test_every_stage_and_report(self):
        d = shared_run()
        state = read_json(d / "run_state.json")
        for stage in ("check", "pilot", "focus", "first_reports", "others", "final_reports"):
            self.assertIn(stage, state["stages"])
        self.assertEqual({m: s["status"] for m, s in state["models"].items()}, dict.fromkeys(MODELS, "done"))
        for f in ("reports/SUMMARY.md", "reports/obj1/obj1.md", "reports/obj1/method_table.csv",
                  "reports/obj1/figures/fig_auroc_heatmap.png", "reports/obj2/obj2.md",
                  "models/detector_full.joblib", "models/classifier_full.joblib"):
            self.assertTrue((d / f).exists(), f)

    def test_semantic_entropy_beats_chance(self):
        t = pd.read_csv(shared_run() / "reports" / "obj1" / "method_table.csv")
        se = t[(t["set"] == "ALL") & (t["method"] == "se_discrete")]
        self.assertTrue((se["auroc"] > 0.6).all(), se[["model", "auroc"]])

    def test_resume_does_no_new_work(self):
        d = shared_run()
        gen = next((d / spec("qwen2.5:3b").slug / "gen").glob("*.jsonl.gz"))
        before = gen.stat().st_mtime
        run_all(RunConfig(backend="fake", results_dir=d, models=MODELS, limit=24, skip_tests=True,
                          until="focus"))
        self.assertEqual(gen.stat().st_mtime, before)

    def test_labels_follow_planted_types(self):
        """The labelling (metadata + interventions) should name the type the fake model planted."""
        d = shared_run()
        from uncertainty.sets.load import load_all

        items = {it.id: it for its in load_all("extended", limit=24).values() for it in its}
        labels = pd.read_csv(d / "reports" / "obj2" / "type_labels.csv").dropna(subset=["type"])
        planted = [PLANTED_TO_TYPE.get(FakeBackend.planted(r.model, items[r.id])) for r in labels.itertuples()]
        known = [(p, t) for p, t in zip(planted, labels["type"]) if p is not None]
        agreement = sum(p == t for p, t in known) / len(known)
        self.assertGreater(agreement, 0.8)

    def test_classifier_beats_pipeline_rule(self):
        s = read_json(shared_run() / "reports" / "obj2" / "summary.json")
        c = s["classifier"]["full"]["classifiers"]
        best = s["classifier"]["full"]["best"]
        self.assertGreater(c[best]["macro_f1"], c["baseline"]["macro_f1"])

    def test_analyze_tool(self):
        import asyncio

        from uncertainty.framework.analyze import analyze
        from uncertainty.generate.backends import make_backend
        from uncertainty.sets.load import load_set

        item = load_set("gsm8k")[0]
        backend = make_backend("fake", [item])
        for mode in ("full", "lite"):
            report = asyncio.run(analyze(item.question, "qwen2.5:3b", mode, backend=backend,
                                         results_dir=shared_run()))
            self.assertIn(report["classify"]["type"], ("knowledge", "reasoning", "aleatoric", "temporal"))
            self.assertIn(report["identify"]["level"], ("low", "medium", "high"))
            self.assertTrue(report["recommended_action"])


if __name__ == "__main__":
    unittest.main()
