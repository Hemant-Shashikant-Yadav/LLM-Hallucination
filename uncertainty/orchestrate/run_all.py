"""
The one command (PLAN §1.1): `python -m uncertainty run-all`.

Stages, each recorded as done in results/run_state.json so a re-run skips finished work:
  1. check          memory → roster; pull models; Ollama version; log-prob check; unit tests
  2. pilot          5 dev questions per set on the pipeline model through every later stage
                    (gates stop the run here if the code or prompts are wrong); speed → hours
  3. focus          focus models: generate (extended) → NLI → labels → scores → interventions
  4. first_reports  Objective 1 and 2 reports from the focus models
  5. others         every other model (core tier, smallest first): generate → NLI → labels → scores
  6. final_reports  both reports with every model, and SUMMARY.md

Reports are rebuilt after every model, so stopping early still leaves complete reports for
the models finished so far. A per-model gate failure skips that model (recorded in the
summary) rather than stopping a week-long run; a pilot failure stops it, since then the
code or prompts are wrong for every model.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger

from uncertainty import thresholds
from uncertainty.generate.backends import make_backend
from uncertainty.generate.runner import generate_set
from uncertainty.label.correctness import label_model
from uncertainty.label.interventions import generate_interventions, resolve_interventions
from uncertainty.methods.nli import FakeNLI, NLIScorer, build_set_matrices, nli_path
from uncertainty.orchestrate import gates, system
from uncertainty.report import obj1, obj2
from uncertainty.roster import focus_models, judge_for, judges_needed, pick_roster, second_judge_for, spec
from uncertainty.score import score_model
from uncertainty.sets.load import STRUCTURED, Item, load_all
from uncertainty.settings import PROJECT_ROOT, settings
from uncertainty.store import load_generations, read_json, write_json

STAGES = ["check", "pilot", "focus", "first_reports", "others", "final_reports"]
PILOT_PER_SET = 5
TRIMMED_K = 6


@dataclass
class RunConfig:
    backend: str = "ollama"
    results_dir: Path | None = None
    models: list[str] | None = None
    pipeline_model: str | None = None
    tier_focus: str = "extended"
    tier_others: str = "core"
    max_hours: float | None = None
    until: str | None = None
    only: str | None = None
    limit: int | None = None          # first n questions per set (dry runs)
    memory_gb: float | None = None
    real_nli: bool = False
    skip_tests: bool = False
    second_judge_n: int = 300
    extra: dict = field(default_factory=dict)


class Run:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        default_dir = settings.fake_results_dir if cfg.backend == "fake" else settings.results_dir
        self.dir = Path(cfg.results_dir or default_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "run_state.json"
        self.state: dict = read_json(self.state_path, None) or {"stages": {}, "models": {}, "degradations": [],
                                                                 "created": datetime.now().isoformat()}
        self.extended = load_all("extended", limit=cfg.limit)
        self.items = {it.id: it for its in self.extended.values() for it in its}
        self.backend = make_backend(cfg.backend, list(self.items.values()))
        self.nli = FakeNLI() if cfg.backend == "fake" and not cfg.real_nli else NLIScorer()
        self._log_sink = logger.add(self.dir / "run.log", level="INFO", encoding="utf-8")

    # ------------------------------------------------------------------ bookkeeping

    def save(self) -> None:
        self.state["updated"] = datetime.now().isoformat()
        write_json(self.state_path, self.state)

    def degrade(self, where: str, what: str) -> None:
        logger.warning("Degradation | {} | {}", where, what)
        self.state["degradations"].append({"time": datetime.now().isoformat(), "where": where, "what": what})
        self.save()

    def model_state(self, model: str) -> dict:
        return self.state["models"].setdefault(model, {"generated": {}, "status": "pending"})

    @property
    def memory(self) -> float:
        return self.state.get("memory_gb", 16.0)

    def ours(self) -> set[str]:
        return set(self.state.get("roster", [])) | set(judges_needed(self.state.get("roster", []), self.memory))

    def items_at(self, tier: str) -> dict[str, list[Item]]:
        return {s: [it for it in its if tier == "extended" or it.core] for s, its in self.extended.items()}

    def k_for(self, set_items: list[Item]) -> int:
        trimmed = "k6_worked" in self.state.get("trims", [])
        return TRIMMED_K if trimmed and set_items and set_items[0].kind in STRUCTURED else settings.k_samples

    # ------------------------------------------------------------------ stages

    async def stage_check(self) -> None:
        mem = self.cfg.memory_gb or (24.0 if self.cfg.backend == "fake" else system.memory_gb())
        roster = self.cfg.models or [s.tag for s in pick_roster(mem)]
        if not roster:
            # e.g. Colab: /proc/meminfo reports ~13 GB of system RAM, below every roster model's need,
            # although the GPU has room. Stop with the fix rather than run an empty study.
            raise system.RunStop(f"No roster model fits {mem:.0f} GB of memory. On a GPU machine (Colab), name "
                                 "the models with --models and the judges' memory class with --memory-gb; see "
                                 "uncertainty/colab/README.md.")
        pipeline_model = self.cfg.pipeline_model
        if pipeline_model and pipeline_model not in roster:
            roster.append(pipeline_model)
        focus = focus_models(roster, pipeline_model)
        if not focus:
            focus = roster[:1]
        self.state.update({"memory_gb": mem, "machine": system.machine_info(), "roster": roster, "focus": focus,
                           "backend": self.cfg.backend, "ollama_version": await self.backend.version()})
        logger.info("Memory {:.0f} GB → {} models: {} | focus: {}", mem, len(roster), roster, focus)
        await system.ensure_only(self.backend, None, self.ours())
        failed = await system.ensure_pulled(self.backend, roster + judges_needed(roster, mem))
        for m in failed:
            self.degrade("check", f"could not pull {m}; it is skipped")
            if m in roster:
                roster.remove(m)
        if focus[0] in failed:
            raise system.RunStop(f"The pipeline model {focus[0]} could not be pulled; pull it by hand and re-run.")
        self.state["focus"] = [m for m in focus if m in roster]
        lp = await system.logprob_check(self.backend, self.state["focus"][0])
        self.state["logprobs"] = lp
        if lp["mode"] != "raw":
            self.degrade("check", f"log-prob mode '{lp['mode']}': {lp}. Token-level methods "
                                  + ("are off." if lp["mode"] == "none" else "use T = 1, top_k = 1 greedy answers."))
        if not self.cfg.skip_tests:
            # In a separate process: the tests start their own event loops
            logger.info("Running the unit tests")
            done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "uncertainty/tests", "-t", "."],
                                  cwd=PROJECT_ROOT, capture_output=True, text=True)
            (self.dir / "unit_tests.log").write_text(done.stdout + done.stderr, encoding="utf-8")
            if done.returncode != 0:
                raise system.RunStop(f"Unit tests failed on this machine (see {self.dir / 'unit_tests.log'}). "
                                     "Nothing was generated.")

    async def stage_pilot(self) -> None:
        model = self.state["focus"][0]
        pilot_dir = self.dir / "_pilot"
        items = {s: [it for it in its if it.split == "dev"][:PILOT_PER_SET] for s, its in self.extended.items()}
        await system.ensure_only(self.backend, model, self.ours())
        total_tokens, total_seconds = 0, 0.0
        per_set = {}
        for set_name, its in items.items():
            if not its:
                continue
            st = await generate_set(self.backend, model, its, pilot_dir, k=settings.k_samples,
                                    logprob_mode=self.state["logprobs"]["mode"])
            recs = load_generations(pilot_dir, model, set_name)
            gates.check_generation(recs, f"pilot {model} {set_name}")
            total_tokens += st.get("completion_tokens", 0)
            total_seconds += st.get("seconds", 0.0)
            per_set[set_name] = sum(r["cost"]["completion_tokens"] for r in recs) / max(len(recs), 1)
        await self._nli(pilot_dir, model, items)
        stats = await label_model(self.backend, model, items, pilot_dir, judge_for(model, self.memory))
        if warning := gates.check_judge(stats, f"pilot judge for {model}"):
            self.degrade("pilot", warning)
        scores = score_model(pilot_dir, model)
        thresholds.tune_model(pilot_dir, model, scores)
        pilot_items = {it.id: it for its in items.values() for it in its}
        from uncertainty.label.interventions import candidates

        todo = candidates(scores, pilot_items, thresholds.load(pilot_dir, model)["low"])
        await generate_interventions(self.backend, model, todo[:6], pilot_dir)
        await resolve_interventions(self.backend, model, pilot_items, pilot_dir, judge_for(model, self.memory), self.nli)
        obj1.build(pilot_dir, split="all")
        obj2.build(pilot_dir, pilot_items, n_boot=50)
        speed = total_tokens / total_seconds if total_seconds else None
        self.state["pilot"] = {"model": model, "tokens_per_s": speed, "tokens_per_question": per_set}
        self._project()
        self._trim()

    def _hours(self) -> dict[str, float]:
        p = self.state.get("pilot") or {}
        speed, per_q = p.get("tokens_per_s"), p.get("tokens_per_question") or {}
        if not speed:
            return {}
        base = spec(p["model"]).size_b or 3
        out = {}
        for m in self.state["roster"]:
            tier = self.cfg.tier_focus if m in self.state["focus"] and "third_focus_core" not in self._trim_flags(m) \
                else self.cfg.tier_others
            tokens = 0.0
            for s, its in self.items_at(tier).items():
                k_ratio = (self.k_for(its) + 1) / (settings.k_samples + 1)
                tokens += per_q.get(s, 0) * len(its) * k_ratio
            if m in self.state["focus"]:
                tokens *= 1.3  # interventions
            out[m] = tokens / (speed * ((base / max(spec(m).size_b, 1)) ** 0.85)) / 3600
        return out

    def _trim_flags(self, model: str) -> set[str]:
        flags = set()
        if "third_focus_core" in self.state.get("trims", []) and len(self.state["focus"]) > 2 \
                and model == self.state["focus"][2]:
            flags.add("third_focus_core")
        return flags

    def _project(self) -> None:
        hours = self._hours()
        self.state["projection_hours"] = {m: round(h, 1) for m, h in hours.items()}
        self.state["projection_total_hours"] = round(sum(hours.values()), 1)
        logger.info("Projected run time: {:.1f} h ({})", sum(hours.values()), self.state["projection_hours"])
        self.save()

    def _trim(self) -> None:
        limit = self.cfg.max_hours
        if not limit:
            return
        trims = self.state.setdefault("trims", [])
        levers = ["k6_worked", "third_focus_core"] + [
            f"drop:{s.tag}" for s in sorted((spec(m) for m in self.state["roster"]), key=lambda s: -s.size_b)
            if s.optional and s.tag not in self.state["focus"]]
        for lever in levers:
            if self.state["projection_total_hours"] <= limit:
                break
            trims.append(lever)
            if lever.startswith("drop:"):
                self.state["roster"].remove(lever[5:])
            self._project()
            self.degrade("time limit", f"trimmed '{lever}' to fit --max-hours {limit}; "
                                       f"projection now {self.state['projection_total_hours']} h")

    async def _nli(self, results_dir: Path, model: str, items: dict[str, list[Item]]) -> None:
        for set_name, its in items.items():
            if its and its[0].kind == "short":
                build_set_matrices(self.nli, load_generations(results_dir, model, set_name),
                                   nli_path(results_dir, spec(model).slug, set_name))

    async def process_model(self, model: str, tier: str, interventions: bool) -> None:
        ms = self.model_state(model)
        if ms["status"] == "done":
            return  # a model skipped by a gate is tried again on the next run (after a fix)
        start = time.perf_counter()
        items = self.items_at(tier)
        try:
            await system.ensure_only(self.backend, model, self.ours())
            for set_name, its in items.items():
                if not its or ms["generated"].get(set_name):
                    continue
                await generate_set(self.backend, model, its, self.dir, k=self.k_for(its),
                                   logprob_mode=self.state["logprobs"]["mode"],
                                   check=lambda recs, s=set_name: gates.check_generation(recs, f"{model} {s}"))
                ms["generated"][set_name] = True
                self.save()
            await system.ensure_only(self.backend, None, self.ours())  # free memory for NLI
            await self._nli(self.dir, model, items)
            stats = await label_model(self.backend, model, items, self.dir, judge_for(model, self.memory),
                                      second_judge_for(model, self.memory), self.cfg.second_judge_n)
            if warning := gates.check_judge(stats, f"judge for {model}"):
                self.degrade(model, warning)
            ms["label_stats"] = stats
            scores = score_model(self.dir, model)
            thresholds.tune_model(self.dir, model, scores)
            if interventions:
                from uncertainty.label.interventions import candidates

                todo = candidates(scores, self.items, thresholds.load(self.dir, model)["low"])
                await system.ensure_only(self.backend, model, self.ours())
                await generate_interventions(self.backend, model, todo, self.dir)
                await resolve_interventions(self.backend, model, self.items, self.dir,
                                            judge_for(model, self.memory), self.nli)
            ms["status"] = "done"
        except gates.GateFailure as exc:
            ms["status"] = "skipped"
            self.degrade(model, f"gate failed, model skipped: {exc}")
        ms["hours"] = round(ms.get("hours", 0) + (time.perf_counter() - start) / 3600, 2)
        self.save()
        self.reports()

    def reports(self) -> None:
        try:
            obj1.build(self.dir)
            obj2.build(self.dir, self.items)
        except Exception as exc:  # a report bug must not lose a night of generation
            self.degrade("reports", f"{type(exc).__name__}: {exc}")
            logger.exception(exc)
        write_summary(self)

    async def stage_focus(self) -> None:
        for m in self.state["focus"]:
            tier = self.cfg.tier_others if self._trim_flags(m) else self.cfg.tier_focus
            await self.process_model(m, tier, interventions=True)

    async def stage_others(self) -> None:
        others = [m for m in self.state["roster"] if m not in self.state["focus"]]
        for m in sorted(others, key=lambda t: (spec(t).size_b, t)):
            await self.process_model(m, self.cfg.tier_others, interventions=False)

    async def run(self) -> None:
        stages = STAGES
        if self.cfg.only:
            stages = [self.cfg.only]
        elif self.cfg.until:
            stages = STAGES[: STAGES.index(self.cfg.until) + 1]
        if "check" not in stages and "roster" not in self.state:
            stages = ["check", *stages]
        try:
            for stage in stages:
                if self.state["stages"].get(stage) and stage not in ("first_reports", "final_reports") \
                        and not self.cfg.only:
                    logger.info("Stage {} already done", stage)
                    continue
                logger.info("=== Stage {} ===", stage)
                if stage in ("first_reports", "final_reports"):
                    self.reports()
                else:
                    await getattr(self, f"stage_{stage}")()
                self.state["stages"][stage] = datetime.now().isoformat()
                self.save()
            write_summary(self)
        finally:
            logger.remove(self._log_sink)


def write_summary(run: Run) -> Path:
    s = run.state
    lines = [f"# Uncertainty study: run summary\n", f"Updated {datetime.now():%Y-%m-%d %H:%M}. "
             f"Backend: {s.get('backend')} (Ollama {s.get('ollama_version')}). "
             f"Memory: {s.get('memory_gb')} GB. Results: `{run.dir}`.\n",
             "## Stages\n", *[f"- {st}: {'done ' + s['stages'][st][:16] if st in s['stages'] else 'pending'}"
                              for st in STAGES],
             f"\n## Models\n\nFocus (extended tier + interventions): {', '.join(s.get('focus', []))}.\n",
             "| Model | Status | Hours | Projected h |", "|---|---|---|---|"]
    for m in s.get("roster", []):
        ms = s["models"].get(m, {})
        lines.append(f"| {m} | {ms.get('status', 'pending')} | {ms.get('hours', '')} | "
                     f"{s.get('projection_hours', {}).get(m, '')} |")
    lp = s.get("logprobs", {})
    lines += [f"\nLog-prob mode: **{lp.get('mode', '?')}** ({lp}).",
              f"Trims for the time limit: {s.get('trims') or 'none'}.\n"]
    o1 = read_json(run.dir / "reports" / "obj1" / "summary.json", {}) or {}
    if o1.get("headline"):
        lines += ["## Objective 1 (test split)\n", "| Model | Accuracy | AUROC of semantic entropy | Best method (AUROC) | Blind spot |",
                  "|---|---|---|---|---|"]
        for h in o1["headline"]:
            f = lambda v: "–" if v is None else f"{v:.3f}"  # noqa: E731
            lines.append(f"| {h['model']} | {f(h['accuracy'])} | {f(h['auroc_se'])} | {h['best_method']} "
                         f"({f(h['best_auroc'])}) | {f(h['blind_spot'])} |")
    o2 = read_json(run.dir / "reports" / "obj2" / "summary.json", {}) or {}
    f = lambda v: "–" if v is None else f"{v:.3f}"
    det = (o2.get("detector") or {}).get("full") or {}
    cls = (o2.get("classifier") or {}).get("full") or {}
    if det or cls:
        lines.append("\n## Objective 2\n")
    if det and not det.get("skipped"):
        lines.append(f"Detector AUROC (full): {f(det.get('auroc'))}; semantic entropy alone: {f(det.get('se_auroc'))}.")
    if cls.get("classifiers"):
        b = cls["best_vs_baseline"]
        lines.append(f"Type classifier ({cls['best']}) macro-F1 {f(cls['classifiers'][cls['best']]['macro_f1'])} vs "
                     f"the pipeline rule {f(cls['classifiers']['baseline']['macro_f1'])} (difference CI "
                     f"{f(b['diff_lo'])} to {f(b['diff_hi'])}).\n")
    lines += ["\n## Degradations\n"] + ([f"- {d['time'][:16]} · {d['where']}: {d['what']}" for d in s["degradations"]]
                                       or ["- none"])
    lines += ["\nReports: `reports/obj1/obj1.md`, `reports/obj2/obj2.md`. Human labels: "
              "`python -m uncertainty annotate export` (see the README).\n"]
    path = run.dir / "reports" / "SUMMARY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_all(cfg: RunConfig) -> Path:
    run = Run(cfg)
    try:
        asyncio.run(run.run())
    except system.RunStop as exc:
        logger.error("Run stopped: {}", exc)
        run.degrade("stopped", str(exc))
        write_summary(run)
        raise
    except gates.GateFailure as exc:
        logger.error("Run stopped by a gate: {}", exc)
        run.degrade("stopped", str(exc))
        write_summary(run)
        raise
    return run.dir
