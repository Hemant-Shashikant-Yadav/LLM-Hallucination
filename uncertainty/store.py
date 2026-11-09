"""
Where results live and how they are read and written.

results/<model>/gen/<set>.jsonl.gz      one record per question (greedy, samples, self-evaluation)
results/<model>/nli/<set>.npz           NLI matrices for free-text questions
results/<model>/labels.jsonl            correctness of every greedy answer
results/<model>/scores.csv              every Objective 1 score per question
results/<model>/interventions.jsonl     Objective 2 re-asks (focus models)
results/reports/                        Objective 1 and 2 reports, SUMMARY.md

A set being generated is written to <set>.partial.jsonl one line at a time (so a crash
loses at most the line in progress) and gzipped when complete.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

from uncertainty.roster import spec


def model_dir(results_dir: Path, model: str) -> Path:
    return results_dir / spec(model).slug


def gen_paths(results_dir: Path, model: str, set_name: str) -> tuple[Path, Path]:
    d = model_dir(results_dir, model) / "gen"
    return d / f"{set_name}.partial.jsonl", d / f"{set_name}.jsonl.gz"


def read_jsonl(path: Path) -> list[dict]:
    """Records of a .jsonl or .jsonl.gz file; a cut-off last line (a crash mid-write) is skipped."""
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    out = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def write_jsonl(path: Path, records: list[dict], compress: bool | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if (path.suffix == ".gz" if compress is None else compress) else open
    with opener(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def repair_partial(path: Path) -> None:
    """Drop a cut-off last line before appending again."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    good = []
    for line in lines:
        try:
            json.loads(line)
            good.append(line if line.endswith("\n") else line + "\n")
        except json.JSONDecodeError:
            pass
    path.write_text("".join(good), encoding="utf-8")


def finalize(partial: Path, final: Path, order: list[str] | None = None) -> None:
    """Gzip a finished set (records in question order) and remove the partial file."""
    records = read_jsonl(partial) + read_jsonl(final)
    by_id = {r["id"]: r for r in records}
    ids = [i for i in order if i in by_id] if order else list(by_id)
    tmp = final.with_name(final.name + ".tmp")
    write_jsonl(tmp, [by_id[i] for i in ids], compress=True)
    shutil.move(str(tmp), str(final))
    partial.unlink(missing_ok=True)


def load_generations(results_dir: Path, model: str, set_name: str) -> list[dict]:
    partial, final = gen_paths(results_dir, model, set_name)
    by_id = {r["id"]: r for r in read_jsonl(final)}
    by_id.update({r["id"]: r for r in read_jsonl(partial)})
    return list(by_id.values())


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    shutil.move(str(tmp), str(path))
