"""
The machine and the Ollama server: memory (to pick the roster), pulling models, keeping
only one model loaded, and checking what the server's log-probs mean.
"""

from __future__ import annotations

import ctypes
import platform
import subprocess
import sys

from loguru import logger

from uncertainty.generate.backends import Backend, normalize_tag


class RunStop(RuntimeError):
    """A problem that must stop the run, with a message saying what to do."""


def memory_gb() -> float:
    """Total memory: unified memory on a Mac (what Ollama can use most of)."""
    try:
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 2**30
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as fh:
                return int(fh.readline().split()[1]) / 2**20
        if sys.platform == "win32":
            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 2**30
    except Exception as exc:
        logger.warning("Could not read memory size: {}", exc)
    return 16.0


def machine_info() -> dict:
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "memory_gb": round(memory_gb(), 1)}


async def ensure_pulled(backend: Backend, models: list[str]) -> list[str]:
    """Pull every model the run needs; returns the models that could not be pulled."""
    available = set(await backend.available())
    if "*" in available:
        return []
    failed = []
    for m in models:
        if normalize_tag(m) in available:
            continue
        try:
            await backend.pull(m)
        except Exception as exc:
            logger.error("Could not pull {}: {}", m, exc)
            failed.append(m)
    return failed


async def ensure_only(backend: Backend, model: str | None, ours: set[str]) -> None:
    """Unload this run's other models; stop if a model the run does not use is loaded (another job)."""
    ours_n = {normalize_tag(m) for m in ours}
    for loaded in await backend.loaded():
        if model and normalize_tag(loaded) == normalize_tag(model):
            continue
        if normalize_tag(loaded) not in ours_n:
            raise RunStop(
                f"Ollama has {loaded} loaded, which this run does not use: is a pipeline benchmark running? "
                f"Wait for it to finish (or `ollama stop {loaded}`), then run again; finished work is kept.")
        logger.info("Unloading {}", loaded)
        await backend.unload(loaded)


async def logprob_check(backend: Backend, model: str) -> dict:
    """
    What the server's log-probs are: present for greedy and sampled tokens? At T = 0, are they
    the model's own probabilities, or ≈ 0 because they are reported after temperature (then
    greedy answers are generated as T = 1, top_k = 1)?
    """
    prompt = "Answer in one short sentence: what is the capital of France?"
    g = await backend.generate(model, prompt, temperature=0.0, max_tokens=12, logprobs=True, top_logprobs=5)
    s = await backend.generate(model, prompt, temperature=1.0, max_tokens=12, seed=1, logprobs=True)
    flat = lambda gen: bool(gen.logprobs) and all(lp > -1e-6 for lp in gen.logprobs) and all(
        len(alts) < 2 or alts[1][1] < -30 for alts in gen.top_logprobs)
    result = {"greedy_logprobs": bool(g.logprobs), "greedy_top_logprobs": bool(g.top_logprobs),
              "sample_logprobs": bool(s.logprobs), "greedy_flat_at_t0": flat(g)}
    if not g.logprobs:
        result["mode"] = "none"
    elif result["greedy_flat_at_t0"]:
        g1 = await backend.generate(model, prompt, temperature=1.0, top_k=1, max_tokens=12, logprobs=True,
                                    top_logprobs=5)
        result["topk1_informative"] = bool(g1.logprobs) and not flat(g1)
        result["mode"] = "topk1" if result["topk1_informative"] else "none"
    else:
        result["mode"] = "raw"
    hot = await backend.generate(model, prompt, temperature=1.5, max_tokens=2, seed=1, logprobs=True, top_logprobs=5)
    cool = await backend.generate(model, prompt, temperature=0.5, max_tokens=2, seed=1, logprobs=True, top_logprobs=5)
    if hot.top_logprobs and cool.top_logprobs:
        result["after_temperature"] = abs(hot.top_logprobs[0][0][1] - cool.top_logprobs[0][0][1]) > 0.05
    return result
