"""Run W1 over a task split, clean or with faults, and save every run.

    python -m cone.run_w1 --config configs/laptop-qwen7b.json --tasks data/w1_tasks.json \
        --split dev --seeds 0 1 2 --faults none wrong_fact:researcher --out runs/pilot \
        [--verify] [--recover restart rollback]

Each run gets its own directory (log, blobs, cache, meta.json), and one row
per run, failed runs included, is appended to <out>/results.jsonl. Runs whose
meta.json exists are skipped, so a sweep can be resumed.

--verify puts a judge on every hand-off. --recover adds one run per method in
<run>-<method>/: the faulted run is replayed from its saved cache, then
recovered (see cone/baselines.py).
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

from .baselines import make_verifier, recover
from .llm import LLMError, LoggedLLM, OpenAICompatLLM, load_cache, save_cache
from .log import Log
from .services import World
from .workflows.w1 import make_fault, run_w1, save_run


def parse_fault(spec: str) -> tuple[str, str] | None:
    if spec == "none":
        return None
    kind, agent = spec.split(":")
    return kind, agent


def fresh(run_dir: Path) -> bool:
    """False if the run is done; a crashed earlier attempt is moved aside for inspection."""
    if (run_dir / "meta.json").exists():
        return False
    if run_dir.exists():
        run_dir.rename(run_dir.with_name(f"{run_dir.name}.crashed-{int(time.time())}"))
    return True


def run_faulted(task, fault, backend, params, verify, log):
    """The original run: clean or faulted, optionally with a judge on every hand-off."""
    llm = LoggedLLM(backend, log, params)
    result = run_w1(task, llm, World(log), fault, make_verifier(llm) if verify else None)
    return result, llm, {"score": result.score, "stats": result.stats, "checks": result.checks}


def run_recovery(task, fault, backend, params, method, cache_path, log):
    """Replay the faulted run from its saved cache, then recover in the same world, so
    the email it sent is still in the outbox."""
    cache = load_cache(cache_path)
    world = World(log)
    llm = LoggedLLM(backend, log, params, cache=dict(cache))
    result = run_w1(task, llm, world, fault)
    rec = recover(method, task, world, backend, params, cache, result)
    return result, llm, {"score": result.score, "recovery": rec}


def record(out: Path, run_dir: Path, task: dict, row: dict, extra: dict, body) -> None:
    """Run body(log) -> (result, llm, row fields), save the run and append one results row;
    a failed model call is recorded as an error row."""
    log = Log(run_dir)
    start = time.perf_counter()
    try:
        result, llm, fields = body(log)
    except LLMError as exc:
        row.update(status="error", error=str(exc))
        (run_dir / "meta.json").write_text(json.dumps({**extra, **row}, indent=2),
                                           encoding="utf-8", newline="\n")
    else:
        save_run(run_dir, result, task, llm, {**extra, **fields})
        save_cache(llm.cache, run_dir / "cache.json")
        row.update(status="ok", **fields)
    row["wall_seconds"] = round(time.perf_counter() - start, 3)
    with open(out / "results.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps({k: row.get(k) for k in ("task", "fault", "method", "seed", "status")}),
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--faults", nargs="+", default=["none"],
                    help="none, or TYPE:AGENT; TYPE is wrong_fact, injection, timeout or stale")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks of the split")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--recover", nargs="*", default=[], choices=["restart", "rollback"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    config = json.loads(Path(a.config).read_text(encoding="utf-8"))
    data = json.loads(Path(a.tasks).read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in data["tasks"]}
    tasks = [by_id[i] for i in data[a.split]][:a.limit]
    backend = OpenAICompatLLM(config["base_url"], config["model"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for i, task in enumerate(tasks):
        for spec in a.faults:
            fs = parse_fault(spec)
            fault = make_fault(task, *fs, prev=tasks[i - 1]) if fs else None
            for seed in a.seeds:
                params = {**config["params"], "seed": seed}
                base = out / task["id"] / spec.replace(":", "-") / f"s{seed}"
                extra = {"config": config, "split": a.split, "seed": seed, "fault_spec": spec,
                         "verify": a.verify}
                row = {"task": task["id"], "fault": spec, "seed": seed, "verify": a.verify,
                       "dir": str(base)}

                if fresh(base):
                    record(out, base, task, row, extra,
                           partial(run_faulted, task, fault, backend, params, a.verify))
                if not fault or not (base / "cache.json").exists():
                    continue
                for method in a.recover:
                    rdir = base.with_name(f"{base.name}-{method}")
                    if fresh(rdir):
                        record(out, rdir, task, {**row, "method": method, "dir": str(rdir)}, extra,
                               partial(run_recovery, task, fault, backend, params, method,
                                       base / "cache.json"))


if __name__ == "__main__":
    main()
