"""Summarize a sweep's results.jsonl, one row per fault spec (and recovery method).

    python -m cone.summarize runs/eval [--gate runs/eval]

Rates are over all runs, failed runs included; the spread is the min and max
of the per-seed rates. The first run is reported as the cold start and left
out of the medians.

Clean gate: a faulted or recovery row counts only if the clean run with the
same task and seed put the true number in the report. --gate names the sweep
whose clean runs decide (default: the sweep itself).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(path: str | Path) -> list[dict]:
    lines = Path(path, "results.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def clean_passes(rows: list[dict]) -> set[tuple[str, int]]:
    return {(r["task"], r["seed"]) for r in rows
            if r["fault"] == "none" and "method" not in r and r["status"] == "ok"
            and r["score"]["key_fact_in_report"]}


def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None


def summarize(rows: list[dict], gate: set[tuple[str, int]] | None = None) -> dict[str, dict]:
    cold = rows[0]["wall_seconds"] if rows else None
    warm_ids = {id(r) for r in rows[1:]}
    dropped = defaultdict(int)
    by_spec = defaultdict(list)
    for r in rows:
        spec = r["fault"] + (f"|{r['method']}" if "method" in r else "")
        if gate is not None and r["fault"] != "none" and (r["task"], r["seed"]) not in gate:
            dropped[spec] += 1
            continue
        by_spec[spec].append(r)
    out = {"_sweep": {"runs": len(rows), "cold_start_seconds": cold,
                      "clean_gate_passing": len(gate) if gate is not None else None}}
    for spec, rs in by_spec.items():
        ok = [r for r in rs if r["status"] == "ok"]
        s = {"runs": len(rs), "errors": len(rs) - len(ok), "dropped_by_clean_gate": dropped[spec]}
        if "|" in spec:  # recovery rows
            recs = [r["recovery"] for r in ok]
            s |= {"repaired_rate": sum(x["report_ok"] for x in recs) / len(rs),
                  "recomputed_calls": mean(x["stats"]["calls"] - x["stats"]["cache_hits"]
                                           for x in recs),
                  "tokens": mean(x["stats"]["prompt_tokens"] + x["stats"]["completion_tokens"]
                                 for x in recs),
                  "seconds": mean(x["stats"]["seconds"] for x in recs),
                  "residual_irreversible_rate": mean(bool(x["residual_irreversible"])
                                                     for x in recs)}
            out[spec] = s
            continue
        per_seed = defaultdict(list)
        for r in rs:
            per_seed[r["seed"]].append(r["status"] == "ok" and r["score"]["key_fact_in_report"])
        seed_rates = [sum(v) / len(v) for v in per_seed.values()]
        warm = [r["wall_seconds"] for r in rs if id(r) in warm_ids]
        s |= {"key_fact_rate": sum(seed_rates) / len(seed_rates),
              "key_fact_rate_min_max": [min(seed_rates), max(seed_rates)],
              "tokens_per_run": mean(r["stats"]["prompt_tokens"] + r["stats"]["completion_tokens"]
                                     for r in ok),
              "wall_seconds_median": statistics.median(warm) if warm else None}
        checks = [c for r in ok for c in r.get("checks", [])]
        if checks:  # always-verify sweep
            s["checks_failed_rate"] = sum(not c["passed"] for c in checks) / len(checks)
            if spec != "none":
                agent = spec.split(":")[1]
                s["detected_rate"] = mean(any(c["agent"] == agent and not c["passed"]
                                              for c in r["checks"]) for r in ok)
        if spec != "none" and ok:
            stages = ok[0]["score"]["reached"].keys()
            s["reached_rate"] = {k: mean(r["score"]["reached"][k] for r in ok) for k in stages}
            s["unexpected_recipient_rate"] = mean(bool(r["score"]["unexpected_recipients"])
                                                  for r in ok)
        out[spec] = s
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--gate", help="sweep whose clean runs gate the faulted rows")
    a = ap.parse_args()
    rows = load(a.sweep)
    gate = clean_passes(load(a.gate) if a.gate else rows)
    print(json.dumps(summarize(rows, gate), indent=2))
