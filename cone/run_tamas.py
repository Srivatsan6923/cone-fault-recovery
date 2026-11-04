"""Run TAMAS scenarios as clean / attacked / placebo triples and save every run.

    python -m cone.run_tamas --config configs/laptop-qwen7b.json --data data/tamas \
        --seeds 0 1 2 --out runs/tamas [--attack DPI Byzantine] [--limit N]

For each scenario and seed the clean run goes first; the attacked and placebo
runs start from its cache with the same seed, so agents whose input did not
change replay exactly. One row per triple is appended to <out>/results.jsonl,
failed triples included.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .llm import LLMError, LoggedLLM, OpenAICompatLLM
from .log import Log
from .run_w1 import fresh
from .services import World
from .workflows.tamas import MODES, load_scenarios, run_tamas, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--attack", nargs="+", default=["DPI", "Byzantine"])
    ap.add_argument("--limit", type=int, default=None, help="first N scenarios per attack")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    config = json.loads(Path(a.config).read_text(encoding="utf-8"))
    backend = OpenAICompatLLM(config["base_url"], config["model"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = [s for s in load_scenarios(a.data) if s["attack"] in a.attack]
    if a.limit:
        scenarios = [s for atk in a.attack for s in [x for x in scenarios if x["attack"] == atk][:a.limit]]

    for sc in scenarios:
        for seed in a.seeds:
            pdir = out / sc["id"] / f"s{seed}"
            if not fresh(pdir):
                continue
            params = {**config["params"], "seed": seed}
            row = {"scenario": sc["id"], "attack": sc["attack"], "seed": seed, "dir": str(pdir)}
            start = time.perf_counter()
            try:
                runs, stats = {}, {}
                for mode in MODES:  # clean first: the others start from its cache
                    llm = LoggedLLM(backend, Log(pdir / mode), params,
                                    cache=dict(runs and runs["clean"][1].cache or {}))
                    runs[mode] = (run_tamas(sc, llm, World(llm.log), mode), llm)
                    stats[mode] = llm.stats
            except LLMError as exc:
                row.update(status="error", error=str(exc))
            else:
                clean = runs["clean"][0]
                row.update(status="ok", clean=score(sc, clean), stats=stats,
                           **{m: score(sc, runs[m][0], clean) for m in ("attacked", "placebo")})
            row["wall_seconds"] = round(time.perf_counter() - start, 3)
            (pdir / "meta.json").write_text(json.dumps({"config": config, **row}, indent=2),
                                            encoding="utf-8", newline="\n")
            with open(out / "results.jsonl", "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("scenario", "seed", "status")}), flush=True)


if __name__ == "__main__":
    main()
