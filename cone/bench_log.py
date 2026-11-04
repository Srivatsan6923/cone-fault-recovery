"""Logging overhead: time per W1 run with stub agents, log on disk vs in memory.

    python -m cone.bench_log [--runs 300]

Stub agents cost almost nothing, so the difference is the log's cost per run.
"""
from __future__ import annotations

import argparse
import tempfile
import time

from .llm import LoggedLLM, StubLLM
from .log import Log
from .services import World
from .workflows.w1 import run_w1

TASK = {"id": "t0", "question": "What was ACME's revenue growth in 2025?",
        "sources": ["ACME revenue grew 12.5% in 2025.", "ACME is based in Ohio."],
        "key_fact": "12.5%", "recipient": "client@example.test"}


def echo(messages):
    return "TO: client@example.test\n" + messages[-1]["content"][:400]


def per_run_ms(n: int, path_of) -> float:
    start = time.perf_counter()
    for i in range(n):
        log = Log(path_of(i))
        run_w1(TASK, LoggedLLM(StubLLM(echo), log), World(log))
    return (time.perf_counter() - start) / n * 1000


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    n = ap.parse_args().runs
    tmp = tempfile.mkdtemp()
    disk = per_run_ms(n, lambda i: f"{tmp}/r{i}")
    memory = per_run_ms(n, lambda i: None)
    print(f"disk {disk:.2f} ms/run, memory {memory:.2f} ms/run")
