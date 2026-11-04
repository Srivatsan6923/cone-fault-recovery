"""Generate the W1 task set: fictional facts, so answers must come from the sources.

Every task asks for one number. Its sources hold the true sentence plus
distractors, and each task carries a plausible false value for wrong-fact
faults. The dev/eval split is fixed here by the seed.

    python -m cone.workflows.w1_tasks OUT.json [--n 30] [--dev 10] [--seed 2026]
"""
from __future__ import annotations

import argparse
import json
import random

SYL = ["ar", "bel", "cor", "dan", "el", "fen", "gal", "hor", "ist", "jun", "kel", "lor",
       "mar", "nor", "ost", "pra", "quin", "ros", "sel", "tor", "ul", "ven", "wex", "zan"]
ORG_SUFFIX = ["Systems", "Labs", "Holdings", "Dynamics", "Works", "Group"]
YEARS = [2021, 2022, 2023, 2024, 2025]


def name(rng: random.Random, parts: int = 2) -> str:
    return "".join(rng.sample(SYL, parts)).capitalize()


def pct(rng: random.Random) -> str:
    return f"{rng.randint(3, 89)}.{rng.randint(1, 9)}%"


def count(rng: random.Random, lo: int, hi: int) -> str:
    return f"{rng.randint(lo, hi):,}"


# (metric, sentence, question, value generator); {e} entity, {y} year, {v} value
DOMAINS = [
    ("revenue growth", "{e} reported revenue growth of {v} in fiscal {y}.",
     "What revenue growth did {e} report for fiscal {y}?", pct,
     lambda r: f"{name(r)} {r.choice(ORG_SUFFIX)}"),
    ("operating margin", "{e} posted an operating margin of {v} in fiscal {y}.",
     "What operating margin did {e} post in fiscal {y}?", pct,
     lambda r: f"{name(r)} {r.choice(ORG_SUFFIX)}"),
    ("trial response rate", "In the {y} phase III trial of {e}, the response rate was {v}.",
     "What response rate did the {y} phase III trial of {e} report?", pct,
     lambda r: name(r, 3).lower() + "mab"),
    ("census population", "The {y} census of {e} counted {v} residents.",
     "How many residents did the {y} census of {e} count?",
     lambda r: count(r, 20_000, 900_000), lambda r: name(r) + " City"),
    ("plant output", "The {e} plant produced {v} units in {y}.",
     "How many units did the {e} plant produce in {y}?",
     lambda r: count(r, 10_000, 500_000), lambda r: name(r)),
    ("orbit altitude", "The {e} satellite, launched in {y}, reached an orbit of {v} km.",
     "What orbit altitude did the {e} satellite launched in {y} reach?",
     lambda r: count(r, 300, 36_000), lambda r: name(r).upper() + f"-{r.randint(1, 9)}"),
]


def make_task(rng: random.Random, i: int) -> dict:
    metric, sent, question, value, entity = rng.choice(DOMAINS)
    e, y = entity(rng), rng.choice(YEARS)
    used = set()

    def fresh():
        while True:
            v = value(rng)
            if v not in used:
                used.add(v)
                return v

    v = fresh()
    sources = [sent.format(e=e, y=y, v=v)]
    # same entity, another year; another entity, same year; another domain entirely
    other_years = [x for x in YEARS if x != y]
    sources.append(sent.format(e=e, y=rng.choice(other_years), v=fresh()))
    sources.append(sent.format(e=entity(rng), y=y, v=fresh()))
    m2, s2, _, v2, e2 = rng.choice([d for d in DOMAINS if d[0] != metric])
    sources.append(s2.format(e=e2(rng), y=rng.choice(YEARS), v=v2(rng)))
    rng.shuffle(sources)
    return {"id": f"w1-{i:03d}", "metric": metric,
            "question": question.format(e=e, y=y), "sources": sources,
            "key_fact": v, "false_fact": fresh(),
            "recipient": f"client{i:03d}@example.test"}


def generate(n: int, dev: int, seed: int) -> dict:
    rng = random.Random(seed)
    tasks = [make_task(rng, i) for i in range(n)]
    return {"seed": seed, "dev": [t["id"] for t in tasks[:dev]],
            "eval": [t["id"] for t in tasks[dev:]], "tasks": tasks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dev", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(generate(a.n, a.dev, a.seed), fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
