"""Regenerate the tables and numbers from the saved runs.

    python -m cone.report_tables --out results

Writes w1_propagation.tex, w1_recovery.tex, w1_compact.tex, tamas.tex, cost.dat
and numbers.tex (macros). Missing values print as "--". Each table fragment
ends with its own bottom rule, because a rule right after an input inside a
tabular does not compile.
"""
from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

from . import bench_log
from . import summarize as w1
from . import summarize_tamas as tm
from .workflows.tamas import MODES

KINDS = [("wrong_fact", "Wrong fact", "accidental"), ("timeout", "Timeout", "accidental"),
         ("stale", "Stale output", "accidental"), ("injection", "Injection", "adversarial")]
AGENTS = ("researcher", "analyst", "writer")
STAGE = {"researcher": "extraction", "analyst": "answering", "writer": "reporting"}


def pct(x):
    return "--" if x is None else f"{100 * x:.0f}"


def num(x, fmt="{:.0f}"):
    return "--" if x is None else fmt.format(x)


def sign_test(more: int, less: int) -> float:
    """Two-sided exact binomial sign test, ties dropped."""
    n, k = more + less, min(more, less)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = w1.load(runs / "eval")
    gate = w1.clean_passes(rows)
    base = w1.summarize(rows, gate)
    ver = w1.summarize(w1.load(runs / "eval-verify"), gate)
    macros = {"cleanGate": len(gate), "cleanRate": pct(base["none"]["key_fact_rate"]),
              "cleanTokens": num(base["none"]["tokens_per_run"]),
              "cleanSeconds": num(base["none"]["wall_seconds_median"], "{:.1f}"),
              "coldSeconds": num(base["_sweep"]["cold_start_seconds"], "{:.1f}"),
              "verifyTokens": num(ver["none"]["tokens_per_run"]),
              "verifySeconds": num(ver["none"]["wall_seconds_median"], "{:.1f}"),
              "verifyFalseFail": pct(ver["none"].get("checks_failed_rate")),
              "wOneRuns": sum(1 for r in rows),
              "verifyRuns": ver["_sweep"]["runs"]}
    clean_dirs = list((runs / "eval").glob("w1-*/none/s*"))
    macros["logEvents"] = num(w1.mean(sum(1 for _ in open(d / "events.jsonl")) for d in clean_dirs))
    macros["logKB"] = num(w1.mean(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()
                                      and f.name not in ("cache.json", "meta.json"))
                                  for d in clean_dirs) / 1024, "{:.1f}")
    tmp = tempfile.mkdtemp()
    macros["logMs"] = num(bench_log.per_run_ms(200, lambda i: f"{tmp}/r{i}")
                          - bench_log.per_run_ms(200, lambda i: None), "{:.0f}")

    prop, rec = [], []
    for kind, label, cls in KINDS:
        for i, agent in enumerate(AGENTS):
            spec = f"{kind}:{agent}"
            s = base.get(spec, {})
            reached = s.get("reached_rate", {})
            cells = [pct(reached.get(x)) if x in reached else "" for x in ("analyst", "writer")]
            first = f"\\multirow{{3}}{{*}}{{{label} ({cls})}}" if i == 0 else ""
            prop.append(" & ".join([first, STAGE[agent], *cells, pct(reached.get("email")),
                                    pct(s.get("key_fact_rate")),
                                    pct(s.get("unexpected_recipient_rate"))]) + r" \\")
            r, b, v = base.get(f"{spec}|restart", {}), base.get(f"{spec}|rollback", {}), ver.get(spec, {})
            wasted = (None if not r or not b
                      else r["recomputed_calls"] - b["recomputed_calls"])
            dtok = (None if not v or not s else v["tokens_per_run"] - s["tokens_per_run"])
            rec.append(" & ".join([first, STAGE[agent],
                                   num(r.get("recomputed_calls"), "{:.2f}"), num(r.get("tokens")),
                                   num(b.get("recomputed_calls"), "{:.2f}"), num(b.get("tokens")),
                                   num(wasted, "{:.2f}"), pct(b.get("repaired_rate")),
                                   pct(b.get("residual_irreversible_rate")),
                                   pct(v.get("detected_rate")), pct(v.get("key_fact_rate")),
                                   pct(v.get("unexpected_recipient_rate")),
                                   num(dtok, "{:+.0f}")]) + r" \\")
        prop.append(r"\midrule")
        rec.append(r"\midrule")
    for name, body in (("w1_propagation", prop), ("w1_recovery", rec)):
        (out / f"{name}.tex").write_text("\n".join(body[:-1] + [r"\bottomrule"]) + "\n",
                                         encoding="utf-8", newline="\n")

    # compact main-text table: one block per fault type, one row per injection stage
    compact = []
    for kind, label, cls in KINDS:
        for i, agent in enumerate(AGENTS):
            spec = f"{kind}:{agent}"
            s, b, v = base.get(spec, {}), base.get(f"{spec}|rollback", {}), ver.get(spec, {})
            reached = s.get("reached_rate", {})
            stacked = label.replace(" ", r"\\")
            first = rf"\multirow{{3}}{{*}}{{\makecell[l]{{{stacked}}}}}" if i == 0 else ""
            compact.append(" & ".join(
                [first, STAGE[agent]] + [pct(reached[x]) if x in reached and x != agent else ""
                                       for x in ("analyst", "writer", "email")]
                + [pct(s.get("key_fact_rate")), pct(s.get("unexpected_recipient_rate")),
                   pct(b.get("residual_irreversible_rate")), pct(v.get("detected_rate")),
                   pct(v.get("unexpected_recipient_rate"))]) + r" \\")
        compact.append(r"\midrule")
    (out / "w1_compact.tex").write_text("\n".join(compact[:-1] + [r"\bottomrule"]) + "\n",
                                        encoding="utf-8", newline="\n")

    # model seconds per run of each faulted run, without and with always-verify
    def model_seconds(rs):
        by = {}
        for r in rs:
            if "method" not in r and r["status"] == "ok" and (r["task"], r["seed"]) in gate:
                by.setdefault(r["fault"], []).append(r["stats"]["seconds"])
        return {k: w1.mean(v) for k, v in by.items()}
    base_s, ver_s = model_seconds(rows), model_seconds(w1.load(runs / "eval-verify"))

    # cost per injection stage, averaged over the four fault types (Figure 2)
    lines = ["stage restart_tok rollback_tok verify_extra_tok restart_s rollback_s verify_extra_s"]
    cost = {}
    for agent in AGENTS:
        specs = [f"{k}:{agent}" for k, _, _ in KINDS]
        r = [base[f"{s}|restart"] for s in specs if f"{s}|restart" in base]
        b = [base[f"{s}|rollback"] for s in specs if f"{s}|rollback" in base]
        extra = [ver[s]["tokens_per_run"] - base[s]["tokens_per_run"]
                 for s in specs if s in ver and s in base]
        extra_s = [ver_s[s] - base_s[s] for s in specs if s in ver_s and s in base_s]
        cost[agent] = [w1.mean(x) for x in (
            [x["tokens"] for x in r], [x["tokens"] for x in b], extra,
            [x["seconds"] for x in r], [x["seconds"] for x in b], extra_s)]
        lines.append(" ".join([agent] + [f"{x:.2f}" for x in cost[agent]]))
    (out / "cost.dat").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # numbers quoted in the text
    extras = [ver[s]["tokens_per_run"] - base[s]["tokens_per_run"]
              for s in (f"{k}:{a}" for k, _, _ in KINDS for a in AGENTS) if s in ver]
    macros |= {
        "logPct": num(float(macros["logMs"]) / (base["none"]["wall_seconds_median"] * 10), "{:.1f}"),
        "staleResearcherReach": pct(base["stale:researcher"]["reached_rate"]["analyst"]),
        "injDropped": pct(1 - base["injection:researcher"]["reached_rate"]["analyst"]),
        "earlyCutoff": sum(1 for r in rows if r["fault"] == "injection:researcher"
                           and r.get("method") == "rollback" and r["status"] == "ok"
                           and r["recovery"]["stats"]["cache_hits"] > 0),
        "saveAnalyst": pct(1 - cost["analyst"][1] / cost["analyst"][0]),
        "saveWriter": pct(1 - cost["writer"][1] / cost["writer"][0]),
        "restartSeconds": num(w1.mean(c[3] for c in cost.values()), "{:.1f}"),
        "rollbackWriterSeconds": num(cost["writer"][4], "{:.1f}"),
        "verifyExtraMin": num(min(extras)), "verifyExtraMax": num(max(extras)),
        "rollbackAnalystSeconds": num(cost["analyst"][4], "{:.1f}"),
        "verifySecMin": num(min(c[5] for c in cost.values()), "{:.1f}"),
        "verifySecMax": num(max(c[5] for c in cost.values()), "{:.1f}")}
    for agent in AGENTS:
        macros[f"timeoutDet{STAGE[agent].title()}"] = pct(ver[f"timeout:{agent}"]["detected_rate"])

    t = tm.summarize(tm.load(runs / "tamas"))
    # Byzantine attack success: TAMAS judge label 0, last judgment per log (cone/judge_tamas.py)
    def judge_labels(d):
        if not (d / "results.jsonl").exists():
            return {}
        return {(r["scenario"], r["seed"], r["mode"]): r["label"]
                for r in w1.load(d) if r["status"] == "ok"}

    labels = judge_labels(runs / "tamas_judge")
    judged = {k: v == 0 for k, v in labels.items()}
    byz_asr = {m: w1.mean(v for (_, _, mode), v in judged.items() if mode == m) for m in MODES}
    macros["judgeLogs"] = len(judged)
    macros |= {f"byzJudge{m.title()}": pct(byz_asr[m]) for m in MODES}
    # second, independent pass over the same logs: how often the judge agrees with itself
    again = judge_labels(runs / "tamas_judge2")
    both = [k for k in labels if k in again]
    macros["judgeRepeat"] = len(both)
    macros["judgeAgree"] = pct(w1.mean(labels[k] == again[k] for k in both))
    if both:  # Cohen's kappa on the attack-success label (label 0)
        po = w1.mean((labels[k] == 0) == (again[k] == 0) for k in both)
        p1, p2 = w1.mean(labels[k] == 0 for k in both), w1.mean(again[k] == 0 for k in both)
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        macros["judgeKappa"] = num((po - pe) / (1 - pe), "{:.2f}")
    for m in MODES:
        macros[f"byzJudgeTwo{m.title()}"] = pct(w1.mean(again[k] == 0 for k in both if k[2] == m))
    lines = []
    for attack in ("DPI", "Byzantine"):
        s = t[attack]
        asr = [pct(s[f"attack_success_{m}"]["mean"]) if f"attack_success_{m}" in s
               else pct(byz_asr[m]) for m in ("clean", "attacked", "placebo")]
        d = s["attacked_minus_placebo_changed_agents"]
        p = sign_test(round(d["attacked_more"] * s["triples"]), round(d["placebo_more"] * s["triples"]))
        cf = [s[f"changed_fraction_{m}"] for m in ("attacked", "placebo")]
        lines.append(" & ".join([attack, *asr,
                                 f"{pct(cf[0]['mean'])} ({pct(cf[0]['seed_min_max'][0])}--{pct(cf[0]['seed_min_max'][1])})",
                                 f"{pct(cf[1]['mean'])} ({pct(cf[1]['seed_min_max'][0])}--{pct(cf[1]['seed_min_max'][1])})",
                                 f"{pct(d['attacked_more'])} / {pct(d['equal'])} / {pct(d['placebo_more'])}",
                                 f"{p:.1g}" if p >= 1e-3 else r"$<$0.001"]) + r" \\")
        macros[f"tamas{attack}Tokens"] = num(s["tokens_per_triple"])
    (out / "tamas.tex").write_text("\n".join(lines + [r"\bottomrule"]) + "\n", encoding="utf-8", newline="\n")
    byz = t["Byzantine"]
    macros |= {"byzAttacked": pct(byz["changed_fraction_attacked"]["mean"]),
               "byzPlacebo": pct(byz["changed_fraction_placebo"]["mean"]),
               "byzEqual": pct(byz["attacked_minus_placebo_changed_agents"]["equal"])}
    macros["dpiRuns"] = t["DPI"]["attack_success_attacked"]["n"]
    macros["byzDownstream"] = byz["changed_fraction_attacked"]["n"]
    macros["tamasTriples"] = t["_sweep"]["triples"]
    macros["tamasRetried"] = t["_sweep"]["retried"]

    (out / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()),
        encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
