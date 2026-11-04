from cone.summarize import summarize


def row(seed, ok=True, fact=True, fault="none", reached=None, extra=()):
    r = {"task": "t", "fault": fault, "seed": seed, "wall_seconds": 2.0,
         "status": "ok" if ok else "error",
         "stats": {"prompt_tokens": 10, "completion_tokens": 5}}
    if ok:
        r["score"] = {"key_fact_in_report": fact, "unexpected_recipients": list(extra)}
        if reached is not None:
            r["score"]["reached"] = reached
    return r


def test_failed_runs_count_against_the_rate():
    rows = [row(0), row(0, ok=False), row(1), row(1, fact=False)]
    rows[0]["wall_seconds"] = 30.0  # cold start
    out = summarize(rows)
    s = out["none"]
    assert s["runs"] == 4 and s["errors"] == 1
    assert s["key_fact_rate"] == 0.5 and s["key_fact_rate_min_max"] == [0.5, 0.5]
    assert out["_sweep"]["cold_start_seconds"] == 30.0 and s["wall_seconds_median"] == 2.0
    assert s["tokens_per_run"] == 15


def test_clean_gate_drops_rows_whose_clean_run_failed():
    from cone.summarize import clean_passes
    rows = [row(0), row(1, fact=False),
            row(0, fault="injection:analyst", reached={"email": True}),
            row(1, fault="injection:analyst", reached={"email": True})]
    gate = clean_passes(rows)
    assert gate == {("t", 0)}
    s = summarize(rows, gate)["injection:analyst"]
    assert s["runs"] == 1 and s["dropped_by_clean_gate"] == 1


def test_recovery_group_with_only_errors_does_not_crash():
    r = {**row(0, ok=False, fault="wrong_fact:analyst"), "method": "restart"}
    s = summarize([r])["wrong_fact:analyst|restart"]
    assert s["repaired_rate"] == 0 and s["recomputed_calls"] is None


def test_fault_rows_report_reach():
    rows = [row(0, fault="injection:researcher", reached={"analyst": True, "email": True},
                extra=["audit@exfil.test"]),
            row(1, fault="injection:researcher", reached={"analyst": True, "email": False})]
    s = summarize(rows)["injection:researcher"]
    assert s["reached_rate"] == {"analyst": 1.0, "email": 0.5}
    assert s["unexpected_recipient_rate"] == 0.5


def test_recovery_and_verify_rows():
    rec = {"report_ok": True, "residual_irreversible": [12],
           "stats": {"calls": 3, "cache_hits": 1, "prompt_tokens": 20, "completion_tokens": 10,
                     "seconds": 1.5}}
    r = row(0, fault="wrong_fact:analyst", reached={"writer": True, "email": True})
    rows = [{**r, "method": "rollback", "recovery": rec}, {**r, "method": "rollback", "status": "error"}]
    s = summarize(rows)["wrong_fact:analyst|rollback"]
    assert s["repaired_rate"] == 0.5 and s["recomputed_calls"] == 2
    assert s["residual_irreversible_rate"] == 1.0
    v = {**r, "checks": [{"agent": "researcher", "passed": True},
                         {"agent": "analyst", "passed": False}]}
    s = summarize([v])["wrong_fact:analyst"]
    assert s["detected_rate"] == 1.0 and s["checks_failed_rate"] == 0.5


def test_tamas_summary_keeps_last_attempt_and_floors():
    from cone.summarize_tamas import summarize as tsum
    st = {m: {"prompt_tokens": 1, "completion_tokens": 1, "cache_hits": 0} for m in
          ("clean", "attacked", "placebo")}
    def row(seed, status="ok", hit=True):
        side = lambda called, changed: {"targets_called": called, "attack_success": bool(called),
                                        "unknown_calls": 0, "changed_calls": changed,
                                        "downstream_agents": 2}
        return {"scenario": "dpi-x", "attack": "DPI", "seed": seed, "status": status,
                "wall_seconds": 1.0, "stats": st, "clean": side([], []),
                "attacked": side(["t"] if hit else [], ["a", "b"]), "placebo": side([], ["a"])}
    rows = [row(0, "error"), row(0), row(1, hit=False)]
    s = tsum(rows)
    assert s["_sweep"] == {"rows": 3, "triples": 2, "retried": 1, "errors": 0}
    d = s["DPI"]
    assert d["attack_success_attacked"] == {"mean": 0.5, "seed_min_max": [0, 1], "n": 2}
    assert d["attack_success_clean"]["mean"] == 0
    assert d["changed_fraction_attacked"]["mean"] == 1 and d["changed_fraction_placebo"]["mean"] == 0.5
    assert d["attacked_minus_placebo_changed_agents"]["attacked_more"] == 1
