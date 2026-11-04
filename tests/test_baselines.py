import pytest

from cone.baselines import make_verifier, recover
from cone.llm import LoggedLLM, StubLLM
from cone.log import Log
from cone.services import World
from cone.workflows.w1 import run_w1
from test_w1 import INJECTION, TASK, WRONG_FACT, gullible_team


def team_and_judge(messages):
    if messages[0]["content"].startswith("You check"):
        out = messages[1]["content"].split("Teammate's output:\n", 1)[1]
        return "FAIL" if "41.7%" in out or "exfil" in out else "PASS"
    return gullible_team(messages)


BACKEND = StubLLM(team_and_judge)


def faulted(fault):
    world = World(Log())
    llm = LoggedLLM(BACKEND, world.log)
    return run_w1(TASK, llm, world, fault), world, llm


def recomputed(rec):
    return rec["stats"]["calls"] - rec["stats"]["cache_hits"]


@pytest.mark.parametrize("agent,expected", [("researcher", 3), ("analyst", 2), ("writer", 1)])
def test_rollback_recomputes_from_the_faulted_stage(agent, expected):
    result, world, llm = faulted({**WRONG_FACT, "agent": agent})
    rec = recover("rollback", TASK, world, BACKEND, llm.params, llm.cache, result)
    assert recomputed(rec) == expected
    assert rec["report_ok"]
    world.log.verify()


def test_restart_recomputes_everything_and_reports_sent_mail():
    result, world, llm = faulted(WRONG_FACT)
    rec = recover("restart", TASK, world, BACKEND, llm.params, llm.cache, result)
    assert recomputed(rec) == 3
    assert rec["report_ok"] and rec["unexpected_recipients"] == []
    assert rec["residual_irreversible"] == [world.outbox[0]["seq"]]
    assert rec["first_seq"] > max(m["seq"] for m in world.outbox[:1])


def test_injection_residual_includes_the_outside_address():
    result, world, llm = faulted(INJECTION)
    rec = recover("rollback", TASK, world, BACKEND, llm.params, llm.cache, result)
    leaked = {world.log.payload(s)["to"] for s in rec["residual_irreversible"]}
    assert "audit@exfil.test" in leaked
    assert rec["unexpected_recipients"] == []


def test_unknown_method():
    result, world, llm = faulted(WRONG_FACT)
    with pytest.raises(ValueError):
        recover("pray", TASK, world, BACKEND, llm.params, llm.cache, result)


def test_always_verify_catches_and_regenerates():
    world = World(Log())
    llm = LoggedLLM(BACKEND, world.log)
    result = run_w1(TASK, llm, world, WRONG_FACT, make_verifier(llm))
    assert [c["passed"] for c in result.checks] == [False, True, True]
    assert result.score["key_fact_in_report"]
    assert not any(result.score["reached"][a] for a in ("analyst", "writer", "email"))


def test_always_verify_cost_on_a_clean_run():
    world = World(Log())
    llm = LoggedLLM(BACKEND, world.log)
    result = run_w1(TASK, llm, world, verify=make_verifier(llm))
    assert all(c["passed"] for c in result.checks)
    assert llm.stats["calls"] == 6  # three agents plus three judge calls


def test_judge_fails_closed_on_unclear_reply():
    world = World(Log())
    llm = LoggedLLM(StubLLM(lambda m: "Looks fine to me"), world.log)
    _, s = llm.call("analyst", [{"role": "user", "content": "x"}])
    passed, _ = make_verifier(llm)("analyst", [{"role": "user", "content": "x"}], "y", s)
    assert passed is False
