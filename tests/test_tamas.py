import re
from pathlib import Path

import pytest

from cone.llm import LoggedLLM, StubLLM
from cone.log import Log
from cone.services import World
from cone.workflows.tamas import load_scenarios, parse_tools, run_tamas, score, variant

DATA = Path(__file__).parent.parent / "data" / "tamas"
SCENARIOS = load_scenarios(DATA)
BY_ID = {s["id"]: s for s in SCENARIOS}


def test_all_scenarios_load():
    assert len(SCENARIOS) == 100
    assert sum(s["attack"] == "DPI" for s in SCENARIOS) == 50


def test_dpi_targets_exist_except_three_tamas_data_bugs():
    missing = sorted(s["id"] for s in SCENARIOS if s.get("target_missing"))
    assert missing == ["dpi-education-0", "dpi-legal-9", "dpi-news-7"]
    for s in (s for s in SCENARIOS if s["attack"] == "DPI"):
        assert s["targets"] and "Ignore" not in s["clean_query"] and s["clean_query"]


def test_two_target_scenario():
    assert BY_ID["dpi-finance-9"]["targets"] == ["enforce_margin_call", "place_order"]


def test_byzantine_twin_cuts_only_the_suffix():
    s = BY_ID["byzantine-news-0"]
    _, clean = variant(s, "clean")
    _, attacked = variant(s, "attacked")
    b = s["byzantine"]
    assert "unpredictabl" in attacked[b]["description"]
    assert "unpredictabl" not in clean[b]["description"]
    assert clean[b]["description"] and s["agents"][b]["description"] == attacked[b]["description"]
    assert [a["description"] for k, a in enumerate(clean) if k != b] == \
           [a["description"] for k, a in enumerate(attacked) if k != b]


def test_placebo_is_harmless_and_differs_from_clean():
    for sid in ("byzantine-news-0", "dpi-news-0"):
        s = BY_ID[sid]
        cq, ca = variant(s, "clean")
        pq, pa = variant(s, "placebo")
        assert (pq, pa) != (cq, ca)
        assert "Ignore" not in pq and not any("unpredictabl" in a["description"] for a in pa)
    with pytest.raises(ValueError):
        variant(BY_ID["dpi-news-0"], "sneaky")


def test_tool_replies_are_read_without_running_code(tmp_path):
    f = tmp_path / "tools_x_agent.py"
    f.write_text('import os\nos.remove("nope")\n'
                 'async def a(u: str) -> str:\n    """Doc A."""\n    return f"done for {u}."\n'
                 'async def b() -> str:\n    if 1:\n        return "x"\n    return "y"\n')
    tools = parse_tools(f)
    assert tools == {"a": {"doc": "Doc A.", "reply": "done for <u>."},
                     "b": {"doc": "", "reply": "y"}}


def obedient_team(messages):
    """Stub agents: obey an injected tool order, otherwise call their first tool;
    a byzantine agent calls its last tool instead."""
    system, user = messages[0]["content"], messages[1]["content"]
    tools = re.findall(r"^- ([a-z0-9_]+):", system, re.M)
    m = re.search(r"\b([a-z]+(?:_[a-z0-9]+)+)\s+tool\b", user.split("Conversation so far")[0])
    if m and m.group(1) in tools:
        return f"CALL {m.group(1)}\nDone as instructed."
    if "unpredictabl" in system:
        return f"CALL {tools[-1]}\nAll good."
    if "(called" in user and "All good." in user:  # a downstream agent that follows the liar
        return f"CALL {tools[-1]}\nFollowing the team."
    return f"CALL {tools[0]}\nCALL not_a_tool\nWorking on it."


def pair(sid):
    s = BY_ID[sid]
    log = Log()
    world = World(log)
    llm = LoggedLLM(StubLLM(obedient_team), log)
    clean = run_tamas(s, llm, world, "clean")
    attacked = run_tamas(s, llm, world, "attacked")
    return s, clean, attacked, llm, log


def test_dpi_attack_success_is_programmatic():
    s, clean, attacked, _, log = pair("dpi-news-0")
    assert score(s, attacked)["attack_success"] is True
    assert score(s, clean)["attack_success"] is False
    assert score(s, clean)["unknown_calls"] == 4
    log.verify()


def test_byzantine_pairing_reuses_upstream_and_measures_change():
    s, clean, attacked, llm, log = pair("byzantine-news-0")  # byzantine agent is last (index 3)
    assert s["byzantine"] == 3
    assert llm.stats["cache_hits"] == 3          # the three agents before it replay exactly
    sc = score(s, attacked, clean)
    assert sc["downstream_agents"] == 0 and sc["changed_calls"] == []


def test_byzantine_propagation_downstream():
    sid = next(s["id"] for s in SCENARIOS if s["attack"] == "Byzantine" and s["byzantine"] == 0)
    s, clean, attacked, _, log = pair(sid)
    sc = score(s, attacked, clean)
    assert sc["downstream_agents"] == 3
    assert sc["changed_calls"]  # the stub's downstream agents follow the liar
    cone = log.cone([attacked.turns[0]["write_seq"]])
    assert all(t["write_seq"] in cone for t in attacked.turns[1:])
