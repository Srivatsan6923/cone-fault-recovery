import json
from dataclasses import replace

import pytest

from cone.log import (BLOB_DIR, EVENTS_FILE, Effect, Event, Kind, Log, LogError, Plane,
                      canonical, happened_before, sha256)


def w1(path=None) -> Log:
    """Toy W1 trace: researcher -> analyst -> writer, with one unrelated step each."""
    log = Log(path)
    log.append("writer", Kind.LLM_CALL, {"prompt": "load style guide"})            # 0
    log.append("researcher", Kind.LLM_CALL, {"prompt": "find facts"})              # 1
    log.append("researcher", Kind.STATE_WRITE, {"facts": ["a", "b"]}, [1], "facts")  # 2
    log.append("researcher", Kind.MSG_SEND, {"to": "analyst"}, [2])                # 3
    log.append("analyst", Kind.MSG_RECV, {"from": "researcher"}, [3])              # 4
    log.append("analyst", Kind.STATE_READ, {"facts": ["a", "b"]}, obj="facts")     # 5
    log.append("analyst", Kind.LLM_CALL, {"prompt": "analyse"}, [4, 5])            # 6
    log.append("analyst", Kind.MSG_SEND, {"to": "writer"}, [6])                    # 7
    log.append("writer", Kind.MSG_RECV, {"from": "analyst"}, [7])                  # 8
    log.append("writer", Kind.LLM_CALL, {"prompt": "draft"}, [8])                  # 9
    log.append("writer", Kind.TOOL_CALL, {"tool": "send_email"}, [9],
               effect=Effect.IRREVERSIBLE)                                         # 10
    log.append("analyst", Kind.LLM_CALL, {"prompt": "tidy notes"})                 # 11
    return log


def test_chain_verifies():
    log = w1()
    log.verify()
    assert log.events[0].prev == "0" * 64
    assert all(e.prev == p.hash for p, e in zip(log.events, log.events[1:]))


def test_planes_and_effects():
    log = w1()
    assert log.events[3].plane is Plane.MESSAGE
    assert log.events[2].plane is Plane.STATE
    assert log.events[10].plane is Plane.EFFECT
    assert log.events[10].effect is Effect.IRREVERSIBLE
    assert log.events[9].effect is Effect.NONE


def test_state_read_records_read_from_edge_and_version():
    log = w1()
    read = log.events[5]
    assert read.deps == (2,)
    assert read.version == 1
    log.append("analyst", Kind.STATE_WRITE, {"facts": ["a"]}, obj="facts")
    log.append("writer", Kind.STATE_WRITE, {"other": 1}, obj="other")
    assert log.events[-2].version == 2
    assert log.events[-1].version == 1


def test_vector_clocks():
    log = w1()
    assert log.events[8].clock["researcher"] == 3  # carried through the analyst
    assert happened_before(log.events[2], log.events[10])
    assert not happened_before(log.events[10], log.events[2])
    # the writer's first step and the researcher's first step are concurrent
    assert not happened_before(log.events[0], log.events[1])
    assert not happened_before(log.events[1], log.events[0])


def test_cone_with_program_order():
    log = w1()
    assert log.cone([2]) == set(range(2, 12))
    assert log.cone([0]) == {0, 8, 9, 10}


def test_cone_explicit_edges_only():
    log = w1()
    assert log.cone([2], program_order=False) == set(range(2, 11))


@pytest.mark.parametrize("args", [
    ("a", Kind.TOOL_CALL, {}, (), None, Effect.NONE),          # tool call without class
    ("a", Kind.LLM_CALL, {}, (), None, Effect.REVERSIBLE),     # class on a non-tool event
    ("a", Kind.MSG_RECV, {}, (), None, Effect.NONE),           # receive without a send
    ("a", Kind.STATE_READ, {}, (), None, Effect.NONE),         # state event without object
    ("a", Kind.LLM_CALL, {}, (), "x", Effect.NONE),            # object on a non-state event
    ("a", Kind.LLM_CALL, {}, (99,), None, Effect.NONE),        # dependency on a future event
])
def test_rejected_events_leave_log_unchanged(args):
    log = w1()
    before = list(log.events)
    agent, kind, payload, deps, obj, effect = args
    with pytest.raises(LogError):
        log.append(agent, kind, payload, deps, obj, effect)
    assert log.events == before


def test_persistence_roundtrip_and_continue(tmp_path):
    w1(tmp_path)
    log = Log.load(tmp_path)
    assert len(log.events) == 12
    assert log.payload(2) == {"facts": ["a", "b"]}
    ev = log.append("researcher", Kind.STATE_WRITE, {"facts": []}, obj="facts")
    assert ev.version == 2
    assert ev.clock["researcher"] == 4
    assert len(Log.load(tmp_path).events) == 13


def test_refuses_to_overwrite_existing_log(tmp_path):
    w1(tmp_path)
    with pytest.raises(LogError):
        Log(tmp_path)


def test_tampered_event_is_detected(tmp_path):
    w1(tmp_path)
    lines = (tmp_path / EVENTS_FILE).read_text(encoding="utf-8").splitlines()
    ev = json.loads(lines[6])
    ev["deps"] = [4]  # hide the read-from dependency of the analyst's call
    lines[6] = json.dumps(ev, sort_keys=True)
    (tmp_path / EVENTS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LogError, match="seq 6"):
        Log.load(tmp_path)


def test_rehashed_tamper_breaks_the_chain(tmp_path):
    w1(tmp_path)
    lines = (tmp_path / EVENTS_FILE).read_text(encoding="utf-8").splitlines()
    ev = Event.from_json(json.loads(lines[6]))
    ev = replace(ev, deps=(4,))
    ev = replace(ev, hash=sha256(canonical(ev.body())))  # event is self-consistent again
    lines[6] = json.dumps(ev.to_json(), sort_keys=True)
    (tmp_path / EVENTS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LogError, match="seq 7: chain broken"):
        Log.load(tmp_path)


def test_tampered_payload_is_detected(tmp_path):
    log = w1(tmp_path)
    blob = tmp_path / BLOB_DIR / f"{log.events[2].payload}.json"
    blob.write_text('{"facts":["a","c"]}', encoding="utf-8")
    with pytest.raises(LogError, match="blob"):
        Log.load(tmp_path)
