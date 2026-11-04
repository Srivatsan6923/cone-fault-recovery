import pytest

from cone.log import Effect, Kind, Log
from cone.services import ServiceError, UndoError, World, effects_in


@pytest.fixture
def world():
    return World(Log(), balances={"client": 100, "vendor": 0})


def test_state_goes_through_the_log(world):
    w = world.write("researcher", "facts", ["a"])
    value, r = world.read("analyst", "facts")
    assert value == ["a"]
    assert world.log.events[r].deps == (w,)
    assert world.log.events[w].kind is Kind.STATE_WRITE


def test_each_tool_has_its_effect_class(world):
    f = world.write_file("writer", "report.md", "text")
    p = world.pay("ops", "client", "vendor", 30)
    e = world.send_email("writer", "boss@example.test", "report", "text")
    assert world.log.events[f].effect is Effect.REVERSIBLE
    assert world.log.events[p].effect is Effect.COMPENSABLE
    assert world.log.events[e].effect is Effect.IRREVERSIBLE
    assert effects_in(world.log, [f, p, e]) == {
        Effect.REVERSIBLE: [f], Effect.COMPENSABLE: [p], Effect.IRREVERSIBLE: [e]}
    world.log.verify()


def test_refund_restores_balances_and_is_logged(world):
    p = world.pay("ops", "client", "vendor", 30)
    assert world.balances == {"client": 70, "vendor": 30}
    r = world.undo(p)
    assert world.balances == {"client": 100, "vendor": 0}
    assert p in world.log.events[r].deps
    assert world.log.payload(r)["compensates"] == p


def test_refund_happens_once(world):
    p = world.pay("ops", "client", "vendor", 30)
    world.undo(p)
    with pytest.raises(UndoError, match="already refunded"):
        world.undo(p)
    assert world.balances == {"client": 100, "vendor": 0}


def test_compensation_can_fail(world):
    """A refund is semantic, not a restore: it fails if the payee spent the money."""
    p = world.pay("ops", "client", "vendor", 30)
    world.pay("ops", "vendor", "client", 20)
    with pytest.raises(UndoError, match="cannot cover"):
        world.undo(p)


def test_refused_payment_is_not_logged(world):
    n = len(world.log.events)
    with pytest.raises(ServiceError):
        world.pay("ops", "client", "vendor", 1000)
    assert len(world.log.events) == n


def test_email_cannot_be_undone(world):
    e = world.send_email("writer", "boss@example.test", "report", "text")
    with pytest.raises(UndoError, match="irreversible"):
        world.undo(e)
    assert len(world.outbox) == 1


def test_restore_rolls_back_owned_state_only(world):
    world.write("researcher", "facts", ["a"])
    snap = world.snapshot()
    world.write("researcher", "facts", ["WRONG"])
    world.write("researcher", "scratch", 1)
    world.write_file("writer", "report.md", "bad report")
    world.pay("ops", "client", "vendor", 30)
    world.send_email("writer", "boss@example.test", "report", "bad report")

    seqs = world.restore(snap)

    assert world.kv == {"facts": ["a"]}
    assert world.files == {}
    assert world.balances == {"client": 70, "vendor": 30}  # outside the snapshot
    assert len(world.outbox) == 1                           # outside the snapshot
    assert all(world.log.events[s].agent == "recovery" for s in seqs)
    # a read after the restore depends on the restore write, not on the bad write
    _, r = world.read("analyst", "facts")
    assert world.log.events[r].deps[0] in seqs
    world.log.verify()


def test_reversible_effect_is_undone_by_restore_not_undo(world):
    f = world.write_file("writer", "report.md", "text")
    with pytest.raises(UndoError, match="restore"):
        world.undo(f)
