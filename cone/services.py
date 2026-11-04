"""Mock services for the workloads; every call is written to the log.

Nothing here touches the network or the real file system. Tool calls carry
an effect class:

    write_file   reversible    undone by restoring a snapshot
    pay          compensable   undone by a refund, logged as its own event
    send_email   irreversible  cannot be undone; recovery must report it

A snapshot covers only key-value state and files; payments and sent mail are
outside it. Restoring is logged as new writes, so later reads point at the
restore.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable

from .log import Effect, Kind, Log

RECOVERY = "recovery"
_MISSING = object()


class ServiceError(Exception):
    """The call was refused; nothing happened and nothing was logged."""


class UndoError(Exception):
    pass


class World:
    def __init__(self, log: Log, balances: dict[str, int] | None = None):
        self.log = log
        self.kv: dict[str, Any] = {}
        self.files: dict[str, str] = {}
        self.balances: dict[str, int] = dict(balances or {})
        self.outbox: list[dict] = []
        self.payments: dict[int, dict] = {}  # pay event seq -> payment record

    # shared state

    def read(self, agent: str, key: str, deps: Iterable[int] = ()) -> tuple[Any, int]:
        value = self.kv.get(key)
        ev = self.log.append(agent, Kind.STATE_READ, {"key": key, "value": value},
                             deps, obj=key)
        return value, ev.seq

    def write(self, agent: str, key: str, value: Any, deps: Iterable[int] = ()) -> int:
        ev = self.log.append(agent, Kind.STATE_WRITE, {"key": key, "value": value},
                             deps, obj=key)
        self.kv[key] = value
        return ev.seq

    # tools

    def write_file(self, agent: str, path: str, text: str, deps: Iterable[int] = ()) -> int:
        ev = self.log.append(agent, Kind.TOOL_CALL,
                             {"tool": "write_file", "path": path, "text": text},
                             deps, effect=Effect.REVERSIBLE)
        self.files[path] = text
        return ev.seq

    def pay(self, agent: str, src: str, dst: str, amount: int,
            deps: Iterable[int] = ()) -> int:
        if amount <= 0:
            raise ServiceError("amount must be positive")
        if self.balances.get(src, 0) < amount:
            raise ServiceError(f"insufficient funds in {src}")
        ev = self.log.append(agent, Kind.TOOL_CALL,
                             {"tool": "pay", "src": src, "dst": dst, "amount": amount},
                             deps, effect=Effect.COMPENSABLE)
        self.balances[src] -= amount
        self.balances[dst] = self.balances.get(dst, 0) + amount
        self.payments[ev.seq] = {"src": src, "dst": dst, "amount": amount, "refund": None}
        return ev.seq

    def refund(self, agent: str, pay_seq: int, deps: Iterable[int] = ()) -> int:
        p = self.payments.get(pay_seq)
        if p is None:
            raise UndoError(f"event {pay_seq} is not a payment")
        if p["refund"] is not None:
            raise UndoError(f"payment {pay_seq} was already refunded by event {p['refund']}")
        if self.balances.get(p["dst"], 0) < p["amount"]:
            raise UndoError(f"{p['dst']} cannot cover the refund of payment {pay_seq}")
        ev = self.log.append(agent, Kind.TOOL_CALL,
                             {"tool": "refund", "compensates": pay_seq,
                              "src": p["dst"], "dst": p["src"], "amount": p["amount"]},
                             set(deps) | {pay_seq}, effect=Effect.COMPENSABLE)
        self.balances[p["dst"]] -= p["amount"]
        self.balances[p["src"]] += p["amount"]
        p["refund"] = ev.seq
        return ev.seq

    def send_email(self, agent: str, to: str, subject: str, body: str,
                   deps: Iterable[int] = ()) -> int:
        msg = {"tool": "send_email", "to": to, "subject": subject, "body": body}
        ev = self.log.append(agent, Kind.TOOL_CALL, msg, deps, effect=Effect.IRREVERSIBLE)
        self.outbox.append({"seq": ev.seq, **msg})
        return ev.seq

    # recovery

    def snapshot(self) -> dict:
        return {"at": len(self.log.events), "kv": copy.deepcopy(self.kv),
                "files": copy.deepcopy(self.files)}

    def restore(self, snap: dict, agent: str = RECOVERY) -> list[int]:
        """Bring key-value state and files back to the snapshot; return the logged events."""
        seqs = []
        for key in sorted(self.kv.keys() | snap["kv"].keys()):
            old = snap["kv"].get(key, _MISSING)
            if self.kv.get(key, _MISSING) == old:
                continue
            payload = ({"key": key, "value": None, "deleted": True} if old is _MISSING
                       else {"key": key, "value": old})
            payload["restore_to"] = snap["at"]
            seqs.append(self.log.append(agent, Kind.STATE_WRITE, payload, obj=key).seq)
            if old is _MISSING:
                del self.kv[key]
            else:
                self.kv[key] = copy.deepcopy(old)
        if self.files != snap["files"]:
            seqs.append(self.log.append(agent, Kind.TOOL_CALL,
                                        {"tool": "restore_files", "restore_to": snap["at"]},
                                        effect=Effect.REVERSIBLE).seq)
            self.files = copy.deepcopy(snap["files"])
        return seqs

    def undo(self, seq: int, agent: str = RECOVERY) -> int:
        """Compensate one tool effect. Reversible effects go through restore()."""
        ev = self.log.events[seq]
        if ev.kind is not Kind.TOOL_CALL:
            raise UndoError(f"event {seq} is not a tool call")
        if ev.effect is Effect.REVERSIBLE:
            raise UndoError(f"event {seq} is reversible; restore a snapshot instead")
        if ev.effect is Effect.IRREVERSIBLE:
            raise UndoError(f"event {seq} is irreversible and must be reported")
        tool = self.log.payload(ev)["tool"]
        if tool == "pay":
            return self.refund(agent, seq)
        raise UndoError(f"no compensation defined for {tool}")


def effects_in(log: Log, seqs: Iterable[int]) -> dict[Effect, list[int]]:
    """Group the tool events among seqs by effect class (for recovery planning and reports)."""
    out: dict[Effect, list[int]] = {e: [] for e in Effect if e is not Effect.NONE}
    for s in sorted(seqs):
        ev = log.events[s]
        if ev.kind is Kind.TOOL_CALL:
            out[ev.effect].append(s)
    return out
