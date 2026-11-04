"""Append-only, hash-chained provenance log.

Events cover messages between agents, shared state, tool effects, and LLM
calls. Each event lists the earlier events it depends on, carries a vector
clock, and stores its payload by content hash. State events carry per-object
versions, and a state read records the write it read from, so dependencies
are captured by the log rather than declared by the agent. Tool events carry
an effect class that says how the effect can be undone.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class Kind(str, Enum):
    LLM_CALL = "llm_call"
    MSG_SEND = "msg_send"
    MSG_RECV = "msg_recv"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    TOOL_CALL = "tool_call"


class Plane(str, Enum):
    LOCAL = "local"
    MESSAGE = "message"
    STATE = "state"
    EFFECT = "effect"


PLANE = {
    Kind.LLM_CALL: Plane.LOCAL,
    Kind.MSG_SEND: Plane.MESSAGE,
    Kind.MSG_RECV: Plane.MESSAGE,
    Kind.STATE_READ: Plane.STATE,
    Kind.STATE_WRITE: Plane.STATE,
    Kind.TOOL_CALL: Plane.EFFECT,
}


class Effect(str, Enum):
    NONE = "none"
    REVERSIBLE = "reversible"      # undone by restoring a snapshot
    COMPENSABLE = "compensable"    # undone by a separate compensating action
    IRREVERSIBLE = "irreversible"  # cannot be undone; must be reported


class LogError(Exception):
    pass


GENESIS = "0" * 64
EVENTS_FILE = "events.jsonl"
BLOB_DIR = "blobs"


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Event:
    seq: int
    agent: str
    kind: Kind
    deps: tuple[int, ...]
    payload: str                 # sha256 of the canonical payload
    clock: dict[str, int]
    obj: str | None = None       # state object name, state events only
    version: int | None = None   # version written, or version read
    effect: Effect = Effect.NONE
    prev: str = GENESIS
    hash: str = ""

    @property
    def plane(self) -> Plane:
        return PLANE[self.kind]

    def body(self) -> dict:
        """Every field the hash covers (all but the hash itself)."""
        return {
            "seq": self.seq, "agent": self.agent, "kind": self.kind.value,
            "deps": list(self.deps), "payload": self.payload, "clock": self.clock,
            "obj": self.obj, "version": self.version, "effect": self.effect.value,
            "prev": self.prev,
        }

    def to_json(self) -> dict:
        return {**self.body(), "hash": self.hash}

    @classmethod
    def from_json(cls, d: dict) -> Event:
        return cls(seq=d["seq"], agent=d["agent"], kind=Kind(d["kind"]),
                   deps=tuple(d["deps"]), payload=d["payload"], clock=dict(d["clock"]),
                   obj=d["obj"], version=d["version"], effect=Effect(d["effect"]),
                   prev=d["prev"], hash=d["hash"])


def happened_before(a: Event, b: Event) -> bool:
    """Vector-clock order: True if a causally precedes b."""
    agents = a.clock.keys() | b.clock.keys()
    return a.seq != b.seq and all(a.clock.get(x, 0) <= b.clock.get(x, 0) for x in agents)


class Log:
    def __init__(self, path: str | Path | None = None):
        self.events: list[Event] = []
        self.blobs: dict[str, Any] = {}
        self._clocks: dict[str, dict[str, int]] = {}
        self._versions: dict[str, tuple[int, int]] = {}  # obj -> (version, seq of write)
        self._path = Path(path) if path is not None else None
        if self._path is not None:
            if (self._path / EVENTS_FILE).exists():
                raise LogError(f"{self._path} already holds a log; use Log.load")
            (self._path / BLOB_DIR).mkdir(parents=True, exist_ok=True)

    def append(self, agent: str, kind: Kind | str, payload: Any,
               deps: Iterable[int] = (), obj: str | None = None,
               effect: Effect | str = Effect.NONE) -> Event:
        kind, effect = Kind(kind), Effect(effect)
        seq = len(self.events)
        deps = set(deps)
        for d in deps:
            if not 0 <= d < seq:
                raise LogError(f"dependency {d} does not name an earlier event")
        if (kind is Kind.TOOL_CALL) != (effect is not Effect.NONE):
            raise LogError("tool calls need an effect class; other events must not have one")

        clock = dict(self._clocks.get(agent, {}))
        if kind is Kind.MSG_RECV:
            sends = [d for d in deps if self.events[d].kind is Kind.MSG_SEND]
            if len(sends) != 1:
                raise LogError("msg_recv must depend on exactly one msg_send")
            for a, c in self.events[sends[0]].clock.items():
                clock[a] = max(clock.get(a, 0), c)
        clock[agent] = clock.get(agent, 0) + 1

        version = None
        if kind in (Kind.STATE_READ, Kind.STATE_WRITE):
            if not obj:
                raise LogError(f"{kind.value} needs an object name")
            current, writer = self._versions.get(obj, (0, None))
            if kind is Kind.STATE_READ:
                version = current
                if writer is not None:
                    deps.add(writer)  # read-from edge
            else:
                version = current + 1
        elif obj is not None:
            raise LogError("only state events name an object")

        data = canonical(payload)
        digest = sha256(data)
        prev = self.events[-1].hash if self.events else GENESIS
        ev = Event(seq, agent, kind, tuple(sorted(deps)), digest, clock,
                   obj, version, effect, prev)
        ev = replace(ev, hash=sha256(canonical(ev.body())))

        self.events.append(ev)
        self.blobs[digest] = payload
        self._clocks[agent] = clock
        if kind is Kind.STATE_WRITE:
            self._versions[obj] = (version, seq)
        if self._path is not None:
            blob = self._path / BLOB_DIR / f"{digest}.json"
            if not blob.exists():
                blob.write_bytes(data)
            with open(self._path / EVENTS_FILE, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(ev.to_json(), sort_keys=True) + "\n")
        return ev

    def payload(self, ev: Event | int) -> Any:
        if isinstance(ev, int):
            ev = self.events[ev]
        return self.blobs[ev.payload]

    def verify(self) -> None:
        """Recompute the hash chain; raise LogError at the first bad event."""
        prev = GENESIS
        for i, ev in enumerate(self.events):
            if ev.seq != i:
                raise LogError(f"seq {i}: event says seq {ev.seq}")
            if ev.prev != prev:
                raise LogError(f"seq {i}: chain broken (prev hash mismatch)")
            if ev.hash != sha256(canonical(ev.body())):
                raise LogError(f"seq {i}: event hash mismatch")
            if any(not 0 <= d < i for d in ev.deps):
                raise LogError(f"seq {i}: dependency on a later event")
            if ev.payload not in self.blobs:
                raise LogError(f"seq {i}: payload missing")
            if sha256(canonical(self.blobs[ev.payload])) != ev.payload:
                raise LogError(f"seq {i}: payload hash mismatch")
            prev = ev.hash

    @classmethod
    def load(cls, path: str | Path) -> Log:
        path = Path(path)
        log = cls()
        for name in (path / BLOB_DIR).glob("*.json"):
            data = name.read_bytes()
            if sha256(data) != name.stem:
                raise LogError(f"blob {name.name}: content does not match its name")
            log.blobs[name.stem] = json.loads(data)
        with open(path / EVENTS_FILE, encoding="utf-8") as fh:
            log.events = [Event.from_json(json.loads(line)) for line in fh if line.strip()]
        log.verify()
        for ev in log.events:
            log._clocks[ev.agent] = ev.clock
            if ev.kind is Kind.STATE_WRITE:
                log._versions[ev.obj] = (ev.version, ev.seq)
        log._path = path
        return log

    def cone(self, seeds: Iterable[int], program_order: bool = True) -> set[int]:
        """Forward cone: the seeds plus every event that transitively depends on them.

        With program_order, each event also depends on the same agent's previous
        event. That is the conservative choice: an agent's context window is hidden
        local state, so a later step may use anything the agent saw earlier.
        """
        seen = set(seeds)
        for s in seen:
            if not 0 <= s < len(self.events):
                raise LogError(f"seed {s} is not in the log")
        children: list[list[int]] = [[] for _ in self.events]
        last: dict[str, int] = {}
        for ev in self.events:
            for d in ev.deps:
                children[d].append(ev.seq)
            if program_order and ev.agent in last:
                children[last[ev.agent]].append(ev.seq)
            last[ev.agent] = ev.seq
        queue = deque(seen)
        while queue:
            for c in children[queue.popleft()]:
                if c not in seen:
                    seen.add(c)
                    queue.append(c)
        return seen
