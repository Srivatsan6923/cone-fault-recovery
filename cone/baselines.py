"""Baselines on W1: full restart, rollback, and always-verify.

Recovery is told which agent was faulted. It re-runs W1 without the fault in
the same world and log; emails sent before recovery stay sent and are reported.

    restart   empty cache: every model call is recomputed
    rollback  reuse the cached calls upstream of the faulted agent; recompute
              the faulted agent and everything after it

On a linear chain this rollback is the same as replaying the fault's cone.
make_verifier() returns a judge that run_w1 calls on every hand-off.
"""
from __future__ import annotations

from .llm import LoggedLLM
from .services import World
from .workflows.w1 import RunResult, mail_tainted, run_w1, tainted

JUDGE = ("You check a teammate's output before it is passed on. Reply PASS if every "
         "number, fact and instruction in the output is supported by the teammate's "
         "input. Otherwise reply FAIL. Reply with one word.")


def make_verifier(llm: LoggedLLM):
    def verify(agent, messages, output, dep):
        user = f"Teammate's input:\n{messages[-1]['content']}\n\nTeammate's output:\n{output}"
        text, v = llm.call("verifier", [{"role": "system", "content": JUDGE},
                                        {"role": "user", "content": user}], [dep], temperature=0)
        return text.strip().upper().startswith("PASS"), v  # anything but PASS fails closed
    return verify


def recover(method: str, task: dict, world: World, backend, params: dict,
            cache: dict, faulted: RunResult) -> dict:
    if method == "restart":
        cache = {}
    elif method == "rollback":
        cache = dict(cache)
        cache.pop(world.log.payload(faulted.calls[faulted.fault["agent"]])["key"], None)
    else:
        raise ValueError(f"unknown recovery method {method}")
    sent_before = list(world.outbox)
    first = len(world.log.events)
    llm = LoggedLLM(backend, world.log, params, cache=cache)
    result = run_w1(task, llm, world)
    report = world.files.get(f"reports/{task['id']}.md", "")
    return {"method": method, "first_seq": first, "stats": llm.stats,
            "report_ok": result.score["key_fact_in_report"]
                         and not tainted(faulted.fault, task, report),
            "unexpected_recipients": result.score["unexpected_recipients"],
            # harm already done: mail sent before recovery that carries the fault
            # or went to someone other than the client
            "residual_irreversible": [m["seq"] for m in sent_before
                                      if mail_tainted(faulted.fault, task, m)
                                      or m["to"] != task["recipient"]]}
