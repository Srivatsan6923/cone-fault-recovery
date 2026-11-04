# CONE

Fault propagation and recovery in multi-agent workflows. Phase 1 project for UC San Diego CSE 291A.

The project studies what happens after an agent produces a faulty result. Which later steps use it? What needs to be rerun? What happens if the workflow has already sent an email?

The main experiment uses three agents to extract a fact, prepare a report, and send it to a simulated client. It compares full restart, rollback with cached results, and verification at each hand-off. A second experiment uses the direct prompt injection and Byzantine scenarios from TAMAS.

## Setup

Requires Python 3.12 or later. Run the commands below from the repository root.

```bash
python -m pip install -e . pytest
python -m pytest -q tests
```

## Reproduce the results

The saved traces are sufficient to regenerate the results. This step runs on a CPU.

Extract the trace archives using Bash, then generate the tables and figure data.

```bash
mkdir -p runs

for f in w1_eval w1_verify tamas tamas_judge tamas_judge2; do
    tar -xzf "traces/$f.tar.gz" -C runs
done

python -m cone.report_tables --out results
```

Outputs are written to `results/`. The archives contain the controlled workflow runs, verification runs, TAMAS runs, and two passes of Byzantine judge labels.

Logging overhead is measured again with a stub model, so that measurement will vary with the machine.

## Run an experiment

Model inference requires an OpenAI-compatible server. The experiments used Qwen2.5-7B-Instruct, quantized as `q4_k_m`, served by llama.cpp on a laptop RTX 4060.

The configuration is in `configs/laptop-qwen7b.json`, including the model and runtime hashes and sampling settings. To use a different server, copy this file and update `base_url` and `model`.

This command runs one held-out task with a stale output from the reporting agent, then applies rollback.

```bash
python -m cone.run_w1 \
    --config configs/laptop-qwen7b.json \
    --tasks data/w1_tasks.json \
    --split eval \
    --limit 1 \
    --seeds 0 \
    --faults stale:writer \
    --recover rollback \
    --out runs/demo
```

The original run is saved under `runs/demo/w1-010/stale-writer/s0/`. The recovered run is saved in the adjacent `s0-rollback/` directory.

A saved example is also available in `traces/example/`. In that example, the stale output causes an email to be sent to the previous task's client. Rollback fixes the report and sends it to the intended client, while the earlier delivery remains in the log.

Each run contains an `events.jsonl` event log, payloads in `blobs/`, and scores and run settings in `meta.json`.

For the full controlled experiment, use seeds `0 1 2` and the evaluation split. Faults are specified as `TYPE:AGENT`, where `TYPE` is `wrong_fact`, `timeout`, `stale` or `injection`, and `AGENT` is `researcher`, `analyst` or `writer` (the extraction, answering and reporting stages). Use `none` for the clean condition, `--recover restart rollback` for recovery comparisons, and `--verify` for the verification baseline.

### TAMAS

```bash
python -m cone.run_tamas \
    --config configs/laptop-qwen7b.json \
    --data data/tamas \
    --seeds 0 1 2 \
    --out runs/tamas

python -m cone.judge_tamas \
    --runs runs/tamas \
    --data data/tamas \
    --out runs/tamas_judge
```

The Byzantine judge uses gpt-oss-120b through TritonGPT and requires `TRITONGPT_TOKEN`.

Completed runs are skipped when a command is rerun.

## Experiments and scoring

The controlled workload contains 30 tasks about fictional entities. Each asks for a number and provides four source sentences containing the answer and three distractors. Ten tasks are used for development. Evaluation uses the remaining 20 tasks with three seeds, giving 60 runs per condition.

Faults replace or extend an agent's output before the next stage receives it. The stale condition uses the previous task's output, wrapping from the first evaluation task to the last.

Correctness is scored by the presence of the expected number in the saved report. Unauthorized delivery is scored from the email tool's recipient field. An address appearing only in the email body does not count as a recipient.

Propagation is measured by looking for the fault marker in later outputs. For timeouts, the check is whether the expected number is missing. A faulted run is included only when its clean counterpart succeeded with the same task and seed.

All recovery methods receive the identity of the faulty agent.

- **Restart** reruns all calls with an empty cache.
- **Rollback** reruns from the faulty agent and reuses cached calls when their prompts are unchanged.
- **Always-verify** checks each hand-off with the same model and regenerates a rejected output once.

Emails sent before recovery remain in the simulated world and count as residual effects.

For TAMAS, each scenario and seed has three conditions sharing a cache. These are clean, attacked, and placebo, where the placebo introduces a harmless edit at the injection site. DPI success is determined from tool calls. Byzantine success uses TAMAS's judge prompt, with label `0` indicating that the task was derailed.

## Limitations

The controlled experiment uses one 7B model and a linear workflow with simulated tools. Marker checks can miss paraphrased faults, and the correctness check only tests whether the expected number appears. Byzantine outcomes depend on an LLM judge.

Phase 2 will examine branching workflows, shared state, and selective recovery.

## Data and license

The TAMAS data comes from [microsoft/TAMAS](https://github.com/microsoft/TAMAS). Its licenses are included in `data/tamas/LICENSE` and `data/tamas/LICENSE.CDLA-2.0`.

The tool files are parsed with Python's `ast` module to extract docstrings and reply text.

The project code is released under the [MIT license](LICENSE).
