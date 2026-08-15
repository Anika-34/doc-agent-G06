"""FIXED — structured logging (auditable NFR). Use get_logger(), never print()."""
from __future__ import annotations
import logging
import sys
import itertools
from pathlib import Path

from .contracts import TraceStep  # adjust import path to match where contracts.py actually lives


def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            '{"ts":"%(asctime)s","lvl":"%(levelname)s","mod":"%(name)s","msg":"%(message)s"}'
        ))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


_TRACE_PATH = Path("traces/run.jsonl")
_step_counter = itertools.count(1)  # monotonic step id across one process run


def _write_trace_step(step: TraceStep) -> None:
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(step.model_dump_json() + "\n")


def register(hooks) -> None:
    """Wire structured tracing at each seam (auditable trail) AND emit traces/run.jsonl.
    Each seam appends a contracts.TraceStep line to traces/run.jsonl so the A3 agentic-feature
    check can read the trajectory (path must depend on observations)."""
    logger = get_logger("trace")

    def _trace_step(ctx: dict) -> dict:
        # ON_STEP fires once per agent decision -- tool="decide", args=whatever the
        # agent chose to do next, obs=what decide() saw when making that choice
        # (e.g. retrieval scores) -- this is the "path depends on observations" signal.
        step = TraceStep(
            step=next(_step_counter),
            tool="decide",
            args=ctx.get("args", ctx.get("action", {})),
            obs=ctx.get("obs", ctx.get("observation", {})),
        )
        _write_trace_step(step)
        logger.info(f"trace step {step.step}: tool=decide")
        return ctx

    def _trace_tool_call(ctx: dict) -> dict:
        # ON_TOOL_CALL fires once per actual tool invocation (e.g. retrieve) --
        # tool = real tool name, args = the actual args used (e.g. the query
        # actually sent to retrieval), obs = what came back / was observed after.
        step = TraceStep(
            step=next(_step_counter),
            tool=ctx.get("tool", ctx.get("tool_name", "unknown_tool")),
            args=ctx.get("args", {}),
            obs=ctx.get("obs", ctx.get("result", {})),
        )
        _write_trace_step(step)
        logger.info(f"trace step {step.step}: tool={step.tool}")
        return ctx

    def _trace_answer(ctx: dict) -> dict:
        # AFTER_ANSWER -- final step, tool="answer"
        step = TraceStep(
            step=next(_step_counter),
            tool="answer",
            args=ctx.get("args", {}),
            obs=ctx.get("obs", {"answer": ctx.get("answer")}),
        )
        _write_trace_step(step)
        logger.info(f"trace step {step.step}: tool=answer")
        return ctx

    hooks.register(hooks.ON_STEP, _trace_step)
    hooks.register(hooks.ON_TOOL_CALL, _trace_tool_call)
    hooks.register(hooks.AFTER_ANSWER, _trace_answer)