"""Explicit opt-in, detached workflow snapshots for context engines."""
from __future__ import annotations

import copy
import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compression_host_state_kwargs(compress_fn: Any, agent: Any) -> dict[str, Any]:
    """Do not extend a forwarding **kwargs wrapper's downstream call contract."""
    try:
        parameter = inspect.signature(compress_fn).parameters.get("host_state")
    except (TypeError, ValueError):
        return {}
    if parameter is None or parameter.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
    ):
        return {}
    return {"host_state": _snapshot(agent)}


def _snapshot(agent: Any) -> dict[str, Any]:
    """Read each source independently; never expose mutable host-owned objects."""
    state: dict[str, Any] = {"todos": [], "goal": None, "plan_path": None}
    try:
        store = getattr(agent, "_todo_store", None)
        if store is not None:
            state["todos"] = copy.deepcopy(store.read())
    except Exception:
        logger.debug("Context-engine todo snapshot unavailable", exc_info=True)
    try:
        from hermes_cli.goals import load_goal

        goal = load_goal(getattr(agent, "session_id", None))
        if goal is not None and goal.status == "active":
            state["goal"] = {
                "text": goal.goal, "status": goal.status,
                "contract": goal.contract.render_block() if goal.contract else "",
                "subgoals": copy.deepcopy(goal.subgoals),
            }
    except Exception:
        logger.debug("Context-engine goal snapshot unavailable", exc_info=True)
    return state
