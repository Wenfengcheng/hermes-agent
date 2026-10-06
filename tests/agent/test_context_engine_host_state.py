"""Opt-in structured state at engine compression boundaries (#133644)."""
from types import SimpleNamespace

import pytest

from agent.conversation_compression import _resolve_compress_call
from hermes_cli import goals
from tools.todo_tool import TodoStore


def dispatch(agent):
    fn, kwargs = _resolve_compress_call(
        agent, approx_tokens=123, focus_topic="focus", force=True,
        memory_context="memory", bypass_cooldown=False,
    )
    return fn([{"role": "user", "content": "input"}], **kwargs)


class StateEngine:
    def compress(self, messages, current_tokens=None, *, host_state=None):
        return host_state


@pytest.fixture
def agent():
    store = TodoStore()
    store.write([{"id": "a", "content": "verify result", "status": "pending"}])
    return SimpleNamespace(session_id="selected-session", _todo_store=store,
                           context_compressor=StateEngine())


def test_compression_receives_current_state_without_mutating_host(agent):
    goal = goals.GoalState("ship result", subgoals=["run checks"],
                           contract=goals.GoalContract(verification="tests green", constraints="offline"))
    goals.save_goal(agent.session_id, goal)
    goals.save_goal("other-session", goals.GoalState("must not leak"))
    assert goals.load_goal(agent.session_id).goal == goal.goal
    state = dispatch(agent)
    assert state is not None, "explicit host_state engine received no session state"
    assert state["todos"] == agent._todo_store.read()
    assert state["goal"] == {"text": goal.goal, "status": "active",
                             "contract": goal.contract.render_block(), "subgoals": goal.subgoals}
    assert state["plan_path"] is None
    state["todos"][0]["content"] = "engine mutation"
    state["goal"]["subgoals"].append("engine mutation")
    assert agent._todo_store.read()[0]["content"] == "verify result"
    assert goals.load_goal(agent.session_id).subgoals == ["run checks"]
    assert dispatch(agent)["goal"]["subgoals"] == ["run checks"]


@pytest.mark.parametrize("status", ["paused", "done", "cleared"])
def test_inactive_goal_not_offered_as_active_work(agent, status):
    goals.save_goal(agent.session_id, goals.GoalState("old", status=status))
    state = dispatch(agent)
    assert state is not None
    assert state["goal"] is None


def test_legacy_engine_call_is_unchanged(agent, monkeypatch):
    def no_read(*args):
        raise AssertionError("legacy engines must not cause goal DB access")
    monkeypatch.setattr(goals, "load_goal", no_read)
    class Legacy:
        def compress(self, messages, current_tokens=None, focus_topic=None):
            return messages, current_tokens, focus_topic
    agent.context_compressor = Legacy()
    assert dispatch(agent) == ([{"role": "user", "content": "input"}], 123, "focus")


def test_forwarding_wrapper_does_not_receive_host_state(agent):
    class Wrapper:
        def compress(self, messages, **kwargs):
            def strict(messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
                return messages, current_tokens, focus_topic, force, memory_context
            return strict(messages, **kwargs)
    agent.context_compressor = Wrapper()
    assert dispatch(agent) == ([{"role": "user", "content": "input"}], 123, "focus", True, "memory")


def test_empty_state(agent):
    agent._todo_store = None
    agent.session_id = None
    assert dispatch(agent) == {"todos": [], "goal": None, "plan_path": None}


def test_failed_goal_read_preserves_todos(agent, monkeypatch):
    def unavailable(*args):
        raise RuntimeError("DB unavailable")
    monkeypatch.setattr(goals, "load_goal", unavailable)
    assert dispatch(agent) == {"todos": agent._todo_store.read(), "goal": None, "plan_path": None}


def test_failed_todo_read_preserves_goal(agent):
    class BrokenStore:
        def read(self):
            raise RuntimeError("store unavailable")
    goals.save_goal(agent.session_id, goals.GoalState("goal remains"))
    agent._todo_store = BrokenStore()
    state = dispatch(agent)
    assert state["todos"] == []
    assert state["goal"]["text"] == "goal remains"


def test_engine_exception_is_not_retried(agent):
    class BrokenEngine:
        calls = 0
        def compress(self, messages, current_tokens=None, *, host_state=None):
            self.calls += 1
            raise TypeError("engine failure")
    agent.context_compressor = BrokenEngine()
    with pytest.raises(TypeError, match="engine failure"):
        dispatch(agent)
    assert agent.context_compressor.calls == 1


def test_builtin_does_not_opt_in(agent, monkeypatch):
    from agent.context_compressor import ContextCompressor
    from agent.context_engine_host_state import compression_host_state_kwargs
    def no_read(*args):
        raise AssertionError("built-in must not read host state")
    monkeypatch.setattr(goals, "load_goal", no_read)
    assert compression_host_state_kwargs(ContextCompressor.compress, agent) == {}


def test_positional_only_name_is_not_keyword_opt_in(agent):
    class Positional:
        def compress(self, messages, host_state=None, /, current_tokens=None):
            return host_state
    agent.context_compressor = Positional()
    assert dispatch(agent) is None

