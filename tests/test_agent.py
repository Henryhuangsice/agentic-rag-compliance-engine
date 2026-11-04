import app.agent as agent
from app.agent import app_graph, initial_state


def _nodes(state):
    return [step["node"] for step in state["trace"]]


def test_answerable_question_goes_through_generation():
    state = app_graph.invoke(initial_state("自然人客户现金交易达到多少需要报告大额交易？"))
    assert _nodes(state)[:2] == ["retrieve", "grade_documents"]
    assert "generate" in _nodes(state) and "grade_generation" in _nodes(state)
    assert state["mode"] == "offline"
    assert state["relevant"]
    assert "[1]" in state["generation"]
    assert state["groundedness"] >= 0.6


def test_unanswerable_question_rewrites_then_declines():
    state = app_graph.invoke(initial_state("火星上明天的天气适合踢足球吗"))
    assert _nodes(state) == [
        "retrieve",
        "grade_documents",
        "transform_query",
        "retrieve",
        "grade_documents",
        "fallback",
    ]
    assert state["mode"] == "fallback"
    assert state["relevant"] == []


class FakeLLM:
    """Stands in for an OpenAI-compatible model."""

    def __init__(self, grounded=True):
        self.grounded = grounded
        self.calls = []

    def chat(self, system, user):
        self.calls.append("chat")
        return "结论：需要报告。\n- 单笔或当日累计人民币现金交易 5 万元以上应报告 [1]"

    def chat_json(self, system, user):
        self.calls.append("json")
        if "相关性" in system:
            return {"relevant": [1]}
        return {"grounded": self.grounded, "score": 0.95 if self.grounded else 0.2, "reason": "ok"}


def test_llm_mode_uses_llm_for_every_step(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(agent, "get_llm", lambda: fake)
    state = app_graph.invoke(initial_state("自然人客户现金交易达到多少需要报告大额交易？"))
    assert state["mode"] == "llm"
    assert len(state["relevant"]) == 1
    assert state["grounded"] is True and state["groundedness"] == 0.95
    assert fake.calls == ["json", "chat", "json"]


def test_llm_hallucination_triggers_one_regeneration(monkeypatch):
    fake = FakeLLM(grounded=False)
    monkeypatch.setattr(agent, "get_llm", lambda: fake)
    state = app_graph.invoke(initial_state("自然人客户现金交易达到多少需要报告大额交易？"))
    assert _nodes(state).count("generate") == 2
    assert state["grounded"] is False


def test_llm_errors_fall_back_to_offline(monkeypatch):
    class Broken:
        def chat(self, *a):
            raise RuntimeError("boom")

        def chat_json(self, *a):
            raise RuntimeError("boom")

    monkeypatch.setattr(agent, "get_llm", lambda: Broken())
    state = app_graph.invoke(initial_state("高风险客户多久需要重新评估一次风险等级？"))
    assert state["mode"] == "offline"
    assert state["generation"]
