"""Self-RAG agent as a LangGraph state machine.

    retrieve -> grade_documents --relevant--> generate -> grade_generation --grounded--> END
                     |                            ^              |
                     |--irrelevant, retries left--> transform_query -> retrieve
                     |                                           |--not grounded--> generate (once)
                     '--irrelevant, no retries----> fallback -> END

Every node works in two modes:
* LLM mode    - an OpenAI-compatible model grades, rewrites, generates and self-checks.
* offline mode - no API key: reranker scores grade relevance, keyword extraction rewrites
                 the query, and the answer is extracted sentence-by-sentence from sources.
"""

from __future__ import annotations

import logging
import operator
import re
import time
from functools import wraps
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.llm import get_llm
from app.loader import split_sentences
from app.retriever import get_hybrid_retriever, tokenize

log = logging.getLogger(__name__)

NODE_LABELS = {
    "retrieve": "混合检索 (FAISS + BM25 → RRF → Rerank)",
    "grade_documents": "文档相关性评估",
    "transform_query": "查询改写",
    "generate": "生成答案",
    "grade_generation": "幻觉检测 / 事实一致性",
    "fallback": "兜底：知识库无答案",
}


MAX_GENERATIONS = 2  # first attempt + one feedback-driven retry


class GraphState(TypedDict, total=False):
    question: str
    query: str
    documents: list[Document]
    relevant: list[Document]
    generation: str
    rewrites: int
    generations: int
    grounded: bool
    groundedness: float
    feedback: str
    mode: str
    trace: Annotated[list[dict[str, Any]], operator.add]


def traced(name: str) -> Callable:
    """Time a node and append a trace entry that the UI renders as a timeline."""

    def deco(fn: Callable[[GraphState], dict]) -> Callable[[GraphState], dict]:
        @wraps(fn)
        def wrapper(state: GraphState) -> dict:
            t0 = time.perf_counter()
            out = fn(state)
            detail = out.pop("_detail", "")
            out["trace"] = [
                {
                    "node": name,
                    "label": NODE_LABELS[name],
                    "detail": detail,
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                }
            ]
            return out

        return wrapper

    return deco


def _strip_prefix(text: str) -> str:
    return re.sub(r"^【[^】]*】", "", text)


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] 来源：{d.metadata.get('title')} · {d.metadata.get('section')}\n{_strip_prefix(d.page_content)}"
        for i, d in enumerate(docs, start=1)
    )


# --------------------------------------------------------------------------- nodes
@traced("retrieve")
def retrieve_node(state: GraphState) -> dict:
    query = state.get("query") or state["question"]
    result = get_hybrid_retriever().search(query)
    timing = ", ".join(f"{k.removesuffix('_ms')} {v:.0f}ms" for k, v in result.timings_ms.items())
    return {
        "query": query,
        "documents": result.documents,
        "_detail": f"查询「{query}」召回 {len(result.documents)} 个片段（{timing}）",
    }


def _keyword_overlap(query: str, text: str) -> float:
    terms = set(tokenize(query))
    if not terms:
        return 0.0
    return sum(1 for t in terms if t in text.lower()) / len(terms)


@traced("grade_documents")
def grade_documents_node(state: GraphState) -> dict:
    docs = state.get("documents", [])
    question = state["question"]
    s = get_settings()
    llm = get_llm()
    relevant: list[Document] = []
    how = ""

    if llm and docs:
        try:
            out = llm.chat_json(
                "你是严格的合规文档相关性评估员。判断每个编号片段是否包含回答问题所需的信息。"
                '输出格式：{"relevant": [相关片段编号列表]}',
                f"问题：{question}\n\n片段：\n{_format_context(docs)}",
            )
            keep = {int(i) for i in out.get("relevant", [])}
            relevant = [d for i, d in enumerate(docs, start=1) if i in keep]
            how = "LLM 判定"
        except Exception as e:  # fall back to model-free grading below
            log.warning("LLM grading failed, falling back: %s", e)
            llm = None

    if not llm:
        reranker = get_hybrid_retriever().reranker
        if docs and reranker is not None and "rerank_score" in docs[0].metadata:
            if state.get("query") and state["query"] != question:
                # After a rewrite, docs were scored against the rewritten query; relevance
                # must still be judged against what the user actually asked.
                probs = reranker.score(question, [d.page_content for d in docs])
                for d, p in zip(docs, probs):
                    d.metadata["rerank_score"] = round(p, 4)
                docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
            best = max(d.metadata["rerank_score"] for d in docs)
            cutoff = max(s.relevance_threshold, s.relative_threshold * best)
            relevant = [d for d in docs if d.metadata["rerank_score"] >= cutoff]
            how = f"Reranker 概率 ≥ {cutoff:.3f}"
        else:
            relevant = [d for d in docs if _keyword_overlap(question, d.page_content) >= 0.5]
            how = "关键词覆盖率 ≥ 0.5"

    for d in docs:
        d.metadata["relevant"] = d in relevant
    return {
        "relevant": relevant,
        "_detail": f"{how}：{len(relevant)}/{len(docs)} 个片段相关",
    }


def decide_to_generate(state: GraphState) -> str:
    if state.get("relevant"):
        return "generate"
    if state.get("rewrites", 0) < get_settings().max_rewrites:
        return "transform_query"
    return "fallback"


@traced("transform_query")
def transform_query_node(state: GraphState) -> dict:
    question = state["question"]
    llm = get_llm()
    new_query = ""
    if llm:
        try:
            new_query = llm.chat(
                "你是检索查询优化专家。把用户问题改写成更适合在企业合规制度、财报文档中检索的查询，"
                "补充可能出现在原文中的专业术语。只输出改写后的查询。",
                f"原问题：{question}\n上一次查询：{state.get('query')}",
            ).strip()
        except Exception as e:
            log.warning("LLM rewrite failed: %s", e)
    if not new_query:
        new_query = " ".join(dict.fromkeys(tokenize(question))) or question
    return {
        "query": new_query,
        "rewrites": state.get("rewrites", 0) + 1,
        "_detail": f"「{state.get('query')}」→「{new_query}」",
    }


def _extractive_answer(question: str, docs: list[Document]) -> str:
    """Offline mode: pick the source sentences that best answer the question."""
    retriever = get_hybrid_retriever()
    candidates: list[tuple[int, str]] = []
    for i, d in enumerate(docs, start=1):
        for sent in split_sentences(_strip_prefix(d.page_content)):
            if len(sent) >= 8:
                candidates.append((i, sent))
    if not candidates:
        return ""
    texts = [c[1] for c in candidates]
    if retriever.reranker is not None:
        scores = retriever.reranker.score(question, texts)
    else:
        q = retriever.embedder.embed_query(question)
        scores = list(retriever.embedder.embed(texts) @ q)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    best = ranked[0][0]
    picked = [c for sc, c in ranked[:3] if sc >= best * 0.5][:3]
    lines = [f"- {sent} [{i}]" for i, sent in picked]
    return "根据知识库原文（离线抽取模式，未配置 LLM）：\n" + "\n".join(lines)


_CITE_VARIANTS = re.compile(r"[\[【［]\s*(?:编号|来源|资料|出处|ref)?\s*[:：]?\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*[\]】］]", re.I)


def normalize_citations(text: str) -> str:
    """Models write citations as [编号：1], 【1】, [1, 2]...; normalize to [1][2]."""
    return _CITE_VARIANTS.sub(
        lambda m: "".join(f"[{n}]" for n in re.split(r"\s*[,，、]\s*", m.group(1))), text
    )


@traced("generate")
def generate_node(state: GraphState) -> dict:
    docs = state["relevant"]
    question = state["question"]
    llm = get_llm()
    strict = state.get("generations", 0) > 0
    answer = ""
    if llm:
        system = (
            "你是企业合规与财报分析助手。只能依据给定的编号资料回答，"
            "每个事实性陈述的句末用方括号数字标注出处，例如：“交易金额 30 万元以上须提交董事会审议 [1]。”；"
            "资料不足时明确说明“资料未提及”。回答使用简体中文，先给结论，再列要点。"
        )
        user = f"资料：\n{_format_context(docs)}\n\n问题：{question}"
        if strict:
            system += "上一版答案没有通过事实核查。请逐句对照资料重写，不要做任何推断，结论句必须与要点一致。"
            user += (
                f"\n\n上一版答案：\n{state.get('generation', '')}"
                f"\n\n核查发现的问题：{state.get('feedback') or '存在资料不支持的陈述'}"
            )
        try:
            answer = normalize_citations(llm.chat(system, user))
        except Exception as e:
            log.warning("LLM generation failed, using extractive answer: %s", e)
    mode = "llm" if answer else "offline"
    if not answer:
        answer = _extractive_answer(question, docs)
    return {
        "generation": answer,
        "generations": state.get("generations", 0) + 1,
        "mode": mode,
        "_detail": f"{'LLM 生成' if mode == 'llm' else '抽取式生成'}，引用 {len(docs)} 个片段",
    }


def _support_ratio(answer: str, docs: list[Document]) -> float:
    """Share of answer sentences whose character bigrams mostly appear in the sources."""
    source = "".join(_strip_prefix(d.page_content) for d in docs)
    grams = {source[i : i + 2] for i in range(len(source) - 1)}
    sents = [
        re.sub(r"\[\d+\]|^[-*\d.、\s]+", "", s)
        for s in split_sentences(answer)
        if not s.startswith("根据知识库原文")
    ]
    sents = [s for s in sents if len(s) >= 6]
    if not sents:
        return 0.0
    supported = 0
    for s in sents:
        sg = [s[i : i + 2] for i in range(len(s) - 1)]
        if sg and sum(g in grams for g in sg) / len(sg) >= 0.6:
            supported += 1
    return supported / len(sents)


@traced("grade_generation")
def grade_generation_node(state: GraphState) -> dict:
    docs, answer = state["relevant"], state["generation"]
    ratio = _support_ratio(answer, docs)
    grounded = ratio >= 0.6
    how = f"字符级支撑率 {ratio:.2f}"
    feedback = ""
    llm = get_llm()
    if llm and state.get("mode") == "llm":
        try:
            out = llm.chat_json(
                "你是严格的事实核查员。逐条检查答案中的每一个陈述（包括开头的结论句）：\n"
                "1. 陈述中的数字、期限、金额、主体是否与资料原文完全一致，且适用于问题所问的对象；\n"
                "2. 答案内部是否前后矛盾（例如结论与要点给出不同的数字）。\n"
                "只要有一个陈述不被资料支持，或答案前后矛盾，grounded 就必须为 false。\n"
                '输出格式：{"claims": [{"text": "陈述", "supported": true或false}], '
                '"grounded": true或false, "score": 被支持陈述的比例(0到1), "reason": "简短理由"}',
                f"问题：{state['question']}\n\n资料：\n{_format_context(docs)}\n\n答案：\n{answer}",
            )
            grounded = bool(out.get("grounded"))
            ratio = float(out.get("score", ratio))
            if ratio > 1:  # some models answer in percent
                ratio /= 100
            ratio = min(max(ratio, 0.0), 1.0)
            how = f"LLM 核查 score={ratio:.2f}：{out.get('reason', '')}"
            feedback = str(out.get("reason", ""))
        except Exception as e:
            log.warning("LLM hallucination check failed: %s", e)
    result = {
        "grounded": grounded,
        "groundedness": round(ratio, 3),
        "feedback": feedback,
        "_detail": f"{'通过' if grounded else '未通过'}（{how}）",
    }
    if not grounded and state.get("generations", 0) >= MAX_GENERATIONS:
        # Never deliver an answer that failed verification twice: fall back to
        # verbatim source sentences, which are grounded by construction.
        extract = _extractive_answer(state["question"], docs)
        result.update(
            generation="⚠️ 模型生成的答案两次未通过事实核查，以下改为直接摘录资料原文：\n"
            + extract.split("\n", 1)[-1],
            grounded=True,
            groundedness=round(_support_ratio(extract, docs), 3),
            mode="extractive_fallback",
        )
        result["_detail"] += "；已改用原文摘录作答"
    return result


def decide_after_check(state: GraphState) -> str:
    if state.get("grounded") or state.get("generations", 0) >= MAX_GENERATIONS:
        return "end"
    return "regenerate"


@traced("fallback")
def fallback_node(state: GraphState) -> dict:
    return {
        "generation": (
            "知识库中未检索到能够回答该问题的内容，为避免编造，本次不作答。"
            "建议补充相关制度文件，或联系合规与法务部人工确认。"
        ),
        "relevant": [],
        "grounded": True,
        "groundedness": None,
        "mode": "fallback",
        "_detail": f"经过 {state.get('rewrites', 0)} 次查询改写仍无相关文档",
    }


# --------------------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade_documents", grade_documents_node)
    g.add_node("transform_query", transform_query_node)
    g.add_node("generate", generate_node)
    g.add_node("grade_generation", grade_generation_node)
    g.add_node("fallback", fallback_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "transform_query": "transform_query", "fallback": "fallback"},
    )
    g.add_edge("transform_query", "retrieve")
    g.add_edge("generate", "grade_generation")
    g.add_conditional_edges(
        "grade_generation", decide_after_check, {"end": END, "regenerate": "generate"}
    )
    g.add_edge("fallback", END)
    return g.compile()


app_graph = build_graph()


def initial_state(question: str) -> GraphState:
    return {"question": question, "query": question, "rewrites": 0, "generations": 0, "trace": []}


def serialize_sources(docs: list[Document]) -> list[dict]:
    return [
        {
            "index": i,
            "source": d.metadata.get("source"),
            "title": d.metadata.get("title"),
            "section": d.metadata.get("section"),
            "content": _strip_prefix(d.page_content),
            "scores": {
                k: d.metadata.get(k)
                for k in ("vector_rank", "bm25_rank", "rrf_score", "rerank_score")
            },
        }
        for i, d in enumerate(docs, start=1)
    ]
