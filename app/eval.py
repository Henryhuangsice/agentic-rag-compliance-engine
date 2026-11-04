"""Benchmark: Naive RAG vs Hybrid vs Hybrid+Rerank vs Agentic Self-RAG.

    python -m app.eval                 # offline, reference-based metrics (no API key needed)
    python -m app.eval --ragas         # additionally run RAGAS LLM-judged metrics (needs LLM_API_KEY)

Offline metrics (computed against the labelled key facts in data/eval/qa_zh.jsonl):
  * Hit@k              - at least one context passed to the generator contains a key fact
  * MRR                - reciprocal rank of the first context containing a key fact
  * Context Precision  - RAGAS-style rank-weighted precision, with relevance = "contains a key fact"
  * Answer Hit         - the final answer contains every key fact
  * No-Answer Acc.     - on unanswerable questions, the system declines instead of answering
  * Latency            - mean wall-clock seconds per question (end to end)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from langchain_core.documents import Document

from app.agent import app_graph, generate_node, initial_state
from app.config import ROOT_DIR, get_settings
from app.retriever import get_hybrid_retriever

DATASET = ROOT_DIR / "data" / "eval" / "qa_zh.jsonl"
RESULTS = ROOT_DIR / "results"

MODES = {
    "naive": "Naive RAG（仅向量检索）",
    "hybrid": "Hybrid（向量 + BM25，RRF）",
    "hybrid_rerank": "Hybrid + BGE Rerank",
    "agentic": "Agentic Self-RAG（本系统）",
}


def load_dataset(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_pipeline(mode: str, question: str) -> tuple[list[Document], str, bool]:
    """Returns (contexts given to the generator, answer, declined?)."""
    if mode == "agentic":
        state = app_graph.invoke(initial_state(question))
        return state.get("relevant", []), state.get("generation", ""), state.get("mode") == "fallback"

    retriever = get_hybrid_retriever()
    if mode == "naive":
        docs = retriever.search(question, mode="vector", rerank=False).documents
    elif mode == "hybrid":
        docs = retriever.search(question, mode="hybrid", rerank=False).documents
    else:
        docs = retriever.search(question, mode="hybrid", rerank=True).documents
    out = generate_node({"question": question, "relevant": docs, "generations": 0})
    return docs, out["generation"], False


def _is_rel(doc: Document, facts: list[str]) -> bool:
    return any(f in doc.page_content for f in facts)


def context_precision(docs: list[Document], facts: list[str]) -> float:
    rels = [_is_rel(d, facts) for d in docs]
    if not any(rels):
        return 0.0
    score, hits = 0.0, 0
    for k, r in enumerate(rels, start=1):
        if r:
            hits += 1
            score += hits / k
    return score / hits


def evaluate_mode(mode: str, data: list[dict]) -> tuple[dict, list[dict]]:
    hits, mrrs, cps, answer_hits, declines, latencies = [], [], [], [], [], []
    records = []
    for item in data:
        t0 = time.perf_counter()
        docs, answer, declined = run_pipeline(mode, item["question"])
        latencies.append(time.perf_counter() - t0)
        facts = item.get("key_facts", [])
        answerable = item.get("answerable", True)
        if answerable:
            rels = [_is_rel(d, facts) for d in docs]
            hits.append(any(rels))
            mrrs.append(next((1 / k for k, r in enumerate(rels, start=1) if r), 0.0))
            cps.append(context_precision(docs, facts))
            norm = answer.replace(" ", "")
            answer_hits.append(all(f.replace(" ", "") in norm for f in facts))
        else:
            declines.append(declined)
        records.append(
            {
                "user_input": item["question"],
                "retrieved_contexts": [d.page_content for d in docs] or [""],
                "response": answer,
                "reference": item.get("reference") or "知识库中没有相关信息。",
            }
        )
    summary = {
        "mode": mode,
        "hit_at_k": statistics.mean(hits),
        "mrr": statistics.mean(mrrs),
        "context_precision": statistics.mean(cps),
        "answer_hit": statistics.mean(answer_hits),
        "no_answer_acc": statistics.mean(declines) if declines else None,
        "latency_s": statistics.mean(latencies),
        "n_answerable": len(hits),
        "n_unanswerable": len(declines),
    }
    return summary, records


def run_ragas(records_by_mode: dict[str, list[dict]]) -> dict[str, dict]:
    """LLM-judged metrics with RAGAS, using the same OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall

    s = get_settings()
    if not s.llm_enabled:
        raise SystemExit("--ragas needs LLM_API_KEY (and LLM_BASE_URL / LLM_MODEL) to be set")
    judge = LangchainLLMWrapper(
        ChatOpenAI(model=s.llm_model, base_url=s.llm_base_url, api_key=s.llm_api_key, temperature=0)
    )
    out = {}
    for mode, records in records_by_mode.items():
        result = evaluate(
            dataset=EvaluationDataset.from_list(records),
            metrics=[LLMContextPrecisionWithReference(), Faithfulness(), LLMContextRecall()],
            llm=judge,
            show_progress=False,
        )
        df = result.to_pandas()
        out[mode] = {
            col: float(df[col].mean())
            for col in ("llm_context_precision_with_reference", "faithfulness", "context_recall")
            if col in df
        }
    return out


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "–"
    return f"{v:.2f}"


def to_markdown(summaries: list[dict], ragas: dict | None, meta: dict) -> str:
    lines = [
        f"_Generated by `python -m app.eval` on {meta['date']} · "
        f"embedding `{meta['embedding']}` · reranker `{meta['reranker']}` · "
        f"generator: {meta['generator']} · {meta['n']} questions "
        f"({summaries[0]['n_answerable']} answerable, {summaries[0]['n_unanswerable']} unanswerable)._",
        "",
        "| 模式 | Hit@4 | MRR | Context Precision | Answer Hit | No-Answer Acc. | Latency (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        name = MODES[s["mode"]]
        if s["mode"] == "agentic":
            name = f"**{name}**"
        lines.append(
            f"| {name} | {_fmt(s['hit_at_k'])} | {_fmt(s['mrr'])} | {_fmt(s['context_precision'])} | "
            f"{_fmt(s['answer_hit'])} | {_fmt(s['no_answer_acc'])} | {s['latency_s']:.2f} |"
        )
    if ragas:
        lines += [
            "",
            "RAGAS (LLM-judged):",
            "",
            "| 模式 | Context Precision | Faithfulness | Context Recall |",
            "| --- | --- | --- | --- |",
        ]
        for mode, m in ragas.items():
            lines.append(
                f"| {MODES[mode]} | {_fmt(m.get('llm_context_precision_with_reference'))} | "
                f"{_fmt(m.get('faithfulness'))} | {_fmt(m.get('context_recall'))} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--ragas", action="store_true", help="also run RAGAS (requires an LLM key)")
    args = parser.parse_args()

    s = get_settings()
    data = load_dataset()
    retriever = get_hybrid_retriever()
    # warm-up so model loading does not count as latency
    run_pipeline("hybrid_rerank", data[0]["question"])

    summaries, records = [], {}
    for mode in args.modes:
        summary, recs = evaluate_mode(mode, data)
        summaries.append(summary)
        records[mode] = recs
        print(f"{mode:14s} {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()}, ensure_ascii=False)}")

    ragas = run_ragas(records) if args.ragas else None
    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "embedding": s.embedding_model if s.embedding_backend != "hash" else "hash",
        "reranker": s.reranker_model if retriever.reranker else "off",
        "generator": f"LLM `{s.llm_model}`" if s.llm_enabled else "offline extractive (no LLM)",
        "n": len(data),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.json").write_text(
        json.dumps({"meta": meta, "summary": summaries, "ragas": ragas}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = to_markdown(summaries, ragas, meta)
    (RESULTS / "benchmark.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
