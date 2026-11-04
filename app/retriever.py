"""Hybrid retrieval: FAISS dense search + BM25 sparse search, fused with RRF,
then optionally reranked by a BGE cross-encoder."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass

import faiss
import jieba
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.loader import load_corpus
from app.models import CrossEncoderReranker, Embedder, build_embedder, build_reranker

jieba.setLogLevel(logging.WARNING)
log = logging.getLogger(__name__)

STOPWORDS = set(
    "的 了 是 在 和 与 或 及 等 对 为 由 被 把 将 有 要 需要 什么 多少 怎样 怎么 如何 哪个 哪些 "
    "是否 能 能不能 可以 吗 呢 吧 啊 公司 一 个 这 那 其 之 以 于 上 下 中 多久 几 谁 "
    "the a an of to is are what how which who when".split()
)
_TOKEN_OK = re.compile(r"[\w一-鿿]")


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in jieba.cut_for_search(text.lower()):
        tok = tok.strip()
        if tok and tok not in STOPWORDS and _TOKEN_OK.search(tok):
            tokens.append(tok)
    return tokens


@dataclass
class SearchResult:
    documents: list[Document]
    timings_ms: dict[str, float]


class HybridRetriever:
    def __init__(
        self,
        documents: list[Document],
        embedder: Embedder,
        reranker: CrossEncoderReranker | None,
        settings: Settings,
    ):
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker
        self.documents = documents
        self._build()

    def _build(self) -> None:
        texts = [d.page_content for d in self.documents]
        if texts:
            vectors = self.embedder.embed(texts)
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
            self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        else:
            self.index = None
            self.bm25 = None

    # -- individual retrievers --------------------------------------------
    def vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.index is None:
            return []
        q = self.embedder.embed_query(query).reshape(1, -1).astype("float32")
        scores, ids = self.index.search(q, min(k, len(self.documents)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    @staticmethod
    def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d))."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # -- main entry point ---------------------------------------------------
    def search(
        self,
        query: str,
        mode: str = "hybrid",
        rerank: bool = True,
        top_n: int | None = None,
    ) -> SearchResult:
        s = self.settings
        top_n = top_n or s.rerank_top_n
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        dense = self.vector_search(query, s.retrieve_k) if mode in ("hybrid", "vector") else []
        timings["vector_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        sparse = self.bm25_search(query, s.retrieve_k) if mode in ("hybrid", "bm25") else []
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000

        dense_rank = {i: r for r, (i, _) in enumerate(dense, start=1)}
        sparse_rank = {i: r for r, (i, _) in enumerate(sparse, start=1)}
        fused = self.rrf([r for r in (dense, sparse) if r], s.rrf_k)

        candidates: list[Document] = []
        for doc_id, score in fused:
            src = self.documents[doc_id]
            candidates.append(
                Document(
                    page_content=src.page_content,
                    metadata={
                        **src.metadata,
                        "vector_rank": dense_rank.get(doc_id),
                        "bm25_rank": sparse_rank.get(doc_id),
                        "rrf_score": round(score, 5),
                    },
                )
            )

        if rerank and self.reranker is not None and candidates:
            t0 = time.perf_counter()
            probs = self.reranker.score(query, [d.page_content for d in candidates])
            for d, p in zip(candidates, probs):
                d.metadata["rerank_score"] = round(p, 4)
            candidates.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

        return SearchResult(candidates[:top_n], {k: round(v, 1) for k, v in timings.items()})

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.search(query).documents


# -- process-wide singleton ---------------------------------------------------
_lock = threading.Lock()
_retriever: HybridRetriever | None = None


def build_retriever(settings: Settings | None = None) -> HybridRetriever:
    settings = settings or get_settings()
    t0 = time.perf_counter()
    docs = load_corpus(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
    retriever = HybridRetriever(docs, build_embedder(settings), build_reranker(settings), settings)
    log.info("Indexed %d chunks in %.1fs", len(docs), time.perf_counter() - t0)
    return retriever


def get_hybrid_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = build_retriever()
    return _retriever


def reindex() -> HybridRetriever:
    """Rebuild the index from disk (after uploads), reusing the loaded models."""
    global _retriever
    with _lock:
        settings = get_settings()
        docs = load_corpus(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
        if _retriever is None:
            _retriever = HybridRetriever(
                docs, build_embedder(settings), build_reranker(settings), settings
            )
        else:
            _retriever = HybridRetriever(docs, _retriever.embedder, _retriever.reranker, settings)
    return _retriever


def set_retriever(retriever: HybridRetriever | None) -> None:
    """Test hook."""
    global _retriever
    _retriever = retriever
