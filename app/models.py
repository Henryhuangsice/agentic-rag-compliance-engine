"""Embedding and reranking backends (ONNX via fastembed, or a test-only hash backend)."""

from __future__ import annotations

import hashlib
import logging
import math
import shutil
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

from app.config import Settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    """BGE embeddings via fastembed (ONNX Runtime, no PyTorch needed)."""

    # BGE-zh recommends this instruction prefix for short queries.
    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.prefix = self.QUERY_PREFIX if "zh" in model_name else ""

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self.model.embed(texts)), dtype="float32"))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([self.prefix + text])[0]


class HashEmbedder:
    """Deterministic character n-gram hashing vectors. No downloads; for tests/CI."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        text = text.lower()
        for n in (1, 2, 3):
            for i in range(len(text) - n + 1):
                h = int(hashlib.md5(text[i : i + n].encode()).hexdigest()[:8], 16)
                v[h % self.dim] += 1.0
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


MODELSCOPE_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "onnx/model.onnx",
]


def download_from_modelscope(repo: str, target: Path) -> Path:
    """Fetch an ONNX model from ModelScope (fast inside mainland China), with resume."""
    for rel in MODELSCOPE_FILES:
        dest = target / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        have = part.stat().st_size if part.exists() else 0
        url = f"https://www.modelscope.cn/models/{repo}/resolve/master/{rel}"
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        log.info("Downloading %s from ModelScope", url)
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab" if have else "wb") as f:
            if have and resp.status != 206:  # server ignored Range; restart
                f.truncate(0)
            shutil.copyfileobj(resp, f, length=1 << 20)
        part.rename(dest)
    return target


class CrossEncoderReranker:
    """BGE cross-encoder reranker; returns relevance probabilities in [0, 1]."""

    def __init__(self, model_name: str, cache_dir: str | None = None, source: str = "huggingface"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        kwargs = {}
        if source == "modelscope":
            root = Path(cache_dir or Path.home() / ".cache" / "fastembed") / "modelscope"
            local = download_from_modelscope(model_name, root / model_name.replace("/", "--"))
            kwargs["specific_model_path"] = str(local)
        self.model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir, **kwargs)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        logits = list(self.model.rerank(query, passages))
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    cache = str(settings.model_cache_dir) if settings.model_cache_dir else None
    return FastEmbedEmbedder(settings.embedding_model, cache)


def build_reranker(settings: Settings) -> CrossEncoderReranker | None:
    if not settings.reranker_model or settings.embedding_backend == "hash":
        return None
    cache = str(settings.model_cache_dir) if settings.model_cache_dir else None
    return CrossEncoderReranker(settings.reranker_model, cache, settings.model_source)
