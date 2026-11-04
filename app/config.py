"""Runtime configuration, read from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Corpus ---------------------------------------------------------
    docs_dir: Path = ROOT_DIR / "data" / "docs"
    chunk_size: int = 360
    chunk_overlap: int = 60

    # --- Retrieval ------------------------------------------------------
    # "fastembed" = real BGE model (ONNX, CPU); "hash" = dependency-free
    # character n-gram vectors, used for tests / CI.
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    # Cross-encoder reranker. Empty string disables reranking.
    reranker_model: str = "BAAI/bge-reranker-base"
    model_cache_dir: Path | None = None
    # Where to fetch the reranker from: "huggingface" (honours HF_ENDPOINT) or
    # "modelscope" (much faster inside mainland China).
    model_source: str = "huggingface"

    retrieve_k: int = 10          # candidates taken from each retriever
    rrf_k: int = 60               # Reciprocal Rank Fusion constant
    rerank_top_n: int = 4         # documents kept after reranking
    # Offline relevance grading with the reranker: a passage counts as relevant if its
    # probability is >= relevance_threshold AND >= relative_threshold * best score.
    # Calibrated on data/eval (lowest answerable top score ~0.056, highest unanswerable ~0.018).
    relevance_threshold: float = 0.03
    relative_threshold: float = 0.1
    max_rewrites: int = 1         # Self-RAG query-rewrite loops

    # --- LLM (any OpenAI-compatible endpoint) ----------------------------
    # DeepSeek: https://api.deepseek.com  / deepseek-chat or deepseek-reasoner
    # Qwen (DashScope): https://dashscope.aliyuncs.com/compatible-mode/v1 / qwen-plus
    # Ollama: http://localhost:11434/v1 / qwen2.5:7b  (api key: "ollama")
    # Leave LLM_API_KEY empty to run in offline extractive mode.
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.0
    llm_timeout: float = 60.0

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
