"""Test setup: no model downloads, no API keys, and a throwaway copy of the corpus."""

import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_tmp_docs = Path(tempfile.mkdtemp(prefix="rag-docs-"))
for f in (ROOT / "data" / "docs").iterdir():
    shutil.copy(f, _tmp_docs)

# Must be set before app.config.get_settings() is first called.
os.environ.update(
    {
        "EMBEDDING_BACKEND": "hash",
        "RERANKER_MODEL": "",
        "LLM_API_KEY": "",
        "DOCS_DIR": str(_tmp_docs),
    }
)
