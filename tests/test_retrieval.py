from app.config import get_settings
from app.loader import chunk_text, load_corpus
from app.retriever import HybridRetriever, get_hybrid_retriever, tokenize


def test_chunker_keeps_structure_and_size():
    text = "# 标题\n\n## 第一章\n" + "这是一句话。" * 200 + "\n## 第二章\n短句。"
    docs = chunk_text(text, "x.md", size=120, overlap=20)
    assert {d.metadata["section"] for d in docs} == {"第一章", "第二章"}
    assert all(d.metadata["title"] == "标题" for d in docs)
    assert all(d.page_content.startswith("【标题 / ") for d in docs)
    assert max(len(d.page_content) for d in docs) < 120 + 30


def test_corpus_loads_all_documents():
    docs = load_corpus(get_settings().docs_dir)
    originals = {f"{i:02d}" for i in range(1, 6)}
    assert originals <= {d.metadata["source"][:2] for d in docs}
    assert len({d.metadata["chunk_id"] for d in docs}) == len(docs)


def test_tokenize_drops_stopwords():
    toks = tokenize("公司的营业收入是多少？")
    assert "营业" in toks or "营业收入" in toks
    assert "的" not in toks and "多少" not in toks and "？" not in toks


def test_rrf_rewards_agreement():
    fused = HybridRetriever.rrf([[(1, 0.9), (2, 0.8)], [(2, 5.0), (3, 4.0)]], k=60)
    assert fused[0][0] == 2  # ranked by both retrievers


def test_hybrid_search_finds_the_right_policy():
    r = get_hybrid_retriever()
    res = r.search("大额交易报告 现金交易 5 万元")
    assert res.documents[0].metadata["source"] == "01_aml_kyc_policy.md"
    meta = res.documents[0].metadata
    assert meta["rrf_score"] > 0
    assert meta["vector_rank"] is not None or meta["bm25_rank"] is not None


def test_search_modes():
    r = get_hybrid_retriever()
    q = "关联交易 股东大会"
    assert all(d.metadata["bm25_rank"] is None for d in r.search(q, mode="vector").documents)
    assert all(d.metadata["vector_rank"] is None for d in r.search(q, mode="bm25").documents)
