import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["llm_enabled"] is False
    assert body["chunks"] > 10


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Agentic RAG" in r.text


def test_query(client):
    r = client.post("/api/v1/query", json={"question": "和关联自然人的交易超过多少钱要上董事会？"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["sources"][0]["source"] == "03_related_party_transactions.md"
    assert body["trace"][0]["node"] == "retrieve"
    assert body["latency_ms"] > 0


def test_query_validation(client):
    assert client.post("/api/v1/query", json={"question": ""}).status_code == 422


def test_stream_emits_steps_then_result(client):
    events = []
    with client.stream("POST", "/api/v1/query/stream", json={"question": "身份证号码属于哪一级数据？"}) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    types = [e["type"] for e in events]
    assert types[-1] == "result"
    assert types.count("step") >= 4
    assert events[0]["step"]["node"] == "retrieve"


def test_upload_document_is_searchable(client):
    content = "# 测试制度\n\n## 第一章\n员工每年享有带薪年假 15 天，入职满五年增加至 20 天。".encode()
    r = client.post("/api/v1/documents", files={"file": ("leave.md", content, "text/markdown")})
    assert r.status_code == 200 and r.json()["chunks"] >= 1
    sources = {d["source"] for d in client.get("/api/v1/documents").json()}
    assert "leave.md" in sources
    body = client.post("/api/v1/query", json={"question": "员工带薪年假有多少天？"}).json()
    assert any(s["source"] == "leave.md" for s in body["sources"])


def test_upload_rejects_other_types(client):
    r = client.post("/api/v1/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_graph_endpoint(client):
    assert "grade_documents" in client.get("/api/v1/graph").json()["mermaid"]
