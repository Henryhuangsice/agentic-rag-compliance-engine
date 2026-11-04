"""MCP server: exposes the knowledge base and the agent as tools for MCP clients
(Claude Desktop, Claude Code, Cursor, ...).

Run:  python -m app.mcp_server            # stdio transport
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from app.agent import app_graph, initial_state, serialize_sources
from app.retriever import get_hybrid_retriever

mcp = MCPServer(
    name="compliance-rag",
    instructions="Search and answer questions over the company's compliance policies and financial reports.",
)


@mcp.tool()
def search_compliance_docs(query: str, top_n: int = 4) -> list[dict]:
    """Hybrid search (BM25 + vector + rerank) over compliance documents; returns raw passages."""
    result = get_hybrid_retriever().search(query, top_n=top_n)
    return serialize_sources(result.documents)


@mcp.tool()
def ask_compliance_agent(question: str) -> dict:
    """Run the full Self-RAG agent and return a cited, self-checked answer."""
    state = app_graph.invoke(initial_state(question))
    return {
        "answer": state.get("generation", ""),
        "sources": serialize_sources(state.get("relevant", [])),
        "grounded": state.get("grounded"),
    }


if __name__ == "__main__":
    mcp.run()
