import type { Health } from "@rag/shared";

export function Header({ health }: { health: Health | "loading" | "down" }) {
  return (
    <header className="topbar">
      <div className="brand">
        <span className="logo" aria-hidden="true">
          <svg viewBox="0 0 32 32"><rect width="32" height="32" rx="8" /><path d="M9 17l5 5 9-11" /></svg>
        </span>
        <div>
          <h1>Agentic RAG 合规分析引擎</h1>
          <p className="sub">LangGraph Self-RAG · FAISS + BM25 混合检索 · BGE Reranker</p>
        </div>
      </div>
      <div className="status">
        {health === "loading" && <span className="pill">连接中…</span>}
        {health === "down" && <span className="pill bad"><span className="dot" />后端不可用</span>}
        {typeof health === "object" && (
          <>
            {health.llm_enabled ? (
              <span className="pill ok"><span className="dot" />LLM · {health.llm_model}</span>
            ) : (
              <span className="pill warn" title="未配置 LLM_API_KEY：答案由原文抽取生成"><span className="dot" />离线抽取模式</span>
            )}
            <span className="pill">Embedding · {health.embedding_model}</span>
            <span className="pill">Reranker · {health.reranker_model ?? "off"}</span>
            <span className="pill">{health.chunks} chunks</span>
          </>
        )}
      </div>
    </header>
  );
}
