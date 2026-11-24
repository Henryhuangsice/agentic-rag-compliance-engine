import type { QueryResult } from "@rag/shared";
import { parseAnswer, type Inline } from "../lib/answer";

const MODES: Record<QueryResult["mode"], { label: string; tone: string }> = {
  llm: { label: "LLM 生成", tone: "ok" },
  offline: { label: "抽取式", tone: "warn" },
  fallback: { label: "已拒答", tone: "warn" },
  extractive_fallback: { label: "核查未通过 · 原文摘录", tone: "warn" },
};

interface Props {
  result: QueryResult;
  onCite: (index: number) => void;
}

export function Answer({ result, onCite }: Props) {
  const render = (parts: Inline[]) =>
    parts.map((part, i) => {
      if (part.type === "text") return part.text;
      if (part.type === "bold") return <b key={i}>{part.text}</b>;
      // A model can cite a source that does not exist; show it, but not as a link.
      if (!result.sources.some((s) => s.index === part.index)) return `[${part.index}]`;
      return (
        <button key={i} type="button" className="cite" aria-label={`查看来源 ${part.index}`} onClick={() => onCite(part.index)}>
          {part.index}
        </button>
      );
    });
  const mode = MODES[result.mode];
  return (
    <article className="card answer">
      <div className="answer-head">
        <h2>回答</h2>
        <div className="metrics">
          <span className="pill">{(result.latency_ms / 1000).toFixed(2)} s</span>
          <span className={`pill ${mode.tone}`}>{mode.label}</span>
          {result.groundedness !== null && (
            <span className={`pill ${result.grounded ? "ok" : "bad"}`} title="答案句子被来源支撑的比例">
              Groundedness {result.groundedness.toFixed(2)}
            </span>
          )}
        </div>
      </div>
      <div className={`answer-body ${result.mode === "fallback" ? "fallback-note" : ""}`}>
        {parseAnswer(result.answer).map((block, i) =>
          block.type === "p" ? (
            <p key={i}>{render(block.inline)}</p>
          ) : (
            <ul key={i}>{block.items.map((item, j) => <li key={j}>{render(item)}</li>)}</ul>
          ),
        )}
      </div>
      {result.rewritten_query && (
        <div className="rewrite">首轮检索不相关，Agent 将查询改写为：「{result.rewritten_query}」</div>
      )}
    </article>
  );
}
