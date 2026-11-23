import type { QueryResult, Source } from "@rag/shared";
import { useEffect, useRef } from "react";

export type SourceTab = "cited" | "candidates";
/** `nonce` makes clicking the same citation twice scroll again. */
export interface CiteFocus {
  index: number;
  nonce: number;
}

function ScoreBar({ label, value, max }: { label: string; value: number | null | undefined; max: number }) {
  if (value == null) return <span className="score">{label} –</span>;
  const pct = Math.max(4, Math.min(100, (value / max) * 100));
  return (
    <span className="score">
      {label} <span className="bar"><span style={{ width: `${pct}%` }} /></span> {value}
    </span>
  );
}

function SourceCard({ source, relevant, focus }: { source: Source; relevant?: boolean; focus?: CiteFocus | null }) {
  const ref = useRef<HTMLDivElement>(null);
  const focused = focus?.index === source.index;
  useEffect(() => {
    if (focused) ref.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [focused, focus?.nonce]);

  const { vector_rank, bm25_rank, rrf_score, rerank_score } = source.scores;
  return (
    <div ref={ref} className={`source ${focused ? "flash" : ""}`} data-source={source.index}>
      <div className="source-head">
        <span className="source-idx">{source.index}</span>
        <span className="source-title">{source.title}</span>
        <span className="source-sec">{source.section}</span>
        {relevant !== undefined && (
          <span className={`tag ${relevant ? "yes" : "no"}`}>{relevant ? "相关" : "已过滤"}</span>
        )}
      </div>
      <p className="source-body">{source.content}</p>
      <div className="scores">
        <span className="score">向量排名 {vector_rank ?? "–"}</span>
        <span className="score">BM25 排名 {bm25_rank ?? "–"}</span>
        <ScoreBar label="RRF" value={rrf_score} max={0.034} />
        <ScoreBar label="Rerank" value={rerank_score} max={1} />
      </div>
    </div>
  );
}

interface Props {
  result: QueryResult;
  tab: SourceTab;
  onTabChange: (tab: SourceTab) => void;
  focus: CiteFocus | null;
}

export function Sources({ result, tab, onTabChange, focus }: Props) {
  const relevant = new Set(result.sources.map((s) => s.content));
  const tabs: [SourceTab, string, number][] = [
    ["cited", "引用来源", result.sources.length],
    ["candidates", "候选召回与打分", result.candidates.length],
  ];
  return (
    <section className="card">
      <div className="tabs" role="tablist">
        {tabs.map(([id, label, count]) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            className={`tab ${tab === id ? "active" : ""}`}
            onClick={() => onTabChange(id)}
          >
            {label} <span className="count">{count}</span>
          </button>
        ))}
      </div>
      <div role="tabpanel">
        {tab === "cited"
          ? result.sources.map((s) => <SourceCard key={s.index} source={s} focus={focus} />)
          : result.candidates.map((s) => (
              <SourceCard key={s.index} source={s} relevant={relevant.has(s.content)} />
            ))}
        {tab === "cited" && result.sources.length === 0 && <p className="hint">没有可引用的来源。</p>}
      </div>
    </section>
  );
}
