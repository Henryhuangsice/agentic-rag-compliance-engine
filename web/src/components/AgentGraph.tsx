import type { NodeName, Step } from "@rag/shared";
import { graphProgress } from "../lib/graph";

const EDGES: Record<string, string> = {
  "START>retrieve": "M160,28 L160,47",
  "retrieve>grade_documents": "M160,82 L160,117",
  "grade_documents>transform_query": "M225,135 C300,135 318,122 318,105",
  "transform_query>retrieve": "M262,80 C248,80 244,66 226,66",
  "grade_documents>generate": "M130,152 L96,195",
  "grade_documents>fallback": "M195,152 L268,195",
  "generate>grade_generation": "M85,230 L85,265",
  "grade_generation>END": "M150,283 C185,295 175,330 168,349",
  "fallback>END": "M275,230 C275,300 245,358 208,361",
  "grade_generation>generate": "M20,283 C3,268 3,228 24,213",
};
const NODES: { name: NodeName; x: number; y: number; width: number }[] = [
  { name: "retrieve", x: 95, y: 48, width: 130 },
  { name: "transform_query", x: 262, y: 70, width: 112 },
  { name: "grade_documents", x: 95, y: 118, width: 130 },
  { name: "generate", x: 25, y: 196, width: 120 },
  { name: "fallback", x: 215, y: 196, width: 120 },
  { name: "grade_generation", x: 20, y: 266, width: 130 },
];
const LABELS: [number, number, string][] = [
  [252, 152, "不相关·可改写"],
  [88, 178, "相关"],
  [250, 176, "无可用文档"],
  [4, 252, "重试"],
  [180, 318, "通过"],
];

export function AgentGraph({ steps, running, finished }: { steps: Step[]; running: boolean; finished: boolean }) {
  const { visited, active, edges } = graphProgress(steps, running, finished);
  const nodeClass = (name: NodeName) => {
    if (!visited.has(name)) return "node";
    if (name === "fallback") return "node warn";
    return name === active ? "node active" : "node visited";
  };
  return (
    <section className="card">
      <h2>Agent 状态图</h2>
      <svg id="graph" viewBox="0 0 380 400" role="img" aria-label="LangGraph Self-RAG 状态图">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" className="arrowhead" />
          </marker>
        </defs>
        <g className="edges">
          {Object.entries(EDGES).map(([id, d]) => (
            <path key={id} data-edge={id} d={d} className={edges.has(id) ? "on" : undefined} />
          ))}
        </g>
        <g className="edge-labels">
          {LABELS.map(([x, y, text]) => <text key={text} x={x} y={y}>{text}</text>)}
        </g>
        <g className="terminal"><rect x="125" y="6" width="70" height="22" rx="11" /><text x="160" y="21">START</text></g>
        {NODES.map(({ name, x, y, width }) => (
          <g key={name} className={nodeClass(name)} data-node={name}>
            <rect x={x} y={y} width={width} height="34" rx="9" />
            <text x={x + width / 2} y={y + 21}>{name}</text>
          </g>
        ))}
        <g className="terminal"><rect x="137" y="350" width="70" height="22" rx="11" /><text x="172" y="365">END</text></g>
      </svg>
    </section>
  );
}
