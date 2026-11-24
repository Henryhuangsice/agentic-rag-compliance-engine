import type { Step } from "@rag/shared";

export function Timeline({ steps }: { steps: Step[] }) {
  return (
    <section className="card">
      <h2>执行轨迹</h2>
      <ol className="timeline" aria-live="polite">
        {steps.length === 0 && (
          <li className="placeholder">提问后，这里会逐步显示每个节点的执行结果与耗时。</li>
        )}
        {steps.map((step, i) => (
          <li key={i} className={step.node === "fallback" ? "fallback" : undefined}>
            <div className="step-head">
              <span className="step-name">{step.label}</span>
              <span className="step-ms">{step.ms.toFixed(0)} ms</span>
            </div>
            <div className="step-detail">{step.detail}</div>
          </li>
        ))}
      </ol>
    </section>
  );
}
