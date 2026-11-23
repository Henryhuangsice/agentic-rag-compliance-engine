import { useEffect, useState } from "react";
import { apiKey } from "./api/client";
import { AgentGraph } from "./components/AgentGraph";
import { Answer } from "./components/Answer";
import { AskForm } from "./components/AskForm";
import { Header } from "./components/Header";
import { KnowledgeBase } from "./components/KnowledgeBase";
import { ApiKeyNotice, CancelledNotice, ErrorNotice } from "./components/Notices";
import { Sources, type CiteFocus, type SourceTab } from "./components/Sources";
import { Timeline } from "./components/Timeline";
import { useAgentQuery } from "./hooks/useAgentQuery";
import { useKnowledgeBase } from "./hooks/useKnowledgeBase";

const initialQuestion = () => new URLSearchParams(location.search).get("q")?.trim() ?? "";

export function App() {
  const { state, ask, cancel } = useAgentQuery();
  const kb = useKnowledgeBase();
  const [question, setQuestion] = useState(initialQuestion);
  const [tab, setTab] = useState<SourceTab>("cited");
  const [focus, setFocus] = useState<CiteFocus | null>(null);

  const submit = (q: string) => {
    setQuestion(q);
    setTab("cited");
    setFocus(null);
    history.replaceState(null, "", `?q=${encodeURIComponent(q)}`);
    void ask(q);
  };

  // Shareable links: /?q=问题 runs the question on load.
  useEffect(() => {
    const q = initialQuestion();
    if (q) void ask(q);
  }, [ask]);

  const cite = (index: number) => {
    setTab("cited");
    setFocus((f) => ({ index, nonce: (f?.nonce ?? 0) + 1 }));
  };

  const queryUnauthorized = state.status === "failed" && state.error.code === "unauthorized";
  const needsKey = queryUnauthorized || kb.error?.code === "unauthorized";
  const saveKey = (key: string) => {
    apiKey.set(key);
    void kb.refresh();
    if (queryUnauthorized) submit(state.question);
  };

  return (
    <>
      <Header health={kb.health} />
      <main className="layout">
        <div className="col-main">
          {needsKey && <ApiKeyNotice onSubmit={saveKey} />}
          <AskForm
            question={question}
            onQuestionChange={setQuestion}
            busy={state.status === "streaming"}
            onAsk={submit}
            onCancel={cancel}
          />
          {state.status === "idle" && <Intro />}
          {state.status === "streaming" && (
            <section className="card streaming-label" role="status">
              <span className="spinner" aria-hidden="true" />
              {state.steps.length ? `已完成：${state.steps.at(-1)!.label}` : "正在检索知识库…"}
            </section>
          )}
          {state.status === "cancelled" && (
            <CancelledNotice stepCount={state.steps.length} onRetry={() => submit(state.question)} />
          )}
          {state.status === "failed" && !queryUnauthorized && (
            <ErrorNotice error={state.error} onRetry={() => submit(state.question)} />
          )}
          {state.status === "done" && (
            <>
              <Answer result={state.result} onCite={cite} />
              <Sources result={state.result} tab={tab} onTabChange={setTab} focus={focus} />
            </>
          )}
        </div>
        <aside className="col-side">
          <AgentGraph steps={state.steps} running={state.status === "streaming"} finished={state.status === "done"} />
          <Timeline steps={state.steps} />
          <KnowledgeBase documents={kb.documents} error={kb.error} onChanged={() => void kb.refresh()} />
        </aside>
      </main>
      <footer className="foot">示例文档均为虚构公司「衡远金融科技」的演示材料</footer>
    </>
  );
}

function Intro() {
  return (
    <section className="card empty">
      <h2>它是怎么工作的</h2>
      <ol>
        <li><b>混合检索</b>：FAISS 向量检索与 BM25 关键词检索并行，用 RRF 融合排序，再由 BGE Cross-Encoder 重排。</li>
        <li><b>相关性评估</b>：Agent 判断召回片段是否真的能回答问题；不相关则<b>改写查询</b>重新检索。</li>
        <li><b>生成与自检</b>：基于片段生成带引用的答案，再做幻觉检测；仍无依据时拒答而不是编造。</li>
      </ol>
      <p className="hint">状态图会实时高亮 Agent 走过的每一个节点。</p>
    </section>
  );
}
