import { MAX_QUESTION_LENGTH } from "@rag/shared";
import type { FormEvent, KeyboardEvent } from "react";

const EXAMPLES = [
  "自然人客户现金交易达到多少需要报告大额交易？",
  "和关联自然人的交易超过多少钱要上董事会？",
  "发生个人信息泄露后，多长时间内必须向监管部门报告？",
  "2025 年公司营业收入是多少，同比增长多少？",
  "去上海出差住酒店一晚的报销上限是多少？",
];
const OUT_OF_SCOPE = "公司的员工年假有多少天？";

interface Props {
  question: string;
  onQuestionChange: (q: string) => void;
  busy: boolean;
  onAsk: (q: string) => void;
  onCancel: () => void;
}

export function AskForm({ question, onQuestionChange, busy, onAsk, onCancel }: Props) {
  const trimmed = question.trim();
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (trimmed && !busy) onAsk(trimmed);
  };
  const onKeyDown = (e: KeyboardEvent) => {
    // isComposing: Enter that confirms a Chinese IME candidate must not submit.
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) submit(e);
  };
  return (
    <form className="card ask" onSubmit={submit}>
      <label htmlFor="question" className="label">向知识库提问</label>
      <div className="ask-row">
        <textarea
          id="question"
          rows={2}
          maxLength={MAX_QUESTION_LENGTH}
          placeholder="例如：和关联自然人的交易超过多少钱要上董事会？"
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          onKeyDown={onKeyDown}
        />
        {busy ? (
          <button key="stop" type="button" className="btn danger" onClick={onCancel}>停止</button>
        ) : (
          <button key="ask" type="submit" className="btn primary" disabled={!trimmed}>分析</button>
        )}
      </div>
      <div className="examples">
        {[...EXAMPLES, OUT_OF_SCOPE].map((q) => (
          <button
            key={q}
            type="button"
            className={q === OUT_OF_SCOPE ? "chip oos" : "chip"}
            title={q === OUT_OF_SCOPE ? "知识库里没有答案，用来演示拒答" : undefined}
            disabled={busy}
            onClick={() => onAsk(q)}
          >
            {q}
          </button>
        ))}
      </div>
    </form>
  );
}
