import { useState } from "react";
import type { ApiError } from "../api/client";

export function ErrorNotice({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  return (
    <section className="card notice error" role="alert">
      <div>
        <strong>{error.message}</strong>
        {error.requestId && <p className="request-id">请求 ID：{error.requestId}</p>}
      </div>
      <button type="button" className="btn small" onClick={onRetry}>重试</button>
    </section>
  );
}

export function CancelledNotice({ stepCount, onRetry }: { stepCount: number; onRetry: () => void }) {
  return (
    <section className="card notice" role="status">
      <div>
        <strong>已停止</strong>
        <p>停止前完成了 {stepCount} 个节点，轨迹保留在右侧。</p>
      </div>
      <button type="button" className="btn small" onClick={onRetry}>重新分析</button>
    </section>
  );
}

export function ApiKeyNotice({ onSubmit }: { onSubmit: (key: string) => void }) {
  const [value, setValue] = useState("");
  return (
    <section className="card notice" role="alert">
      <div>
        <strong>需要访问密钥</strong>
        <p>仅保存在当前标签页。</p>
      </div>
      <form
        className="key-form"
        onSubmit={(e) => {
          e.preventDefault();
          if (value.trim()) onSubmit(value.trim());
        }}
      >
        <input type="password" aria-label="访问密钥" autoComplete="off" value={value} onChange={(e) => setValue(e.target.value)} />
        <button type="submit" className="btn small">确认</button>
      </form>
    </section>
  );
}
