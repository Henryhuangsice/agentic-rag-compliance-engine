import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { App } from "./App";

const encoder = new TextEncoder();
const frame = (event: unknown) => encoder.encode(`data: ${JSON.stringify(event)}\n\n`);
const stepEvent = { type: "step", step: { node: "retrieve", label: "混合检索", detail: "召回 4", ms: 5 } };
const source = {
  index: 1, source: "a.md", title: "关联交易制度", section: "第二章", content: "30 万元以上须提交董事会",
  scores: { vector_rank: 1, bm25_rank: null, rrf_score: 0.03, rerank_score: 0.9 },
};
const resultEvent = {
  type: "result",
  result: {
    answer: "须提交董事会 [1]，参见 [9]", sources: [source], candidates: [source], trace: [], mode: "llm",
    rewritten_query: null, grounded: true, groundedness: 1, latency_ms: 1200,
  },
};
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

type QueryHandler = (init: RequestInit) => Response;
function mockApi(onQuery: QueryHandler) {
  const queries: RequestInit[] = [];
  vi.stubGlobal("fetch", async (path: string, init: RequestInit = {}) => {
    if (path.endsWith("/health")) return json({ status: "ok", llm_enabled: false, llm_model: null, embedding_model: "e", reranker_model: null, chunks: 1 });
    if (path.endsWith("/documents")) return json([]);
    queries.push(init);
    return onQuery(init);
  });
  return queries;
}
/** Emits events, then either closes or stays open until aborted. */
const stream = (events: unknown[], { open = false } = {}): QueryHandler => (init) =>
  new Response(new ReadableStream({
    start(c) {
      events.forEach((e) => c.enqueue(frame(e)));
      if (!open) c.close();
      else init.signal?.addEventListener("abort", () => c.error(new DOMException("", "AbortError")));
    },
  }));

async function submit(text = "测试问题") {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("向知识库提问"), text);
  await user.click(screen.getByRole("button", { name: "分析" }));
  return user;
}

beforeEach(() => {
  history.replaceState(null, "", "/");
  sessionStorage.clear();
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("streams steps, renders the answer, and links only real citations", async () => {
  mockApi(stream([stepEvent, resultEvent]));
  render(<App />);
  const user = await submit();
  await user.click(await screen.findByRole("button", { name: "查看来源 1" }));
  expect(document.querySelector("[data-source='1']")?.className).toContain("flash");
  expect(screen.queryByRole("button", { name: "查看来源 9" })).toBeNull();
  expect(screen.getByText("混合检索")).toBeTruthy();
});

test("stop aborts the request and keeps the partial trace", async () => {
  const queries = mockApi(stream([stepEvent], { open: true }));
  render(<App />);
  const user = await submit();
  await screen.findByText("召回 4");
  await user.click(screen.getByRole("button", { name: "停止" }));
  expect(queries[0]!.signal!.aborted).toBe(true);
  expect(screen.getByText("已停止")).toBeTruthy();
  expect(screen.getByText("召回 4")).toBeTruthy();
});

test("a truncated stream is an error that can be retried", async () => {
  const handlers = [stream([stepEvent]), stream([stepEvent, resultEvent])];
  mockApi((init) => handlers.shift()!(init));
  render(<App />);
  const user = await submit();
  await screen.findByText("响应不完整，请重试");
  await user.click(screen.getByRole("button", { name: "重试" }));
  await screen.findByRole("button", { name: "查看来源 1" });
});

test("an unknown event is reported, not rendered", async () => {
  mockApi(stream([{ type: "step", step: { node: "unknown" } }]));
  render(<App />);
  await submit();
  await screen.findByText("收到无法识别的事件");
});

test("401 asks for a key and retries with it", async () => {
  const handlers: QueryHandler[] = [
    () => json({ error: { code: "unauthorized", message: "no" } }, 401),
    stream([resultEvent]),
  ];
  const queries = mockApi((init) => handlers.shift()!(init));
  render(<App />);
  const user = await submit();
  await user.type(await screen.findByLabelText("访问密钥"), "test-key");
  await user.click(screen.getByRole("button", { name: "确认" }));
  await screen.findByRole("button", { name: "查看来源 1" });
  expect(new Headers(queries[1]!.headers).get("authorization")).toBe("Bearer test-key");
});
