import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, expect, test, vi } from "vitest";
import { loadConfig } from "../src/config";
import { createRateLimiter } from "../src/rateLimit";
import { buildServer } from "../src/server";

type Handler = (req: IncomingMessage, res: ServerResponse) => void;
const cleanups: (() => unknown)[] = [];
afterEach(async () => {
  for (const fn of cleanups.splice(0).reverse()) await fn();
});

async function setup(handler: Handler, env: Record<string, string> = {}) {
  const upstream: Server = createServer(handler);
  await new Promise<void>((r) => upstream.listen(0, "127.0.0.1", r));
  const { port } = upstream.address() as AddressInfo;
  const app = await buildServer(loadConfig({ UPSTREAM_URL: `http://127.0.0.1:${port}`, ...env }));
  const base = await app.listen({ host: "127.0.0.1", port: 0 });
  cleanups.push(() => new Promise((r) => { upstream.closeAllConnections(); upstream.close(r); }));
  cleanups.push(() => { app.server.closeAllConnections(); return app.close(); });
  return base;
}

const ask = (base: string, init: { question?: string; key?: string; signal?: AbortSignal } = {}) =>
  fetch(`${base}/api/v1/query/stream`, {
    method: "POST",
    headers: { "content-type": "application/json", ...(init.key && { authorization: `Bearer ${init.key}` }) },
    body: JSON.stringify({ question: init.question ?? "问题" }),
    signal: init.signal,
  });
const sse: Handler = (_req, res) => {
  res.writeHead(200, { "content-type": "text/event-stream" });
  res.end('data: {"type":"x"}\n\n');
};

test("streams upstream frames through unchanged", async () => {
  const base = await setup(sse);
  const res = await ask(base);
  expect(res.headers.get("content-type")).toContain("text/event-stream");
  expect(res.headers.get("x-request-id")).toBeTruthy();
  expect(await res.text()).toBe('data: {"type":"x"}\n\n');
});

test("rejects blank and oversized questions without calling upstream", async () => {
  const handler = vi.fn(sse);
  const base = await setup(handler);
  for (const question of ["   ", "长".repeat(1001)]) {
    const res = await ask(base, { question });
    expect(res.status).toBe(400);
    expect(await res.json()).toMatchObject({ error: { code: "invalid_request" } });
  }
  expect(handler).not.toHaveBeenCalled();
});

test("requires a valid key when keys are configured", async () => {
  const base = await setup(sse, { GATEWAY_API_KEYS: "test-a, test-b" });
  expect((await ask(base)).status).toBe(401);
  expect((await ask(base, { key: "wrong" })).status).toBe(401);
  expect((await ask(base, { key: "test-b" })).status).toBe(200);
});

test("limits per caller and says when to retry", async () => {
  const base = await setup(sse, { RATE_LIMIT_BURST: "1", RATE_LIMIT_PER_MINUTE: "6", GATEWAY_API_KEYS: "a,b" });
  expect((await ask(base, { key: "a" })).status).toBe(200);
  const limited = await ask(base, { key: "a" });
  expect(limited.status).toBe(429);
  expect(limited.headers.get("retry-after")).toBe("10");
  expect((await ask(base, { key: "b" })).status).toBe(200);
});

test("cancels the upstream request when the client disconnects", async () => {
  const closed = Promise.withResolvers<void>();
  const base = await setup((_req, res) => {
    res.on("close", () => closed.resolve());
    res.writeHead(200);
    res.write("data: 1\n\n");
  });
  const controller = new AbortController();
  const res = await ask(base, { signal: controller.signal });
  await res.body!.getReader().read();
  controller.abort();
  await closed.promise;
});

test("cancels upstream even before it has sent headers", async () => {
  const started = Promise.withResolvers<void>();
  const closed = Promise.withResolvers<void>();
  const base = await setup((_req, res) => {
    res.on("close", () => closed.resolve());
    started.resolve();
  });
  const controller = new AbortController();
  const pending = ask(base, { signal: controller.signal }).catch(() => null);
  await started.promise;
  controller.abort();
  await Promise.all([pending, closed.promise]);
});

test("reports an upstream that dies mid-frame as a parseable error event", async () => {
  const base = await setup((_req, res) => {
    res.writeHead(200);
    res.write('data: {"half', () => res.destroy());
  });
  const text = await (await ask(base)).text();
  const last = text.trim().split("\n\n").at(-1)!;
  expect(JSON.parse(last.slice(6))).toMatchObject({ type: "error", code: "upstream_unavailable" });
});

test("times out a silent stream", async () => {
  const base = await setup((_req, res) => void res.writeHead(200).write("data: 1\n\n"), {
    UPSTREAM_IDLE_TIMEOUT_MS: "80",
  });
  expect(await (await ask(base)).text()).toContain('"code":"upstream_timeout"');
});

test("sends heartbeats only between frames", async () => {
  const base = await setup((_req, res) => {
    res.writeHead(200);
    res.write("data: a\n\ndata: {half");
    setTimeout(() => res.write("}\n\n"), 80);
    setTimeout(() => res.end(), 160);
  }, { HEARTBEAT_MS: "20" });
  const text = await (await ask(base)).text();
  expect(text).toContain("data: {half}\n\n: ping\n\n");
});

test("hides upstream internals on 5xx, keeps the reason on 4xx", async () => {
  let status = 500;
  const base = await setup((_req, res) => {
    res.writeHead(status, { "content-type": "application/json" }).end('{"detail":"secret path"}');
  });
  const bad = await ask(base);
  expect(bad.status).toBe(502);
  expect(await bad.text()).not.toContain("secret");
  status = 400;
  expect(await (await ask(base)).text()).toContain("secret path");
});

test("maps an unreachable upstream to 503", async () => {
  const base = await setup(sse, { UPSTREAM_URL: "http://127.0.0.1:1" });
  const res = await fetch(`${base}/api/v1/health`);
  expect(res.status).toBe(503);
});

test("streams uploads through", async () => {
  const base = await setup((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ ok: body.includes("年假") })));
  });
  const form = new FormData();
  form.append("file", new Blob(["年假"]), "a.md");
  const res = await fetch(`${base}/api/v1/documents`, { method: "POST", body: form });
  expect(await res.json()).toEqual({ ok: true });
});

test("rejects an oversized upload with no declared length", async () => {
  const base = await setup((req, res) => { req.resume(); req.on("end", () => res.end("{}")); });
  const chunk = new Uint8Array(1024 * 1024);
  async function* body() { for (let i = 0; i < 12; i++) yield chunk; }
  const res = await fetch(`${base}/api/v1/documents`, {
    method: "POST", headers: { "content-type": "multipart/form-data; boundary=x" }, body: body(), duplex: "half",
  } as RequestInit);
  expect(res.status).toBe(413);
});

test("token bucket refills over time and forgets idle callers", () => {
  let t = 0;
  const limiter = createRateLimiter({ burst: 2, perMinute: 60 }, () => t);
  expect([limiter.take("a"), limiter.take("a"), limiter.take("a")]).toEqual([0, 0, 1]);
  t = 1000;
  expect(limiter.take("a")).toBe(0);
  t = 10_000;
  limiter.take("b");
  expect(limiter.size).toBe(1);
});
