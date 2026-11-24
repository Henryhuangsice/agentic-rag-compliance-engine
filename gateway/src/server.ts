import { randomUUID } from "node:crypto";
import type { Readable } from "node:stream";
import fastifyStatic from "@fastify/static";
import { MAX_UPLOAD_BYTES, queryRequestSchema } from "@rag/shared";
import Fastify, { type FastifyReply, type FastifyRequest } from "fastify";
import { createAuthenticator } from "./auth";
import type { Config } from "./config";
import { GatewayError } from "./errors";
import { createRateLimiter } from "./rateLimit";
import { pipeSse } from "./sseProxy";
import { callUpstream, createDeadline } from "./upstream";

// Multipart framing adds a little on top of the file itself.
const MAX_MULTIPART_BYTES = MAX_UPLOAD_BYTES + 64 * 1024;

async function* limitBytes(stream: Readable, max: number) {
  let total = 0;
  for await (const chunk of stream) {
    total += (chunk as Buffer).length;
    if (total > max) throw new GatewayError("payload_too_large", "文件超过 10 MB");
    yield chunk as Buffer;
  }
}

export async function buildServer(config: Config, { logger = false } = {}) {
  const app = Fastify({ logger, genReqId: () => randomUUID(), bodyLimit: 64 * 1024 });
  const authenticate = createAuthenticator(config.apiKeys);
  const limiter = createRateLimiter(config.rateLimit);

  // Uploads are streamed through, never buffered.
  app.addContentTypeParser("multipart/form-data", (_req, payload, done) => done(null, payload));

  app.addHook("onRequest", async (req, reply) => {
    reply.header("x-request-id", req.id);
    reply.header("x-content-type-options", "nosniff");
  });

  app.setErrorHandler((err, req, reply) => {
    let e: GatewayError;
    if (err instanceof GatewayError) e = err;
    else {
      const status = (err as { statusCode?: number }).statusCode ?? 500;
      if (status === 413) e = new GatewayError("payload_too_large", "请求体过大");
      else if (status < 500) e = new GatewayError("invalid_request", "请求格式不正确", status);
      else {
        req.log.error(err);
        e = new GatewayError("internal", "网关内部错误");
      }
    }
    return reply.code(e.status).send(e.toBody(req.id));
  });
  app.setNotFoundHandler((req, reply) => {
    const e = new GatewayError("not_found", "资源不存在");
    return reply.code(e.status).send(e.toBody(req.id));
  });

  const requireAuth = async (req: FastifyRequest) => {
    if (config.apiKeys.length && !authenticate(req.headers.authorization)) {
      throw new GatewayError("unauthorized", "API Key 缺失或无效");
    }
  };
  const rateLimit = async (req: FastifyRequest, reply: FastifyReply) => {
    const wait = limiter.take(authenticate(req.headers.authorization) ?? `ip:${req.ip}`);
    if (wait > 0) {
      reply.header("retry-after", wait);
      throw new GatewayError("rate_limited", `请求过于频繁，请 ${wait} 秒后重试`);
    }
  };
  const protectedRoute = { onRequest: requireAuth };
  const limitedRoute = { onRequest: [requireAuth, rateLimit] };

  /** Run an upstream call that is cancelled when the client disconnects. */
  async function withDeadline<T>(
    reply: FastifyReply,
    fn: (deadline: ReturnType<typeof createDeadline>) => Promise<T>,
  ): Promise<T> {
    const deadline = createDeadline(config.idleTimeoutMs);
    reply.raw.once("close", deadline.cancel);
    try {
      return await fn(deadline);
    } finally {
      deadline.clear();
      reply.raw.off("close", deadline.cancel);
    }
  }

  const proxyJson = (path: string) => (req: FastifyRequest, reply: FastifyReply) =>
    withDeadline(reply, async ({ signal }) => {
      const res = await callUpstream(config, path, { signal, headers: { "x-request-id": req.id } });
      return res.json();
    });

  app.get("/healthz", async () => ({ status: "ok" }));
  app.get("/api/v1/health", proxyJson("/api/v1/health"));
  app.get("/api/v1/documents", protectedRoute, proxyJson("/api/v1/documents"));

  app.post("/api/v1/documents", limitedRoute, (req, reply) => {
    const contentType = req.headers["content-type"];
    if (!contentType?.startsWith("multipart/form-data")) {
      throw new GatewayError("invalid_request", "需要 multipart/form-data");
    }
    if (Number(req.headers["content-length"] ?? 0) > MAX_MULTIPART_BYTES) {
      throw new GatewayError("payload_too_large", "文件超过 10 MB");
    }
    return withDeadline(reply, async ({ signal }) => {
      const res = await callUpstream(config, "/api/v1/documents", {
        method: "POST",
        headers: { "content-type": contentType, "x-request-id": req.id },
        body: limitBytes(req.body as Readable, MAX_MULTIPART_BYTES),
        duplex: "half",
        signal,
      });
      return res.json();
    });
  });

  app.post("/api/v1/query/stream", limitedRoute, async (req, reply) => {
    const parsed = queryRequestSchema.safeParse(req.body);
    if (!parsed.success) throw new GatewayError("invalid_request", "问题长度须为 1–1000 字");

    await withDeadline(reply, async (deadline) => {
      const upstream = await callUpstream(config, "/api/v1/query/stream", {
        method: "POST",
        headers: { "content-type": "application/json", "x-request-id": req.id },
        body: JSON.stringify(parsed.data),
        signal: deadline.signal,
      });
      if (!upstream.body) throw new GatewayError("upstream_error", "分析服务返回了空响应");
      reply.hijack();
      reply.raw.writeHead(200, {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache, no-transform",
        "x-accel-buffering": "no",
        "x-request-id": req.id,
      });
      await pipeSse(upstream.body, reply.raw, deadline, config.heartbeatMs);
    });
  });

  if (config.webDist) await app.register(fastifyStatic, { root: config.webDist });
  return app;
}
