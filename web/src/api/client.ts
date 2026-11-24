import {
  createSseParser,
  documentSchema,
  errorBodySchema,
  healthSchema,
  streamEventSchema,
  uploadResultSchema,
  type ErrorCode,
  type StreamEvent,
} from "@rag/shared";
import { z } from "zod";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: ErrorCode | "network" | "protocol",
    readonly requestId?: string,
  ) {
    super(message);
  }
}

// Kept for the tab's lifetime only.
const KEY = "rag.apiKey";
export const apiKey = {
  get: () => sessionStorage.getItem(KEY),
  set: (value: string) => sessionStorage.setItem(KEY, value),
};

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const key = apiKey.get();
  if (key) headers.set("authorization", `Bearer ${key}`);

  let res: Response;
  try {
    res = await fetch(path, { ...init, headers });
  } catch (err) {
    if (init.signal?.aborted) throw err;
    throw new ApiError("无法连接到服务，请检查网络", "network");
  }
  if (res.ok) return res;
  const body = errorBodySchema.safeParse(await res.json().catch(() => null));
  if (body.success) {
    const { message, code, requestId } = body.data.error;
    throw new ApiError(message, code, requestId);
  }
  throw new ApiError(`服务返回 ${res.status}`, "protocol");
}

async function json<T>(schema: z.ZodType<T>, res: Response): Promise<T> {
  const parsed = schema.safeParse(await res.json().catch(() => null));
  if (!parsed.success) throw new ApiError("服务返回了无法识别的数据", "protocol");
  return parsed.data;
}

export const getHealth = async (signal?: AbortSignal) =>
  json(healthSchema, await request("/api/v1/health", { signal }));

export const getDocuments = async (signal?: AbortSignal) =>
  json(z.array(documentSchema), await request("/api/v1/documents", { signal }));

export async function uploadDocument(file: File) {
  const body = new FormData();
  body.append("file", file);
  return json(uploadResultSchema, await request("/api/v1/documents", { method: "POST", body }));
}

/** Resolves once a `result` has been delivered; rejects on error, truncation or abort. */
export async function streamQuery(
  question: string,
  onEvent: (event: Exclude<StreamEvent, { type: "error" }>) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await request("/api/v1/query/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  const requestId = res.headers.get("x-request-id") ?? undefined;
  let finished = false;

  const parser = createSseParser((data) => {
    let raw: unknown;
    try {
      raw = JSON.parse(data);
    } catch {
      throw new ApiError("收到无法解析的数据", "protocol", requestId);
    }
    const event = streamEventSchema.safeParse(raw);
    if (!event.success) throw new ApiError("收到无法识别的事件", "protocol", requestId);
    if (event.data.type === "error") {
      throw new ApiError(event.data.error, event.data.code ?? "upstream_error", requestId);
    }
    if (event.data.type === "result") finished = true;
    onEvent(event.data);
  });

  const reader = res.body!.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      parser.push(value);
    }
  } catch (err) {
    if (signal.aborted || err instanceof ApiError) throw err;
    throw new ApiError("连接中断，请重试", "network", requestId);
  } finally {
    // Stops the server's work too when we bail out early.
    await reader.cancel().catch(() => {});
  }
  if (!finished) throw new ApiError("响应不完整，请重试", "protocol", requestId);
}
