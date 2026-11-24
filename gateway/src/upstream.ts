import type { Config } from "./config";
import { GatewayError } from "./errors";

export class IdleTimeout extends Error {}

/** Aborts when the client leaves, or when nothing has arrived for `idleMs`. */
export function createDeadline(idleMs: number) {
  const controller = new AbortController();
  const expire = () => controller.abort(new IdleTimeout());
  let timer = setTimeout(expire, idleMs);
  return {
    signal: controller.signal,
    touch() {
      clearTimeout(timer);
      timer = setTimeout(expire, idleMs);
    },
    cancel() {
      clearTimeout(timer);
      controller.abort();
    },
    clear() {
      clearTimeout(timer);
    },
  };
}

export function toGatewayError(err: unknown): GatewayError {
  if (err instanceof GatewayError) return err;
  if (err instanceof IdleTimeout) return new GatewayError("upstream_timeout", "分析服务响应超时");
  // fetch() wraps errors thrown by a streamed request body.
  if (err instanceof Error && err.cause instanceof GatewayError) return err.cause;
  return new GatewayError("upstream_unavailable", "分析服务暂时不可用");
}

export async function callUpstream(config: Config, path: string, init: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(config.upstreamUrl + path, init);
  } catch (err) {
    throw toGatewayError(err);
  }
  if (res.ok) return res;

  const body = (await res.json().catch(() => null)) as { detail?: unknown } | null;
  // Client mistakes keep their explanation; upstream internals are not exposed.
  if (res.status < 500) {
    const detail = typeof body?.detail === "string" ? body.detail : "请求未被分析服务接受";
    throw new GatewayError(res.status === 413 ? "payload_too_large" : "invalid_request", detail, res.status);
  }
  throw new GatewayError("upstream_error", "分析服务内部错误");
}
