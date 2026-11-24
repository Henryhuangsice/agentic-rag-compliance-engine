import type { ServerResponse } from "node:http";
import { once } from "node:events";
import type { StreamEvent } from "@rag/shared";
import { toGatewayError, type createDeadline } from "./upstream";

const NEWLINE = 10;

/**
 * Pipe an upstream SSE body to the client, with backpressure, heartbeats, and
 * an in-band error event: once headers are sent a status code is no longer possible.
 */
export async function pipeSse(
  body: ReadableStream<Uint8Array>,
  res: ServerResponse,
  deadline: ReturnType<typeof createDeadline>,
  heartbeatMs: number,
): Promise<void> {
  // A heartbeat inside a half-delivered frame would corrupt it, so track frame boundaries.
  let tail: number[] = [NEWLINE, NEWLINE];
  const atBoundary = () => tail[0] === NEWLINE && tail[1] === NEWLINE;

  const heartbeat = setInterval(() => {
    if (atBoundary() && !res.writableNeedDrain) res.write(": ping\n\n");
  }, heartbeatMs);

  try {
    for await (const chunk of body) {
      deadline.touch();
      tail = [...tail, ...chunk.slice(-2)].slice(-2);
      if (!res.write(chunk)) await once(res, "drain", { signal: deadline.signal });
    }
  } catch (err) {
    if (!res.destroyed) {
      const e = toGatewayError(deadline.signal.reason ?? err);
      const event: StreamEvent = { type: "error", error: e.message, code: e.code };
      // Close any half-delivered frame with a blank line before adding ours.
      res.write(`${atBoundary() ? "" : "\n\n"}data: ${JSON.stringify(event)}\n\n`);
    }
  } finally {
    clearInterval(heartbeat);
    res.end();
  }
}
