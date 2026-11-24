import type { ErrorBody, ErrorCode } from "@rag/shared";

const STATUS: Record<ErrorCode, number> = {
  invalid_request: 400,
  unauthorized: 401,
  not_found: 404,
  payload_too_large: 413,
  rate_limited: 429,
  internal: 500,
  upstream_error: 502,
  upstream_unavailable: 503,
  upstream_timeout: 504,
};

export class GatewayError extends Error {
  readonly status: number;
  constructor(readonly code: ErrorCode, message: string, status?: number) {
    super(message);
    this.status = status ?? STATUS[code];
  }
  toBody(requestId: string): ErrorBody {
    return { error: { code: this.code, message: this.message, requestId } };
  }
}
