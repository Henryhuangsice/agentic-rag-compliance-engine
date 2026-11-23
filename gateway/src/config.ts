import { z } from "zod";

const envSchema = z.object({
  HOST: z.string().default("127.0.0.1"),
  PORT: z.coerce.number().int().default(8787),
  UPSTREAM_URL: z.url().default("http://127.0.0.1:8000"),
  /** Comma-separated. Empty disables authentication (local development). */
  GATEWAY_API_KEYS: z.string().default(""),
  RATE_LIMIT_BURST: z.coerce.number().int().positive().default(10),
  RATE_LIMIT_PER_MINUTE: z.coerce.number().positive().default(30),
  /** Max silence from upstream before a request is given up. */
  UPSTREAM_IDLE_TIMEOUT_MS: z.coerce.number().int().positive().default(120_000),
  HEARTBEAT_MS: z.coerce.number().int().positive().default(15_000),
  WEB_DIST: z.string().default(""),
});

export function loadConfig(env: Record<string, string | undefined> = process.env) {
  const e = envSchema.parse(env);
  return {
    host: e.HOST,
    port: e.PORT,
    upstreamUrl: e.UPSTREAM_URL.replace(/\/$/, ""),
    apiKeys: e.GATEWAY_API_KEYS.split(",").map((k) => k.trim()).filter(Boolean),
    rateLimit: { burst: e.RATE_LIMIT_BURST, perMinute: e.RATE_LIMIT_PER_MINUTE },
    idleTimeoutMs: e.UPSTREAM_IDLE_TIMEOUT_MS,
    heartbeatMs: e.HEARTBEAT_MS,
    webDist: e.WEB_DIST || undefined,
  };
}
export type Config = ReturnType<typeof loadConfig>;
