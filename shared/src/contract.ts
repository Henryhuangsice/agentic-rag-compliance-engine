/** The wire contract: the types on both sides are inferred from these schemas. */
import { z } from "zod";

export const MAX_QUESTION_LENGTH = 1000;
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

export const queryRequestSchema = z.object({
  question: z.string().trim().min(1).max(MAX_QUESTION_LENGTH),
});

export const nodeNames = [
  "retrieve",
  "grade_documents",
  "transform_query",
  "generate",
  "grade_generation",
  "fallback",
] as const;
export type NodeName = (typeof nodeNames)[number];

export const stepSchema = z.object({
  node: z.enum(nodeNames),
  label: z.string(),
  detail: z.string(),
  ms: z.number(),
});
export type Step = z.infer<typeof stepSchema>;

const score = z.number().nullish();
export const sourceSchema = z.object({
  index: z.number(),
  source: z.string().nullable(),
  title: z.string().nullable(),
  section: z.string().nullable(),
  content: z.string(),
  scores: z.object({
    vector_rank: score,
    bm25_rank: score,
    rrf_score: score,
    rerank_score: score,
  }),
});
export type Source = z.infer<typeof sourceSchema>;

export const queryResultSchema = z.object({
  answer: z.string(),
  sources: z.array(sourceSchema),
  candidates: z.array(sourceSchema),
  trace: z.array(stepSchema),
  mode: z.enum(["llm", "offline", "fallback", "extractive_fallback"]),
  rewritten_query: z.string().nullable(),
  grounded: z.boolean().nullable(),
  groundedness: z.number().nullable(),
  latency_ms: z.number(),
});
export type QueryResult = z.infer<typeof queryResultSchema>;

export const errorCodes = [
  "invalid_request",
  "unauthorized",
  "rate_limited",
  "payload_too_large",
  "upstream_unavailable",
  "upstream_timeout",
  "upstream_error",
  "not_found",
  "internal",
] as const;
export type ErrorCode = (typeof errorCodes)[number];

/** Shape of every non-2xx response the gateway produces itself. */
export const errorBodySchema = z.object({
  error: z.object({
    code: z.enum(errorCodes),
    message: z.string(),
    requestId: z.string().optional(),
  }),
});
export type ErrorBody = z.infer<typeof errorBodySchema>;

export const streamEventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("step"), step: stepSchema }),
  z.object({ type: z.literal("result"), result: queryResultSchema }),
  z.object({ type: z.literal("error"), error: z.string(), code: z.enum(errorCodes).optional() }),
]);
export type StreamEvent = z.infer<typeof streamEventSchema>;

export const healthSchema = z.object({
  status: z.string(),
  llm_enabled: z.boolean(),
  llm_model: z.string().nullable(),
  embedding_model: z.string(),
  reranker_model: z.string().nullable(),
  chunks: z.number(),
});
export type Health = z.infer<typeof healthSchema>;

export const documentSchema = z.object({ source: z.string(), title: z.string(), chunks: z.number() });
export type DocumentInfo = z.infer<typeof documentSchema>;

export const uploadResultSchema = z.object({
  source: z.string(),
  chunks: z.number(),
  total_chunks: z.number(),
});
export type UploadResult = z.infer<typeof uploadResultSchema>;
