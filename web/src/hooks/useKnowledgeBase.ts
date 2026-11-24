import type { DocumentInfo, Health } from "@rag/shared";
import { useCallback, useEffect, useState } from "react";
import { ApiError, getDocuments, getHealth } from "../api/client";

export function useKnowledgeBase() {
  const [health, setHealth] = useState<Health | "loading" | "down">("loading");
  const [documents, setDocuments] = useState<DocumentInfo[] | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    const [h, d] = await Promise.allSettled([getHealth(signal), getDocuments(signal)]);
    if (signal?.aborted) return;
    setHealth(h.status === "fulfilled" ? h.value : "down");
    if (d.status === "fulfilled") {
      setDocuments(d.value);
      setError(null);
    } else {
      setError(d.reason instanceof ApiError ? d.reason : new ApiError("加载失败", "protocol"));
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  return { health, documents, error, refresh };
}
