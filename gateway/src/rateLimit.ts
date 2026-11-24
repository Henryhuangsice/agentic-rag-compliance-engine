interface Bucket {
  tokens: number;
  updatedAt: number;
}

/** In-memory token bucket per caller. Multiple gateway replicas would need a shared store. */
export function createRateLimiter(
  { burst, perMinute }: { burst: number; perMinute: number },
  now: () => number = Date.now,
) {
  const buckets = new Map<string, Bucket>();
  const perMs = perMinute / 60_000;
  const refillMs = burst / perMs;
  let sweptAt = now();

  return {
    /** Seconds to wait, or 0 when the call is allowed. */
    take(caller: string): number {
      const t = now();
      if (t - sweptAt > refillMs) {
        // A bucket idle this long is full again, so forgetting it changes nothing.
        for (const [id, b] of buckets) if (t - b.updatedAt > refillMs) buckets.delete(id);
        sweptAt = t;
      }
      const b = buckets.get(caller) ?? { tokens: burst, updatedAt: t };
      const tokens = Math.min(burst, b.tokens + (t - b.updatedAt) * perMs);
      const allowed = tokens >= 1;
      buckets.set(caller, { tokens: allowed ? tokens - 1 : tokens, updatedAt: t });
      return allowed ? 0 : Math.ceil((1 - tokens) / perMs / 1000);
    },
    get size() {
      return buckets.size;
    },
  };
}
