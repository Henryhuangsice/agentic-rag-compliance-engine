import { createHash, timingSafeEqual } from "node:crypto";

const digest = (s: string) => createHash("sha256").update(s).digest();

/** Returns a caller id for a valid key, or null. Hashing first gives equal-length buffers. */
export function createAuthenticator(keys: string[]) {
  const known = keys.map(digest);
  return (header: string | undefined): string | null => {
    const token = header?.match(/^Bearer (.+)$/)?.[1];
    if (!token) return null;
    const actual = digest(token);
    const match = known.filter((k) => timingSafeEqual(k, actual)).length > 0;
    return match ? `key:${actual.toString("hex").slice(0, 12)}` : null;
  };
}
