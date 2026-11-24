/**
 * Incremental SSE parser. Network chunks split frames (and multi-byte characters)
 * at arbitrary points, so state is kept between pushes. Only `data:` is surfaced;
 * comments (heartbeats) and other fields are ignored.
 */
export function createSseParser(onData: (data: string) => void) {
  const decoder = new TextDecoder();
  let buffer = "";
  let data: string[] = [];

  return {
    push(chunk: Uint8Array) {
      buffer += decoder.decode(chunk, { stream: true });
      let end: number;
      while ((end = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, end).replace(/\r$/, "");
        buffer = buffer.slice(end + 1);
        if (line === "") {
          if (data.length) onData(data.join("\n"));
          data = [];
        } else if (line.startsWith("data:")) {
          data.push(line.slice(5).replace(/^ /, ""));
        }
      }
    },
  };
}
