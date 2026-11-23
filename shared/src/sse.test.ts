import { expect, test } from "vitest";
import { createSseParser } from "./sse";

function parse(text: string, chunkSize: number) {
  const out: string[] = [];
  const parser = createSseParser((d) => out.push(d));
  const bytes = new TextEncoder().encode(text);
  for (let i = 0; i < bytes.length; i += chunkSize) parser.push(bytes.slice(i, i + chunkSize));
  return out;
}

test.each([1, 2, 3, 7, 1000])("reassembles frames and Chinese text from %i-byte chunks", (size) => {
  expect(parse('data: {"a":"合规"}\n\ndata: 二\n\n', size)).toEqual(['{"a":"合规"}', "二"]);
});

test("ignores heartbeats, accepts CRLF, joins multi-line data", () => {
  expect(parse(": ping\n\ndata: a\r\ndata: b\r\n\r\nevent: x\n\n", 4)).toEqual(["a\nb"]);
});

test("does not emit an unterminated frame", () => {
  expect(parse("data: half", 3)).toEqual([]);
});
