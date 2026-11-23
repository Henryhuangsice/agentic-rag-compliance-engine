import type { Step } from "@rag/shared";
import { expect, test } from "vitest";
import { initialQueryState, queryReducer } from "../hooks/queryReducer";
import { parseAnswer, parseInline } from "./answer";
import { graphProgress } from "./graph";

const step = (node: Step["node"]): Step => ({ node, label: node, detail: "", ms: 1 });

test("parses paragraphs, lists, bold and adjacent citations", () => {
  expect(parseAnswer("结论：\n- **50 万** [1][2]\n\n完")).toEqual([
    { type: "p", inline: [{ type: "text", text: "结论：" }] },
    { type: "ul", items: [[
      { type: "bold", text: "50 万" }, { type: "text", text: " " },
      { type: "cite", index: 1 }, { type: "cite", index: 2 },
    ]] },
    { type: "p", inline: [{ type: "text", text: "完" }] },
  ]);
});

test("does not mistake decimals or bold text for list items", () => {
  expect(parseAnswer("3.5 亿元")[0]?.type).toBe("p");
  expect(parseAnswer("**结论**：是")[0]?.type).toBe("p");
});

test("keeps HTML as text", () => {
  expect(parseInline("<img onerror=x>")).toEqual([{ type: "text", text: "<img onerror=x>" }]);
});

test("graph follows the rewrite loop and only reaches END when finished", () => {
  const steps = (["retrieve", "grade_documents", "transform_query", "retrieve"] as const).map(step);
  const running = graphProgress(steps, true, false);
  expect(running.active).toBe("retrieve");
  expect(running.edges).toContain("transform_query>retrieve");
  expect([...running.edges].some((e) => e.endsWith(">END"))).toBe(false);
  expect(graphProgress([step("fallback")], false, true).edges).toContain("fallback>END");
});

test("reducer ignores events after cancellation and clears steps on restart", () => {
  let s = queryReducer(initialQueryState, { type: "start", question: "a" });
  s = queryReducer(s, { type: "step", step: step("retrieve") });
  s = queryReducer(s, { type: "cancel" });
  expect(queryReducer(s, { type: "step", step: step("generate") })).toBe(s);
  expect(queryReducer(s, { type: "start", question: "b" }).steps).toEqual([]);
});
