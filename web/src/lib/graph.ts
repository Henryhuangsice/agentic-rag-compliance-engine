import type { NodeName, Step } from "@rag/shared";

/** Which nodes and edges a run has taken, derived from its steps. */
export function graphProgress(steps: Step[], running: boolean, finished: boolean) {
  const path: string[] = ["START", ...steps.map((s) => s.node), ...(finished ? ["END"] : [])];
  return {
    visited: new Set<NodeName>(steps.map((s) => s.node)),
    active: running ? steps.at(-1)?.node : undefined,
    edges: new Set(path.slice(1).map((to, i) => `${path[i]}>${to}`)),
  };
}
