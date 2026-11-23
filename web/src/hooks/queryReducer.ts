import type { QueryResult, Step } from "@rag/shared";
import type { ApiError } from "../api/client";

/** A discriminated union, so impossible combinations (e.g. error + result) cannot be represented. */
export type QueryState =
  | { status: "idle"; steps: Step[] }
  | { status: "streaming"; question: string; steps: Step[] }
  | { status: "done"; question: string; steps: Step[]; result: QueryResult }
  | { status: "cancelled"; question: string; steps: Step[] }
  | { status: "failed"; question: string; steps: Step[]; error: ApiError };

export type QueryAction =
  | { type: "start"; question: string }
  | { type: "step"; step: Step }
  | { type: "result"; result: QueryResult }
  | { type: "cancel" }
  | { type: "fail"; error: ApiError };

export const initialQueryState: QueryState = { status: "idle", steps: [] };

export function queryReducer(state: QueryState, action: QueryAction): QueryState {
  if (action.type === "start") return { status: "streaming", question: action.question, steps: [] };
  // Anything arriving after a terminal state is stale.
  if (state.status !== "streaming") return state;

  const { question, steps } = state;
  switch (action.type) {
    case "step":
      return { ...state, steps: [...steps, action.step] };
    case "result":
      return { status: "done", question, steps, result: action.result };
    case "cancel":
      return { status: "cancelled", question, steps };
    case "fail":
      return { status: "failed", question, steps, error: action.error };
  }
}
