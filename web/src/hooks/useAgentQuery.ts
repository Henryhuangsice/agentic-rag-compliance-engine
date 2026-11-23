import { useCallback, useEffect, useReducer, useRef } from "react";
import { ApiError, streamQuery } from "../api/client";
import { initialQueryState, queryReducer } from "./queryReducer";

export function useAgentQuery() {
  const [state, dispatch] = useReducer(queryReducer, initialQueryState);
  const active = useRef<AbortController | null>(null);

  const ask = useCallback(async (question: string) => {
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    dispatch({ type: "start", question });
    try {
      await streamQuery(
        question,
        (event) => {
          // A chunk already being parsed can outlive an abort.
          if (!controller.signal.aborted) dispatch(event);
        },
        controller.signal,
      );
    } catch (err) {
      if (controller.signal.aborted) return;
      dispatch({
        type: "fail",
        error: err instanceof ApiError ? err : new ApiError("发生未知错误", "protocol"),
      });
    }
  }, []);

  const cancel = useCallback(() => {
    active.current?.abort();
    dispatch({ type: "cancel" });
  }, []);

  useEffect(() => () => active.current?.abort(), []);

  return { state, ask, cancel };
}
