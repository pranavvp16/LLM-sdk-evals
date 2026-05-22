export type Turn =
  | { kind: "user"; content: string }
  | { kind: "assistant"; content: string; thinking?: string; pending?: boolean }
  | {
      kind: "tool";
      id: string;
      name: string;
      args: Record<string, unknown>;
      result?: unknown;
      isError?: boolean;
    }
  | { kind: "system"; content: string };

export function appendText(turns: Turn[], delta: string): Turn[] {
  const last = turns[turns.length - 1];
  if (last && last.kind === "assistant" && last.pending) {
    const next = [...turns];
    next[next.length - 1] = { ...last, content: last.content + delta };
    return next;
  }
  // A tool round-trip closed the previous bubble — open a new one.
  return [...turns, { kind: "assistant", content: delta, pending: true }];
}

export function appendThinking(turns: Turn[], delta: string): Turn[] {
  const last = turns[turns.length - 1];
  if (last && last.kind === "assistant" && last.pending) {
    const next = [...turns];
    next[next.length - 1] = { ...last, thinking: (last.thinking ?? "") + delta };
    return next;
  }
  // Open a new pending assistant turn to host the thinking stream.
  return [...turns, { kind: "assistant", content: "", thinking: delta, pending: true }];
}

export function addToolCall(
  turns: Turn[],
  id: string,
  name: string,
  args: Record<string, unknown>,
): Turn[] {
  const next = [...turns];
  // Close the current pending assistant bubble (next text starts a new one).
  const lastIdx = next.length - 1;
  if (lastIdx >= 0 && next[lastIdx].kind === "assistant") {
    const a = next[lastIdx] as Extract<Turn, { kind: "assistant" }>;
    if (!a.content && !a.thinking) {
      // Drop empty placeholder so we don't render a stray "…".
      next.pop();
    } else {
      next[lastIdx] = { ...a, pending: false };
    }
  }
  next.push({ kind: "tool", id, name, args });
  return next;
}

export function attachToolResult(
  turns: Turn[],
  id: string,
  result: unknown,
  isError: boolean,
): Turn[] {
  return turns.map((t) =>
    t.kind === "tool" && t.id === id ? { ...t, result, isError } : t,
  );
}

export function finalizeLastAssistant(turns: Turn[]): Turn[] {
  const next = [...turns];
  const lastIdx = next.length - 1;
  if (lastIdx >= 0 && next[lastIdx].kind === "assistant") {
    const a = next[lastIdx] as Extract<Turn, { kind: "assistant" }>;
    next[lastIdx] = { ...a, pending: false };
  }
  return next;
}
