import { describe, expect, it } from "vitest";
import { isConversationView, isStreamEvent } from "./types";

describe("isStreamEvent", () => {
  it("accepts the three real event shapes", () => {
    expect(isStreamEvent({ type: "delta", text: "hi" })).toBe(true);
    expect(
      isStreamEvent({ type: "done", request_id: "r", usage: { tokens_in: 1, tokens_out: 2 } }),
    ).toBe(true);
    expect(
      isStreamEvent({
        type: "error",
        request_id: "r",
        code: "unknown",
        message: "m",
        retryable: false,
      }),
    ).toBe(true);
  });

  it("rejects wrong shapes and non-objects", () => {
    expect(isStreamEvent(null)).toBe(false);
    expect(isStreamEvent("delta")).toBe(false);
    expect(isStreamEvent({ type: "delta" })).toBe(false); // no text
    expect(isStreamEvent({ type: "delta", text: 5 })).toBe(false);
    expect(isStreamEvent({ type: "done", request_id: "r" })).toBe(false); // no usage
    expect(isStreamEvent({ type: "nope" })).toBe(false);
  });
});

describe("the agent's events", () => {
  const tool = {
    type: "tool",
    id: "c1",
    name: "add_row",
    label: "Adding a row",
    status: "ok",
    summary: "Added row Globex",
    changed_matrix: true,
  };
  const ask = {
    type: "confirmation",
    action_id: "a1",
    tool: "remove_row",
    risk: "destructive",
    summary: "Remove the row Acme",
  };

  it("accepts a tool step and a confirmation request", () => {
    expect(isStreamEvent(tool)).toBe(true);
    expect(isStreamEvent(ask)).toBe(true);
  });

  it("rejects a tool step with an unknown status or a missing flag", () => {
    expect(isStreamEvent({ ...tool, status: "exploded" })).toBe(false);
    expect(isStreamEvent({ ...tool, changed_matrix: undefined })).toBe(false);
  });

  it("rejects a confirmation with an unknown risk level", () => {
    expect(isStreamEvent({ ...ask, risk: "fine" })).toBe(false);
    expect(isStreamEvent({ ...ask, action_id: 5 })).toBe(false);
  });

  it("accepts a finished answer with warnings, and rejects warnings that are not text", () => {
    const done = { type: "done", request_id: "r", usage: { tokens_in: 1, tokens_out: 1 } };
    expect(isStreamEvent({ ...done, warnings: ["stopped after 6 steps"] })).toBe(true);
    expect(isStreamEvent({ ...done, warnings: [1] })).toBe(false);
  });
});

describe("isConversationView", () => {
  const message = { id: "1", role: "user", content: "hi", status: "complete", tokens: null };

  it("accepts a stored conversation", () => {
    expect(isConversationView({ id: "c", title: null, messages: [message] })).toBe(true);
    expect(isConversationView({ id: "c", title: "t", messages: [] })).toBe(true);
  });

  it("rejects bad roles, bad statuses and missing parts", () => {
    expect(isConversationView({ id: "c", messages: [{ ...message, role: "system" }] })).toBe(
      false,
    );
    expect(isConversationView({ id: "c", messages: [{ ...message, status: "weird" }] })).toBe(
      false,
    );
    expect(isConversationView({ id: "c" })).toBe(false);
    expect(isConversationView(null)).toBe(false);
  });
});
