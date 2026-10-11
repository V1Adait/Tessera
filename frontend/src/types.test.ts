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
