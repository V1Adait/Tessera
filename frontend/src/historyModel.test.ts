import { afterEach, describe, expect, it, vi } from "vitest";
import { isChatToggleKey, readChatOpen, writeChatOpen } from "./chatPanel";
import { historyTitle, isConversationSummary, messageCount, timeAgo } from "./historyModel";

const NOW = new Date("2026-10-17T12:00:00Z");
const ago = (ms: number) => new Date(NOW.getTime() - ms).toISOString();

describe("isConversationSummary", () => {
  const ok = { id: "c1", title: "Palantir leverage", updated_at: "2026-10-17T11:00:00Z", message_count: 4 };

  it("accepts what the server sends, with or without a title", () => {
    expect(isConversationSummary(ok)).toBe(true);
    expect(isConversationSummary({ ...ok, title: null })).toBe(true);
  });

  it("rejects a wrong shape", () => {
    expect(isConversationSummary(null)).toBe(false);
    expect(isConversationSummary({ ...ok, id: 5 })).toBe(false);
    expect(isConversationSummary({ ...ok, message_count: "4" })).toBe(false);
    expect(isConversationSummary({ id: "c1" })).toBe(false);
  });
});

describe("historyTitle", () => {
  const base = { id: "c", updated_at: "", message_count: 1 };

  it("uses the title, or says the chat is untitled", () => {
    expect(historyTitle({ ...base, title: "How leveraged is Palantir?" })).toBe("How leveraged is Palantir?");
    expect(historyTitle({ ...base, title: null })).toBe("Untitled chat");
    expect(historyTitle({ ...base, title: "   " })).toBe("Untitled chat");
  });
});

describe("timeAgo", () => {
  it("reads in the units a person thinks in", () => {
    expect(timeAgo(ago(10_000), NOW)).toBe("just now");
    expect(timeAgo(ago(5 * 60_000), NOW)).toBe("5 min ago");
    expect(timeAgo(ago(3 * 3_600_000), NOW)).toBe("3 h ago");
    expect(timeAgo(ago(2 * 86_400_000), NOW)).toBe("2 d ago");
  });

  it("switches to a date after a week", () => {
    expect(timeAgo(ago(30 * 86_400_000), NOW)).toMatch(/\d/);
    expect(timeAgo(ago(30 * 86_400_000), NOW)).not.toContain("ago");
  });

  it("treats a time slightly in the future as just now, not as a negative age", () => {
    expect(timeAgo(new Date(NOW.getTime() + 5_000).toISOString(), NOW)).toBe("just now");
  });

  it("is empty for a time it cannot read", () => {
    expect(timeAgo("not a time", NOW)).toBe("");
  });
});

describe("messageCount", () => {
  it("is singular for one", () => {
    expect(messageCount(1)).toBe("1 message");
    expect(messageCount(4)).toBe("4 messages");
  });
});

describe("the chat panel setting", () => {
  afterEach(() => vi.unstubAllGlobals());

  function fakeStorage(initial: Record<string, string> = {}) {
    const data = { ...initial };
    vi.stubGlobal("localStorage", {
      getItem: (k: string) => data[k] ?? null,
      setItem: (k: string, v: string) => {
        data[k] = v;
      },
    });
  }

  it("is open until the person folds it away", () => {
    fakeStorage();

    expect(readChatOpen()).toBe(true);
  });

  it("remembers the choice", () => {
    fakeStorage();

    writeChatOpen(false);
    expect(readChatOpen()).toBe(false);
    writeChatOpen(true);
    expect(readChatOpen()).toBe(true);
  });

  it("stays open, without throwing, when storage is blocked", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    });

    expect(readChatOpen()).toBe(true);
    expect(() => writeChatOpen(false)).not.toThrow();
  });
});

describe("isChatToggleKey", () => {
  const key = (over: Partial<{ key: string; ctrlKey: boolean; metaKey: boolean; altKey: boolean }>) => ({
    key: "\\",
    ctrlKey: false,
    metaKey: false,
    altKey: false,
    ...over,
  });

  it("is Ctrl or Cmd with the backslash key", () => {
    expect(isChatToggleKey(key({ ctrlKey: true }))).toBe(true);
    expect(isChatToggleKey(key({ metaKey: true }))).toBe(true);
  });

  it("is not a bare backslash, another key, or a combination with Alt", () => {
    expect(isChatToggleKey(key({}))).toBe(false);
    expect(isChatToggleKey(key({ ctrlKey: true, key: "b" }))).toBe(false);
    expect(isChatToggleKey(key({ ctrlKey: true, altKey: true }))).toBe(false);
  });
});
