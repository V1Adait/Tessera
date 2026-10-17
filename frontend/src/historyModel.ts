// The chat history: what one line of it says. Pure, so every rule is easy to test.

export interface ConversationSummary {
  id: string;
  title: string | null;
  updated_at: string; // an ISO time
  message_count: number;
}

export function isConversationSummary(v: unknown): v is ConversationSummary {
  if (typeof v !== "object" || v === null) return false;
  const c = v as Record<string, unknown>;
  return (
    typeof c.id === "string" &&
    (c.title === null || typeof c.title === "string") &&
    typeof c.updated_at === "string" &&
    typeof c.message_count === "number"
  );
}

/** What a conversation is called in the list. A chat has no title until it has a first message. */
export function historyTitle(c: ConversationSummary): string {
  const title = (c.title ?? "").trim();
  return title === "" ? "Untitled chat" : title;
}

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** "just now", "5 min ago", "3 h ago", "2 d ago", then a date. "" when the time cannot be read. */
export function timeAgo(iso: string, now: Date): string {
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "";
  const age = now.getTime() - then.getTime();
  if (age < MINUTE) return "just now"; // also covers a clock a little ahead of the server's
  if (age < HOUR) return `${Math.floor(age / MINUTE)} min ago`;
  if (age < DAY) return `${Math.floor(age / HOUR)} h ago`;
  if (age < 7 * DAY) return `${Math.floor(age / DAY)} d ago`;
  return then.toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}

/** "4 messages" / "1 message". */
export function messageCount(count: number): string {
  return `${count} message${count === 1 ? "" : "s"}`;
}
