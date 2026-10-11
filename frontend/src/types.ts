// These mirror the backend models in backend/app/models.py. If you change one side, change the other.

export type Role = "system" | "user" | "assistant";

// What we send to POST /chat (the backend fills in ids and defaults).
export interface ChatMessage {
  role: Role;
  content: string;
}

export interface Usage {
  tokens_in: number;
  tokens_out: number;
}

// The three kinds of event the backend streams. `type` tells them apart (a discriminated union).
export interface DeltaEvent {
  type: "delta";
  text: string;
}

export interface DoneEvent {
  type: "done";
  request_id: string;
  usage: Usage;
}

export type ErrorCode =
  | "auth"
  | "model_not_found"
  | "rate_limited"
  | "timeout"
  | "unavailable"
  | "bad_request"
  | "context_too_large"
  | "unknown";

export interface ErrorEvent {
  type: "error";
  request_id: string;
  code: ErrorCode;
  message: string;
  retryable: boolean;
}

export type StreamEvent = DeltaEvent | DoneEvent | ErrorEvent;

// Types are erased at runtime, so data from the network must be checked by hand.
// `unknown` forces us to do that before we can use the value.
export function isStreamEvent(value: unknown): value is StreamEvent {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  switch (v.type) {
    case "delta":
      return typeof v.text === "string";
    case "done":
      return typeof v.request_id === "string" && typeof v.usage === "object" && v.usage !== null;
    case "error":
      return (
        typeof v.request_id === "string" &&
        typeof v.code === "string" &&
        typeof v.message === "string" &&
        typeof v.retryable === "boolean"
      );
    default:
      return false;
  }
}
