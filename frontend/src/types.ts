import { isPlanPayload, type PlanPayload } from "./planModel";

// These mirror the backend models (backend/app/models.py and backend/app/db/records.py).
// If you change one side, change the other.

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
  warnings?: string[]; // things the person should know about this answer
}

// The agent used a tool. Sent when it starts ("running") and again when it ends, with the same id.
export type ToolStatus = "running" | "ok" | "error" | "needs_confirmation";

export interface ToolStreamEvent {
  type: "tool";
  id: string;
  name: string;
  label: string;
  status: ToolStatus;
  summary: string;
  changed_matrix: boolean; // the grid should reload
}

// The agent wants to do something costly or irreversible. Nothing has happened yet.
export interface ConfirmationStreamEvent {
  type: "confirmation";
  action_id: string;
  tool: string;
  risk: "costly" | "destructive";
  summary: string;
  plan?: PlanPayload | null; // for a column plan: the columns, the cells and the cost
}

export type ErrorCode =
  | "auth"
  | "model_not_found"
  | "rate_limited"
  | "timeout"
  | "unavailable"
  | "bad_request"
  | "context_too_large"
  | "budget"
  | "unknown";

export interface ErrorEvent {
  type: "error";
  request_id: string;
  code: ErrorCode;
  message: string;
  retryable: boolean;
}

export type StreamEvent =
  | DeltaEvent
  | DoneEvent
  | ErrorEvent
  | ToolStreamEvent
  | ConfirmationStreamEvent;

// A message as the server stores it (GET /conversations/{id}).
export type StoredStatus = "streaming" | "complete" | "stopped" | "error";

export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: StoredStatus;
  tokens: number | null;
}

export interface ConversationView {
  id: string;
  workspace_id?: string;
  title: string | null;
  messages: StoredMessage[];
}

const TOOL_STATUSES: readonly string[] = ["running", "ok", "error", "needs_confirmation"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

// Types are erased at runtime, so data from the network must be checked by hand.
// `unknown` forces us to do that before we can use the value.
export function isStreamEvent(value: unknown): value is StreamEvent {
  if (!isRecord(value)) return false;
  switch (value.type) {
    case "delta":
      return typeof value.text === "string";
    case "done":
      return (
        typeof value.request_id === "string" &&
        isRecord(value.usage) &&
        (value.warnings === undefined ||
          (Array.isArray(value.warnings) && value.warnings.every((w) => typeof w === "string")))
      );
    case "tool":
      return (
        typeof value.id === "string" &&
        typeof value.name === "string" &&
        typeof value.label === "string" &&
        typeof value.status === "string" &&
        TOOL_STATUSES.includes(value.status) &&
        typeof value.summary === "string" &&
        typeof value.changed_matrix === "boolean"
      );
    case "confirmation":
      return (
        typeof value.action_id === "string" &&
        typeof value.tool === "string" &&
        (value.risk === "costly" || value.risk === "destructive") &&
        typeof value.summary === "string" &&
        (value.plan === undefined || value.plan === null || isPlanPayload(value.plan))
      );
    case "error":
      return (
        typeof value.request_id === "string" &&
        typeof value.code === "string" &&
        typeof value.message === "string" &&
        typeof value.retryable === "boolean"
      );
    default:
      return false;
  }
}

const STORED_STATUSES: readonly string[] = ["streaming", "complete", "stopped", "error"];

function isStoredMessage(value: unknown): value is StoredMessage {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    (value.role === "user" || value.role === "assistant") &&
    typeof value.content === "string" &&
    typeof value.status === "string" &&
    STORED_STATUSES.includes(value.status)
  );
}

export function isConversationView(value: unknown): value is ConversationView {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    Array.isArray(value.messages) &&
    value.messages.every(isStoredMessage)
  );
}
