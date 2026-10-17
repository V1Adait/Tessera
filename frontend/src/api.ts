import { isConversationSummary, type ConversationSummary } from "./historyModel";
import { parseSseChunk } from "./sse";
import {
  isConversationView,
  isStreamEvent,
  type ConversationView,
  type StreamEvent,
} from "./types";

// Where the backend lives. Override with VITE_API_URL in frontend/.env.local if needed.
export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export interface ChatCall {
  conversationId: string | null; // null starts a new conversation
  workspaceId?: string | null; // which workspace a NEW conversation belongs to (its tools act there)
  message: string; // ONE new message: the server keeps the history
  onStart: (start: { conversationId: string; requestId: string }) => void;
  onEvent: (event: StreamEvent) => void;
  signal?: AbortSignal;
}

/**
 * POST one message to /chat and call `onEvent` for every event as it arrives.
 * `onStart` fires once, as soon as the server answers, with the conversation id (read from the
 * X-Conversation-ID header) so the caller can remember it even if the stream is cut short.
 *
 * Resolves when the stream ends normally. Rejects with:
 *  - ApiError        if the server rejected the request up front (404 unknown conversation,
 *                    422 malformed body, 503 no database)
 *  - an AbortError   if `signal` was aborted (the Stop button)
 *  - a plain Error   if the network failed or the stream ended without a final event
 */
export async function streamChat(call: ChatCall): Promise<void> {
  await streamRequest("/chat", {
    ...call,
    body: {
      conversation_id: call.conversationId,
      message: call.message,
      // Only a NEW conversation names its workspace: the server rejects both ids together.
      ...(call.conversationId === null && call.workspaceId ? { workspace_id: call.workspaceId } : {}),
    },
  });
}

export interface AnswerCall {
  actionId: string; // the confirmed plan whose filled cells the answer is written from
  onStart: (start: { conversationId: string; requestId: string }) => void;
  onEvent: (event: StreamEvent) => void;
  signal?: AbortSignal;
}

/** Stream the answer to a confirmed plan's question, written from the cells it filled. The server
 * answers 409 (with its reason) while the table is still being filled. */
export async function streamAnswer(call: AnswerCall): Promise<void> {
  await streamRequest(`/actions/${call.actionId}/answer`, { ...call, body: {} });
}

interface StreamRequest {
  body: unknown;
  onStart: (start: { conversationId: string; requestId: string }) => void;
  onEvent: (event: StreamEvent) => void;
  signal?: AbortSignal;
}

async function streamRequest(path: string, call: StreamRequest): Promise<void> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(call.body),
    signal: call.signal,
  });

  if (!response.ok) {
    let reason = `The server rejected the request (${response.status}).`;
    try {
      const problem: unknown = await response.json();
      if (typeof problem === "object" && problem !== null && "detail" in problem) {
        const detail = (problem as { detail: unknown }).detail;
        if (typeof detail === "string") reason = detail;
      }
    } catch {
      // keep the generic text
    }
    throw new ApiError(response.status, reason);
  }
  if (response.body === null) {
    throw new Error("The server sent no response body.");
  }
  const conversationId = response.headers.get("X-Conversation-ID");
  const requestId = response.headers.get("X-Request-ID");
  if (conversationId === null || requestId === null) {
    throw new Error("The server did not say which conversation this belongs to.");
  }
  call.onStart({ conversationId, requestId });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawFinalEvent = false;

  const handlePayloads = (payloads: string[]): void => {
    for (const payload of payloads) {
      let parsed: unknown;
      try {
        parsed = JSON.parse(payload);
      } catch {
        console.warn("Ignoring an unreadable event:", payload);
        continue;
      }
      if (!isStreamEvent(parsed)) {
        console.warn("Ignoring an unexpected event shape:", parsed);
        continue;
      }
      if (parsed.type === "done" || parsed.type === "error") sawFinalEvent = true;
      call.onEvent(parsed);
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // `stream: true` keeps a multi-byte character that was split across chunks intact.
    buffer += decoder.decode(value, { stream: true });
    const { payloads, rest } = parseSseChunk(buffer);
    buffer = rest;
    handlePayloads(payloads);
  }

  // Flush anything the decoder was holding, and any last complete message.
  buffer += decoder.decode();
  handlePayloads(parseSseChunk(buffer + "\n\n").payloads);

  if (!sawFinalEvent) {
    throw new Error("The connection ended before the answer was complete.");
  }
}

/** Load a stored conversation (to restore the screen after a refresh). */
export async function fetchConversation(id: string): Promise<ConversationView> {
  const response = await fetch(`${API_URL}/conversations/${id}`);
  if (!response.ok) {
    throw new ApiError(response.status, `Could not load the conversation (${response.status}).`);
  }
  const data: unknown = await response.json();
  if (!isConversationView(data)) {
    throw new Error("The server sent a conversation in an unexpected shape.");
  }
  return data;
}

/** The chat history of a workspace, newest first. */
export async function fetchConversations(workspaceId: string): Promise<ConversationSummary[]> {
  const response = await fetch(`${API_URL}/workspaces/${workspaceId}/conversations`);
  if (!response.ok) {
    throw new ApiError(response.status, `Could not load the history (${response.status}).`);
  }
  const data: unknown = await response.json();
  if (!Array.isArray(data) || !data.every(isConversationSummary)) {
    throw new Error("The server sent the history in an unexpected shape.");
  }
  return data;
}

/** Delete a conversation and its messages. Cannot be undone: the caller asks first. */
export async function deleteConversation(id: string): Promise<void> {
  const response = await fetch(`${API_URL}/conversations/${id}`, { method: "DELETE" });
  if (response.status === 204) return;
  throw new ApiError(response.status, `Could not delete the conversation (${response.status}).`);
}
