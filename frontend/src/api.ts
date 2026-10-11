import { parseSseChunk } from "./sse";
import { isStreamEvent, type ChatMessage, type StreamEvent } from "./types";

// Where the backend lives. Override with VITE_API_URL in frontend/.env.local if needed.
const API_URL: string = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * POST the conversation to /chat and call `onEvent` for every event as it arrives.
 *
 * Resolves when the stream ends normally. Rejects with:
 *  - ApiError        if the server rejected the request up front (e.g. 422)
 *  - an AbortError   if `signal` was aborted (the Stop button)
 *  - a plain Error   if the network failed or the stream ended without a final event
 */
export async function streamChat(
  messages: ChatMessage[],
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
    signal,
  });

  if (!response.ok) {
    throw new ApiError(response.status, `The server rejected the request (${response.status}).`);
  }
  if (response.body === null) {
    throw new Error("The server sent no response body.");
  }

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
      onEvent(parsed);
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
