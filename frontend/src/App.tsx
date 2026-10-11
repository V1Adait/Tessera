import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { ApiError, streamChat } from "./api";
import type { ChatMessage, StreamEvent, Usage } from "./types";
import "./App.css";

// What the screen keeps for each message. The backend only needs `role` and `content`.
interface UiMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "streaming" | "done" | "error";
  usage?: Usage;
  requestId?: string;
  errorText?: string;
}

export default function App() {
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Keep the newest text in view while the answer streams in.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  // Change one message without mutating the old array (React needs a new copy to re-render).
  function updateMessage(id: string, change: (m: UiMessage) => UiMessage): void {
    setMessages((prev) => prev.map((m) => (m.id === id ? change(m) : m)));
  }

  async function send(): Promise<void> {
    const text = input.trim();
    if (text === "" || busy) return;

    const userMessage: UiMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      status: "done",
    };
    const assistantId = crypto.randomUUID();
    const assistantMessage: UiMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      status: "streaming",
    };

    // The conversation we send: everything so far that has real content, plus the new message.
    const history: ChatMessage[] = [...messages, userMessage]
      .filter((m) => m.content !== "")
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;

    function onEvent(event: StreamEvent): void {
      switch (event.type) {
        case "delta":
          updateMessage(assistantId, (m) => ({ ...m, content: m.content + event.text }));
          break;
        case "done":
          updateMessage(assistantId, (m) => ({
            ...m,
            status: "done",
            usage: event.usage,
            requestId: event.request_id,
          }));
          break;
        case "error":
          updateMessage(assistantId, (m) => ({
            ...m,
            status: "error",
            errorText: `${event.message} (${event.code}${event.retryable ? ", you can retry" : ""})`,
            requestId: event.request_id,
          }));
          break;
      }
    }

    try {
      await streamChat(history, onEvent, controller.signal);
    } catch (err) {
      if (controller.signal.aborted) {
        // The user pressed Stop: keep whatever text arrived.
        updateMessage(assistantId, (m) => ({ ...m, status: "done", errorText: "Stopped." }));
      } else {
        const text =
          err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : "Something went wrong.";
        updateMessage(assistantId, (m) => ({ ...m, status: "error", errorText: text }));
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  function stop(): void {
    abortRef.current?.abort();
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>): void {
    // Enter sends; Shift+Enter adds a new line.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Tessera</h1>
        <button onClick={() => setMessages([])} disabled={busy || messages.length === 0}>
          Clear
        </button>
      </header>

      <main>
        {messages.length === 0 && <p className="hint">Ask something to start.</p>}
        {messages.map((m) => (
          <div key={m.id} className={`message ${m.role}`}>
            <div className="role">{m.role}</div>
            <div className="content">
              {m.content}
              {m.status === "streaming" && <span className="cursor">▍</span>}
            </div>
            {m.errorText && <div className="error">{m.errorText}</div>}
            {m.usage && (
              <div className="meta" title={m.requestId}>
                {m.usage.tokens_in} tokens in · {m.usage.tokens_out} tokens out
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      <footer>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a message. Enter to send, Shift+Enter for a new line."
          rows={2}
        />
        {busy ? (
          <button onClick={stop}>Stop</button>
        ) : (
          <button onClick={() => void send()} disabled={input.trim() === ""}>
            Send
          </button>
        )}
      </footer>
    </div>
  );
}
