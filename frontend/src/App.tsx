import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { ApiError, fetchConversation, streamChat } from "./api";
import type { StoredMessage, StreamEvent, Usage } from "./types";
import "./App.css";

// What the screen keeps for each message.
interface UiMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "streaming" | "done" | "error";
  usage?: Usage;
  requestId?: string;
  errorText?: string;
}

const STORAGE_KEY = "tessera.conversationId";

// localStorage can throw (private windows, blocked storage), so every use is wrapped.
function readStoredId(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function writeStoredId(id: string | null): void {
  try {
    if (id === null) localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, id);
  } catch {
    // not remembering the conversation is acceptable
  }
}

// A message as stored on the server -> a message as shown on screen.
function toUiMessage(m: StoredMessage): UiMessage {
  const failed = m.status === "error";
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    status: failed ? "error" : "done",
    errorText: failed ? "This answer failed." : m.status === "stopped" ? "Stopped." : undefined,
  };
}

export default function App() {
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  // Start in the loading state only if there is a conversation to bring back.
  const [restoring, setRestoring] = useState<boolean>(() => readStoredId() !== null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // On first load: bring back the conversation we were in, if the server still has it.
  useEffect(() => {
    const storedId = readStoredId();
    if (storedId === null) return;
    let cancelled = false;
    fetchConversation(storedId)
      .then((conversation) => {
        if (cancelled) return;
        setConversationId(conversation.id);
        setMessages(conversation.messages.map(toUiMessage));
      })
      .catch(() => {
        if (!cancelled) writeStoredId(null); // gone or unreachable: start fresh
      })
      .finally(() => {
        if (!cancelled) setRestoring(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Keep the newest text in view while the answer streams in.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  // Change one message without mutating the old array (React needs a new copy to re-render).
  function updateMessage(id: string, change: (m: UiMessage) => UiMessage): void {
    setMessages((prev) => prev.map((m) => (m.id === id ? change(m) : m)));
  }

  function forgetConversation(): void {
    setConversationId(null);
    writeStoredId(null);
  }

  async function send(): Promise<void> {
    const text = input.trim();
    if (text === "" || busy || restoring) return;

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
      await streamChat({
        conversationId,
        message: text, // only the new message: the server keeps the history
        signal: controller.signal,
        onStart: ({ conversationId: id }) => {
          // Remember the conversation as soon as the server names it, even if the stream
          // is cut short later.
          setConversationId(id);
          writeStoredId(id);
        },
        onEvent,
      });
    } catch (err) {
      if (controller.signal.aborted) {
        // The user pressed Stop: keep whatever text arrived.
        updateMessage(assistantId, (m) => ({ ...m, status: "done", errorText: "Stopped." }));
      } else if (err instanceof ApiError && err.status === 404) {
        forgetConversation(); // the server no longer has it; the next message starts a new one
        updateMessage(assistantId, (m) => ({
          ...m,
          status: "error",
          errorText: "That conversation no longer exists. Send your message again to start a new one.",
        }));
      } else {
        const message = err instanceof Error ? err.message : "Something went wrong.";
        updateMessage(assistantId, (m) => ({ ...m, status: "error", errorText: message }));
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  function stop(): void {
    abortRef.current?.abort();
  }

  function newChat(): void {
    setMessages([]);
    forgetConversation();
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
        <button onClick={newChat} disabled={busy || restoring}>
          New chat
        </button>
      </header>

      <main>
        {restoring && <p className="hint">Loading your conversation…</p>}
        {!restoring && messages.length === 0 && <p className="hint">Ask something to start.</p>}
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
          <button onClick={() => void send()} disabled={input.trim() === "" || restoring}>
            Send
          </button>
        )}
      </footer>
    </div>
  );
}
