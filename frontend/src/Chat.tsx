import { ArrowUp, FileText, Loader2, Paperclip, Plus, ShieldAlert, Square, X } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Markdown } from "@/components/Markdown";
import { Select, Textarea } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { ApiError, fetchConversation, streamAnswer, streamChat } from "./api";
import { cancelAction, confirmAction, fetchPendingActions } from "./agentApi";
import PlanCard from "./PlanCard";
import { planOfArgs, type PlanPayload } from "./planModel";
import {
  STATUS_TEXT,
  assignDocument,
  getDocument,
  getIntake,
  isWorking,
  uploadDocument,
  type DocumentInfo,
  type DocumentStatus,
  type IntakeView,
} from "./documentsApi";
import { awaitsIntake, describeIntake } from "./intakeModel";
import { isFinished } from "./matrixModel";
import { createEntity, getRun } from "./matrixApi";
import type { Tone } from "./matrixModel";
import type { Entity } from "./matrixTypes";
import { type UiTool } from "./stepsModel";
import ToolSteps from "./ToolSteps";
import type { StoredMessage, StreamEvent, Usage } from "./types";

// What the screen keeps for each message.
interface UiMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "streaming" | "done" | "error";
  usage?: Usage;
  requestId?: string;
  errorText?: string;
  tools?: UiTool[]; // what the agent did while answering
  warnings?: string[];
}

// Something the agent will not do without a yes. `waiting` until the person answers.
interface Card {
  id: string; // the action id
  risk: "costly" | "destructive";
  summary: string;
  plan?: PlanPayload | null; // a column plan: shown as columns, not as a sentence
  state: "waiting" | "working" | "done" | "failed";
  note?: string;
}

// A file the person attached: uploading, then being read by the server, then ready (or not).
interface Attachment {
  key: string;
  name: string;
  doc?: DocumentInfo;
  error?: string;
  intake?: IntakeView | null; // which row the file went to: undefined until intake has decided
  tries?: number; // how many times intake was asked; after a while the chip stops asking
  placeError?: string; // a manual choice that failed
}

const FILL_WAIT_TICKS = 600; // up to 20 minutes at one look every 2 s
const INTAKE_TRIES = 40; // about a minute at one look every 1.5 s

function docTone(status: DocumentStatus): Tone {
  if (status === "parsed") return "green";
  if (status === "failed" || status === "needs_ocr") return "red";
  if (status === "needs_review") return "amber";
  return "gray";
}

interface ChatProps {
  workspaceId: string | null; // where attached documents go
  entities: Entity[]; // the rows a document can belong to
  onMatrixChanged: () => void; // the agent changed the grid: reload it
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

/** What intake did with one attached file and, when it was unsure, the choice of row. */
function IntakeLine({
  attachment,
  entities,
  onPlace,
  onCreate,
}: {
  attachment: Attachment;
  entities: Entity[];
  onPlace: (key: string, documentId: string, rowId: string) => Promise<void>;
  onCreate: (key: string, documentId: string, name: string) => Promise<void>;
}) {
  const doc = attachment.doc;
  const view = attachment.intake;
  if (doc === undefined || !view) return null;
  const line = describeIntake(view);
  const guess = view.detected.company ?? null;
  return (
    <div className="flex flex-col gap-1 pl-5">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={line.tone}>{line.text}</Badge>
        {view.detected.ticker && <span className="text-muted-foreground">{view.detected.ticker}</span>}
        {view.detected.doc_type && <span className="text-muted-foreground">· {view.detected.doc_type}</span>}
        {view.detected.period && <span className="text-muted-foreground">· {view.detected.period}</span>}
      </div>
      {line.needsChoice && (
        <div className="flex flex-wrap items-center gap-1">
          <Select
            className="h-7 w-auto max-w-40 text-xs"
            value=""
            onChange={(e) => e.target.value !== "" && void onPlace(attachment.key, doc.id, e.target.value)}
            aria-label="Choose the row for this file"
          >
            <option value="">Choose a row…</option>
            {entities.map((row) => (
              <option key={row.id} value={row.id}>
                {row.name}
              </option>
            ))}
          </Select>
          {guess !== null && (
            <Button size="sm" onClick={() => void onCreate(attachment.key, doc.id, guess)}>
              <Plus className="size-3" /> New row “{guess}”
            </Button>
          )}
        </div>
      )}
      {attachment.placeError && <span className="text-destructive">{attachment.placeError}</span>}
    </div>
  );
}

export default function Chat({ workspaceId, entities, onMatrixChanged }: ChatProps) {
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversationWorkspace, setConversationWorkspace] = useState<string | null>(null);
  const [cards, setCards] = useState<Card[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  // Start in the loading state only if there is a conversation to bring back.
  const [restoring, setRestoring] = useState<boolean>(() => readStoredId() !== null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const latest = useRef<Attachment[]>([]); // what the poll reads, so it never needs a state updater
  const [targetRow, setTargetRow] = useState(""); // "" = not tied to a row yet

  // On first load: bring back the conversation we were in, if the server still has it.
  useEffect(() => {
    const storedId = readStoredId();
    if (storedId === null) return;
    let cancelled = false;
    fetchConversation(storedId)
      .then((conversation) => {
        if (cancelled) return;
        setConversationId(conversation.id);
        setConversationWorkspace(conversation.workspace_id ?? null);
        setMessages(conversation.messages.map(toUiMessage));
        // Anything still waiting for a yes comes back as a card.
        fetchPendingActions(conversation.id)
          .then((found) => {
            if (cancelled) return;
            setCards(
              found.map((a) => ({
                id: a.id,
                risk: a.risk,
                summary: a.summary,
                plan: planOfArgs(a.args),
                state: "waiting",
              })),
            );
          })
          .catch(() => undefined);
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

  // Upload each chosen file to the current workspace (and row, if one is picked). The server answers
  // at once and reads the file in the background, so a poll below watches it until it is ready.
  async function attach(files: File[]): Promise<void> {
    if (workspaceId === null) return;
    const entityId = targetRow === "" ? null : targetRow;
    for (const file of files) {
      const key = crypto.randomUUID();
      setAttachments((prev) => [...prev, { key, name: file.name }]);
      try {
        const doc = await uploadDocument(file, workspaceId, entityId);
        setAttachments((prev) => prev.map((a) => (a.key === key ? { ...a, doc } : a)));
      } catch (err) {
        const error = err instanceof Error ? err.message : "Upload failed.";
        setAttachments((prev) => prev.map((a) => (a.key === key ? { ...a, error } : a)));
      }
    }
  }

  // While any attachment is still being read, ask the server again every 1.5 s.
  useEffect(() => {
    latest.current = attachments;
  }, [attachments]);
  const waiting = attachments.some(
    (a) =>
      a.doc !== undefined &&
      (isWorking(a.doc.status) || (awaitsIntake(a.doc.status, a.intake) && (a.tries ?? 0) < INTAKE_TRIES)),
  );
  useEffect(() => {
    if (!waiting) return;
    const timer = setInterval(() => {
      for (const a of latest.current) {
        if (a.doc === undefined) continue;
        const id = a.doc.id;
        if (isWorking(a.doc.status)) {
          getDocument(id)
            .then((doc) => setAttachments((prev) => prev.map((x) => (x.key === a.key ? { ...x, doc } : x))))
            .catch(() => undefined); // the next tick tries again
        } else if (awaitsIntake(a.doc.status, a.intake) && (a.tries ?? 0) < INTAKE_TRIES) {
          setAttachments((prev) => prev.map((x) => (x.key === a.key ? { ...x, tries: (x.tries ?? 0) + 1 } : x)));
          getIntake(id)
            .then((intake) => {
              if (intake === null) return; // not decided yet: ask again next tick
              setAttachments((prev) => prev.map((x) => (x.key === a.key ? { ...x, intake } : x)));
              if (intake.outcome !== "needs_review") onMatrixChanged(); // a row may have appeared
            })
            .catch(() => undefined);
        }
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [waiting, onMatrixChanged]);

  // The person's choice of row for a file intake was unsure about. It always wins over intake.
  async function placeFile(key: string, documentId: string, rowId: string): Promise<void> {
    try {
      await assignDocument(documentId, rowId);
      const intake = await getIntake(documentId);
      setAttachments((prev) => prev.map((x) => (x.key === key ? { ...x, intake, placeError: undefined } : x)));
      onMatrixChanged();
    } catch (err) {
      const placeError = err instanceof Error ? err.message : "Could not attach the file.";
      setAttachments((prev) => prev.map((x) => (x.key === key ? { ...x, placeError } : x)));
    }
  }

  async function createRowFor(key: string, documentId: string, name: string): Promise<void> {
    if (workspaceId === null) return;
    try {
      const row = await createEntity(workspaceId, name);
      await placeFile(key, documentId, row.id);
    } catch (err) {
      const placeError = err instanceof Error ? err.message : "Could not create the row.";
      setAttachments((prev) => prev.map((x) => (x.key === key ? { ...x, placeError } : x)));
    }
  }

  function forgetConversation(): void {
    setConversationId(null);
    writeStoredId(null);
  }

  // One place that turns a streamed event into what the screen shows. `assistantId` is the message
  // being written: a normal reply, or the answer written from a filled table.
  function handleEvent(assistantId: string, event: StreamEvent): void {
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
          warnings: event.warnings,
        }));
        break;
      case "tool": {
        const step: UiTool = {
          id: event.id,
          label: event.label,
          status: event.status,
          summary: event.summary,
        };
        // The same step arrives twice (running, then its end): update it in place.
        updateMessage(assistantId, (m) => {
          const tools = m.tools ?? [];
          const known = tools.some((t) => t.id === step.id);
          return { ...m, tools: known ? tools.map((t) => (t.id === step.id ? step : t)) : [...tools, step] };
        });
        if (event.changed_matrix) onMatrixChanged();
        break;
      }
      case "confirmation":
        setCards((prev) => [
          ...prev,
          {
              id: event.action_id,
              risk: event.risk,
              summary: event.summary,
              plan: event.plan ?? null,
              state: "waiting",
            },
        ]);
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

    // A conversation belongs to one workspace, and its tools act there. If the grid now shows a
    // different workspace, start a fresh conversation instead of acting on the wrong matrix.
    const switched =
      conversationId !== null &&
      workspaceId !== null &&
      conversationWorkspace !== null &&
      conversationWorkspace !== workspaceId;
    if (switched) {
      forgetConversation();
      setCards([]);
    }
    setMessages((prev) => [...(switched ? [] : prev), userMessage, assistantMessage]);
    setInput("");
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat({
        conversationId: switched ? null : conversationId,
        workspaceId,
        message: text, // only the new message: the server keeps the history
        signal: controller.signal,
        onStart: ({ conversationId: id }) => {
          // Remember the conversation as soon as the server names it, even if the stream
          // is cut short later.
          setConversationId(id);
          writeStoredId(id);
          if (switched || conversationId === null) setConversationWorkspace(workspaceId);
        },
        onEvent: (event) => handleEvent(assistantId, event),
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
    setCards([]);
    forgetConversation();
  }

  function setCard(id: string, change: Partial<Card>): void {
    setCards((prev) => prev.map((c) => (c.id === id ? { ...c, ...change } : c)));
  }

  // The person's answer to something the agent would not do alone. The server runs (or drops) the
  // exact action that was shown; this only sends the yes or the no.
  async function answer(card: Card, verb: "confirm" | "cancel"): Promise<void> {
    setCard(card.id, { state: "working" });
    try {
      const result = verb === "confirm" ? await confirmAction(card.id) : await cancelAction(card.id);
      setCard(card.id, { state: result.ok ? "done" : "failed", note: result.message });
      if (result.changed_matrix) onMatrixChanged();
      if (verb === "confirm" && result.ok && result.runId !== null) {
        void answerFromCells(card.id, result.runId);
      }
    } catch (err) {
      setCard(card.id, { state: "failed", note: err instanceof Error ? err.message : "Something went wrong." });
    }
  }

  // After a confirmed plan started a fill: wait for it to end (the grid fills in meanwhile), then
  // stream the answer, written from the filled cells, as a new message.
  async function answerFromCells(actionId: string, runId: string): Promise<void> {
    const ws = conversationWorkspace ?? workspaceId;
    if (ws === null) return;
    const assistantId = crypto.randomUUID();
    const fillStep = (status: UiTool["status"], summary: string): UiTool => ({
      id: "fill",
      label: "Filling the table",
      status,
      summary,
    });
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "", status: "streaming", tools: [fillStep("running", "")] },
    ]);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      for (let tick = 0; tick < FILL_WAIT_TICKS && !controller.signal.aborted; tick++) {
        const run = await getRun(ws, runId);
        if (isFinished(run.status)) break;
        updateMessage(assistantId, (m) => ({
          ...m,
          tools: [fillStep("running", `${run.done} of ${run.total} cells`)],
        }));
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
      if (controller.signal.aborted) throw new DOMException("stopped", "AbortError");
      updateMessage(assistantId, (m) => ({ ...m, tools: [fillStep("ok", "done")] }));
      onMatrixChanged();
      await streamAnswer({
        actionId,
        signal: controller.signal,
        onStart: () => undefined,
        onEvent: (event) => handleEvent(assistantId, event),
      });
    } catch (err) {
      if (controller.signal.aborted) {
        updateMessage(assistantId, (m) => ({ ...m, status: "done", errorText: "Stopped." }));
      } else {
        const message = err instanceof Error ? err.message : "Something went wrong.";
        updateMessage(assistantId, (m) => ({ ...m, status: "error", errorText: message }));
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>): void {
    // Enter sends; Shift+Enter adds a new line.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-muted/50 text-foreground">
      <header className="flex items-center justify-between border-b px-4 py-3">
        <h1 className="text-base font-semibold">Tessera</h1>
        <Button variant="ghost" size="sm" onClick={newChat} disabled={busy || restoring}>
          <Plus className="size-4" /> New chat
        </Button>
      </header>

      <main className="flex-1 overflow-y-auto px-4 py-4">
        {restoring && <p className="mt-10 text-center text-sm text-muted-foreground">Loading your conversation…</p>}
        {!restoring && messages.length === 0 && (
          <p className="mt-10 text-center text-sm text-muted-foreground">Ask something to start.</p>
        )}
        {messages.map((m) => (
          <div key={m.id} className={cn("mb-4 flex flex-col", m.role === "user" && "items-end")}>
            {m.tools && m.tools.length > 0 && (
              <ToolSteps tools={m.tools} working={m.status === "streaming" && m.content === ""} />
            )}
            <div
              className={cn(
                "max-w-[90%] break-words text-sm leading-6",
                m.role === "user" && "whitespace-pre-wrap rounded-md bg-border/70 px-3 py-2",
              )}
            >
              {m.role === "assistant" ? <Markdown>{m.content}</Markdown> : m.content}
              {m.status === "streaming" && <span className="animate-pulse">▍</span>}
            </div>
            {m.warnings?.map((w) => (
              <div key={w} className="mt-1 text-xs text-[#cb912f]">
                {w}
              </div>
            ))}
            {m.errorText && <div className="mt-1 text-sm text-destructive">{m.errorText}</div>}
            {m.usage && (
              <div className="mt-1 text-xs text-muted-foreground" title={m.requestId}>
                {m.usage.tokens_in} tokens in · {m.usage.tokens_out} tokens out
              </div>
            )}
          </div>
        ))}
        {cards.map((c) => (
          <div
            key={c.id}
            className={cn(
              "mb-3 rounded-md border bg-background p-3 text-sm",
              c.state === "waiting" && c.risk === "destructive" && "border-destructive/50",
            )}
          >
            {c.plan ? (
              <PlanCard plan={c.plan} />
            ) : (
              <>
                <div className="mb-1 flex items-center gap-1.5 font-medium">
                  <ShieldAlert className="size-4 text-[#cb912f]" />
                  {c.risk === "destructive" ? "Needs your confirmation" : "This will cost money"}
                </div>
                <p className="text-muted-foreground">{c.summary}</p>
              </>
            )}
            {c.note && <p className="mt-1">{c.note}</p>}
            {c.state === "waiting" || c.state === "working" ? (
              <div className="mt-2 flex gap-2">
                <Button
                  variant="default"
                  size="sm"
                  disabled={c.state === "working"}
                  onClick={() => void answer(c, "confirm")}
                >
                  {c.state === "working" ? "Working…" : c.plan ? "Create and fill" : "Confirm"}
                </Button>
                <Button size="sm" disabled={c.state === "working"} onClick={() => void answer(c, "cancel")}>
                  Cancel
                </Button>
              </div>
            ) : (
              <div className="mt-1 text-xs text-muted-foreground">
                {c.state === "done" ? "Done." : "Could not be done."}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      <footer className="px-3 pb-3">
        {attachments.length > 0 && (
          <ul className="mb-2 flex flex-col gap-1">
            {attachments.map((a) => (
              <li key={a.key} className="flex flex-col gap-1 rounded-md border bg-background px-2 py-1 text-xs">
                <div className="flex items-center gap-2">
                <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                <span className="truncate">{a.name}</span>
                <span className="ml-auto flex shrink-0 items-center gap-1">
                  {a.error !== undefined ? (
                    <Badge tone="red" title={a.error}>
                      {a.error}
                    </Badge>
                  ) : a.doc === undefined ? (
                    <Loader2 className="size-3.5 animate-spin text-muted-foreground" />
                  ) : (
                    <Badge tone={docTone(a.doc.status)} title={a.doc.status_detail ?? undefined}>
                      {a.doc.duplicate ? "Already uploaded" : STATUS_TEXT[a.doc.status]}
                    </Badge>
                  )}
                  <button
                    className="rounded p-0.5 text-muted-foreground hover:bg-muted"
                    onClick={() => setAttachments((prev) => prev.filter((x) => x.key !== a.key))}
                    aria-label={`Remove ${a.name} from this list`}
                    title="Remove from this list (the document stays uploaded)"
                  >
                    <X className="size-3" />
                  </button>
                </span>
                </div>
                {a.doc !== undefined && a.intake && (
                  <IntakeLine attachment={a} entities={entities} onPlace={placeFile} onCreate={createRowFor} />
                )}
              </li>
            ))}
          </ul>
        )}

        <div className="rounded-md border bg-background p-2 focus-within:outline-2 focus-within:outline-ring">
          <Textarea
            className="resize-none border-0 px-1 py-0.5 focus-visible:outline-0"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a question. Enter to send, Shift+Enter for a new line."
            rows={2}
          />
          <div className="mt-1 flex items-center gap-1.5">
            <input
              ref={fileRef}
              type="file"
              multiple
              hidden
              accept=".pdf,.docx,.xlsx,.pptx,.html,.htm,.csv,.txt,.md"
              onChange={(e) => {
                void attach(Array.from(e.target.files ?? []));
                e.target.value = ""; // so choosing the same file again still fires
              }}
            />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => fileRef.current?.click()}
              disabled={workspaceId === null}
              aria-label="Attach documents"
              title={workspaceId === null ? "Create a workspace first" : "Attach documents"}
            >
              <Paperclip className="size-4" />
            </Button>
            <Select
              className="h-7 w-auto max-w-44 text-xs"
              value={targetRow}
              onChange={(e) => setTargetRow(e.target.value)}
              disabled={workspaceId === null}
              aria-label="Which row the documents belong to"
              title="Answers for a row come only from that row's documents"
            >
              <option value="">Detect the row automatically</option>
              {entities.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.name}
                </option>
              ))}
            </Select>
            <span className="ml-auto" />
            {busy ? (
              <Button size="sm" onClick={stop}>
                <Square className="size-3" /> Stop
              </Button>
            ) : (
              <Button
                variant="default"
                size="icon"
                onClick={() => void send()}
                disabled={input.trim() === "" || restoring}
                aria-label="Send"
              >
                <ArrowUp className="size-4" />
              </Button>
            )}
          </div>
        </div>
      </footer>
    </div>
  );
}
