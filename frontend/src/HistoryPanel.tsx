import { Trash2 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { historyTitle, messageCount, timeAgo, type ConversationSummary } from "./historyModel";

interface Props {
  items: ConversationSummary[] | null; // null while loading
  activeId: string | null; // the conversation that is open now
  error: string | null;
  disabled: boolean; // an answer is being written: switching now would lose it
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}

/** The earlier chats of this workspace. Opening one brings its messages back; deleting needs a second
 * click, because it cannot be undone. */
export default function HistoryPanel({ items, activeId, error, disabled, onOpen, onDelete }: Props) {
  const [confirming, setConfirming] = useState<string | null>(null);
  const [now] = useState(() => new Date()); // read once, when the list opens: not on every render

  return (
    <div className="max-h-64 overflow-y-auto border-b bg-background">
      {error && <div className="px-4 py-2 text-sm text-destructive">{error}</div>}
      {items === null && !error && <p className="px-4 py-3 text-sm text-muted-foreground">Loading…</p>}
      {items !== null && items.length === 0 && (
        <p className="px-4 py-3 text-sm text-muted-foreground">No earlier chats in this workspace yet.</p>
      )}
      <ul>
        {items?.map((c) => (
          <li key={c.id} className={cn("group flex items-center gap-1 px-2", c.id === activeId && "bg-muted")}>
            <button
              className="flex min-w-0 flex-1 flex-col px-2 py-2 text-left hover:text-foreground disabled:opacity-50"
              disabled={disabled}
              onClick={() => onOpen(c.id)}
              title={disabled ? "Wait for the answer to finish" : "Open this chat"}
            >
              <span className="truncate text-sm font-medium">{historyTitle(c)}</span>
              <span className="text-xs text-muted-foreground">
                {timeAgo(c.updated_at, now)} · {messageCount(c.message_count)}
              </span>
            </button>
            {confirming === c.id ? (
              <span className="flex shrink-0 items-center gap-1">
                <Button
                  size="sm"
                  variant="default"
                  className="bg-destructive hover:bg-destructive/90"
                  onClick={() => {
                    setConfirming(null);
                    onDelete(c.id);
                  }}
                >
                  Delete
                </Button>
                <Button size="sm" onClick={() => setConfirming(null)}>
                  Keep
                </Button>
              </span>
            ) : (
              <Button
                variant="ghost"
                size="icon"
                className="size-6 shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                onClick={() => setConfirming(c.id)}
                aria-label={`Delete the chat ${historyTitle(c)}`}
                title="Delete this chat"
              >
                <Trash2 className="size-3.5" />
              </Button>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
