import { PanelLeftOpen } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import Chat from "./Chat";
import { isChatToggleKey, readChatOpen, writeChatOpen } from "./chatPanel";
import Matrix from "./Matrix";
import type { Entity } from "./matrixTypes";

// Chat on the left, the matrix on the right. The matrix owns which workspace is open; it tells the
// chat, so documents attached in the chat land in the same workspace (and row) the grid shows.
//
// The chat can be folded to a thin rail so the table fills the screen. It stays MOUNTED while folded
// (only hidden), so an answer that is still being written, and the whole conversation, are not lost.
export default function App() {
  // Bumped when the agent changes the matrix, so the grid knows to reload.
  const [refreshSignal, setRefreshSignal] = useState(0);
  const [context, setContext] = useState<{ workspaceId: string | null; entities: Entity[] }>({
    workspaceId: null,
    entities: [],
  });
  const [chatOpen, setChatOpenState] = useState<boolean>(() => readChatOpen());
  const [unread, setUnread] = useState(false); // an answer finished while the chat was folded

  // The chat's callbacks are held by a stream that outlives a render, so they read the CURRENT state
  // from a ref: a callback that captured `chatOpen` when the message was sent would be out of date.
  const openRef = useRef(chatOpen);
  useEffect(() => {
    openRef.current = chatOpen;
  }, [chatOpen]);

  // Stable identity: the matrix reports in an effect that lists this function, so a new function on
  // every render would make it report again and again.
  const onContext = useCallback(
    (workspaceId: string | null, entities: Entity[]) => setContext({ workspaceId, entities }),
    [],
  );

  const onMatrixChanged = useCallback(() => setRefreshSignal((n) => n + 1), []);

  const setChatOpen = useCallback((open: boolean) => {
    openRef.current = open;
    setChatOpenState(open);
    writeChatOpen(open);
    if (open) setUnread(false);
  }, []);

  // A yes is needed (a confirmation card): it must never wait unseen behind a folded panel.
  const onNeedsAttention = useCallback(() => {
    if (!openRef.current) setChatOpen(true);
  }, [setChatOpen]);

  const onActivity = useCallback(() => {
    if (!openRef.current) setUnread(true);
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent): void {
      if (!isChatToggleKey(e)) return;
      e.preventDefault();
      setChatOpen(!openRef.current);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setChatOpen]);

  return (
    <div className="shell">
      <aside className={cn("shell-chat", !chatOpen && "shell-chat-collapsed")}>
        <div className={chatOpen ? "h-full" : "hidden"}>
          <Chat
            workspaceId={context.workspaceId}
            entities={context.entities}
            onMatrixChanged={onMatrixChanged}
            onCollapse={() => setChatOpen(false)}
            onNeedsAttention={onNeedsAttention}
            onActivity={onActivity}
          />
        </div>
        {!chatOpen && (
          <div className="flex h-full flex-col items-center gap-3 pt-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setChatOpen(true)}
              aria-label="Open the chat"
              title="Open the chat (Ctrl+\)"
            >
              <PanelLeftOpen className="size-4" />
            </Button>
            {unread && (
              <span className="size-2 rounded-full bg-primary" title="The chat has a new reply" role="status" />
            )}
          </div>
        )}
      </aside>
      <Matrix onContext={onContext} refreshSignal={refreshSignal} />
    </div>
  );
}
