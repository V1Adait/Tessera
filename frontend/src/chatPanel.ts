// Whether the chat panel is open or folded away, so the table can use the whole screen.
// The choice is remembered, and defaults to open.

const KEY = "tessera.chatOpen";

// localStorage can throw (private windows, blocked storage): the panel is then simply open.
export function readChatOpen(): boolean {
  try {
    return localStorage.getItem(KEY) !== "closed";
  } catch {
    return true;
  }
}

export function writeChatOpen(open: boolean): void {
  try {
    localStorage.setItem(KEY, open ? "open" : "closed");
  } catch {
    // not remembering the choice is acceptable
  }
}

/** Is this key press the shortcut that folds or opens the chat (Ctrl or Cmd + backslash)? */
export function isChatToggleKey(e: { key: string; ctrlKey: boolean; metaKey: boolean; altKey: boolean }): boolean {
  return e.key === "\\" && (e.ctrlKey || e.metaKey) && !e.altKey;
}
