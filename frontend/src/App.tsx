import { useCallback, useState } from "react";
import Chat from "./Chat";
import Matrix from "./Matrix";
import type { Entity } from "./matrixTypes";

// Chat on the left, the matrix on the right. The matrix owns which workspace is open; it tells the
// chat, so documents attached in the chat land in the same workspace (and row) the grid shows.
// Beyond that they are independent: the chat will drive the matrix when the agent tools are built.
export default function App() {
  // Bumped when the agent changes the matrix, so the grid knows to reload.
  const [refreshSignal, setRefreshSignal] = useState(0);
  const [context, setContext] = useState<{ workspaceId: string | null; entities: Entity[] }>({
    workspaceId: null,
    entities: [],
  });

  // Stable identity: the matrix reports in an effect that lists this function, so a new function on
  // every render would make it report again and again.
  const onContext = useCallback(
    (workspaceId: string | null, entities: Entity[]) => setContext({ workspaceId, entities }),
    [],
  );

  const onMatrixChanged = useCallback(() => setRefreshSignal((n) => n + 1), []);

  return (
    <div className="shell">
      <aside className="shell-chat">
        <Chat
          workspaceId={context.workspaceId}
          entities={context.entities}
          onMatrixChanged={onMatrixChanged}
        />
      </aside>
      <Matrix onContext={onContext} refreshSignal={refreshSignal} />
    </div>
  );
}
