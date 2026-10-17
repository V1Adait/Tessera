import { useCallback, useEffect, useState } from "react";
import { getRun, listRuns, planRun, startRun, stopRun } from "./matrixApi";
import { isFinished } from "./matrixModel";
import type { Plan, Run, RunScope } from "./matrixTypes";

// One run at a time per workspace, so the screen needs only one of these states.
export type RunPhase =
  | { kind: "idle" }
  | { kind: "working"; text: string }
  | { kind: "confirm"; scope: RunScope; message: string; plan: Plan }
  | { kind: "running"; run: Run }
  | { kind: "finished"; run: Run }
  | { kind: "notice"; text: string }
  | { kind: "error"; text: string };

const POLL_MS = 1000;

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong.";
}

/**
 * Plan -> (confirm) -> run -> watch. `onTick` fires after every poll so the grid can reload its
 * cells while a run is filling them.
 */
export function useRunner(workspaceId: string | null, onTick: () => void) {
  const [phase, setPhase] = useState<RunPhase>({ kind: "idle" });

  // A new workspace starts clean; if the server already has a run going there, pick it up.
  useEffect(() => {
    setPhase({ kind: "idle" });
    if (workspaceId === null) return;
    let cancelled = false;
    listRuns(workspaceId)
      .then((runs) => {
        const live = runs.find((r) => !isFinished(r.status));
        if (!cancelled && live) setPhase({ kind: "running", run: live });
      })
      .catch(() => undefined); // not being able to look is fine: the grid still works
    return () => {
      cancelled = true;
    };
  }, [workspaceId]);

  // While a run is going, ask how it is doing once a second.
  const runningId = phase.kind === "running" ? phase.run.id : null;
  useEffect(() => {
    if (workspaceId === null || runningId === null) return;
    const timer = setInterval(() => {
      getRun(workspaceId, runningId)
        .then((run) => {
          setPhase(isFinished(run.status) ? { kind: "finished", run } : { kind: "running", run });
          onTick();
        })
        .catch((err: unknown) => setPhase({ kind: "error", text: messageOf(err) }));
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [workspaceId, runningId, onTick]);

  const start = useCallback(
    async (scope: RunScope, confirmed: boolean) => {
      if (workspaceId === null) return;
      try {
        const result = await startRun(workspaceId, scope, confirmed);
        if (result.kind === "needs_confirmation") {
          setPhase({ kind: "confirm", scope, message: result.message, plan: result.plan });
        } else {
          setPhase({ kind: "running", run: result.run });
        }
      } catch (err) {
        setPhase({ kind: "error", text: messageOf(err) });
      }
    },
    [workspaceId],
  );

  /** Look at what would run. Nothing to do -> say so. Otherwise start (or ask first). */
  const begin = useCallback(
    async (scope: RunScope) => {
      if (workspaceId === null) return;
      setPhase({ kind: "working", text: "Planning…" });
      try {
        const plan = await planRun(workspaceId, scope);
        if (plan.cells === 0) {
          const pinned =
            plan.skipped_pinned > 0 ? ` ${plan.skipped_pinned} cell(s) set by a person were left alone.` : "";
          setPhase({ kind: "notice", text: `Nothing to fill: every cell here already has an answer.${pinned}` });
          return;
        }
        await start(scope, false);
      } catch (err) {
        setPhase({ kind: "error", text: messageOf(err) });
      }
    },
    [workspaceId, start],
  );

  const confirm = useCallback(async () => {
    if (phase.kind === "confirm") await start(phase.scope, true);
  }, [phase, start]);

  const stop = useCallback(async () => {
    if (workspaceId === null || phase.kind !== "running") return;
    try {
      setPhase({ kind: "running", run: await stopRun(workspaceId, phase.run.id) });
    } catch (err) {
      setPhase({ kind: "error", text: messageOf(err) });
    }
  }, [workspaceId, phase]);

  const dismiss = useCallback(() => setPhase({ kind: "idle" }), []);

  return { phase, begin, confirm, stop, dismiss };
}
