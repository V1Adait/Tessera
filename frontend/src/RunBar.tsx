import { Button } from "@/components/ui/button";
import { describeEstimate, formatUsd, runLabel, runPercent } from "./matrixModel";
import type { RunPhase } from "./useRunner";

interface Props {
  phase: RunPhase;
  onConfirm: () => void;
  onStop: () => void;
  onDismiss: () => void;
}

const BAR = "flex items-center justify-between gap-3 border-t px-5 py-2 text-sm";
const MUTED = "text-muted-foreground";

/** The strip under the grid: what a run will do, how it is going, how it ended. */
export default function RunBar({ phase, onConfirm, onStop, onDismiss }: Props) {
  switch (phase.kind) {
    case "idle":
      return null;
    case "working":
      return <div className={BAR}>{phase.text}</div>;
    case "notice":
      return (
        <div className={BAR}>
          <span>{phase.text}</span>
          <Button variant="ghost" size="sm" onClick={onDismiss}>
            Dismiss
          </Button>
        </div>
      );
    case "error":
      return (
        <div className={`${BAR} bg-[#ffe2dd] text-[#5d1715]`}>
          <span>{phase.text}</span>
          <Button variant="ghost" size="sm" onClick={onDismiss}>
            Dismiss
          </Button>
        </div>
      );
    case "confirm": {
      const { plan } = phase;
      return (
        <div className={`${BAR} bg-muted`}>
          <div>
            <strong>{plan.cells} cells</strong> will be filled, {describeEstimate(plan.estimate)}. It
            stops by itself at {formatUsd(plan.cap_usd)}.
            <div className={MUTED}>{phase.message}</div>
            {plan.warnings.map((w) => (
              <div key={w} className={MUTED}>
                {w}
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <Button variant="default" size="sm" onClick={onConfirm}>
              Run
            </Button>
            <Button size="sm" onClick={onDismiss}>
              Cancel
            </Button>
          </div>
        </div>
      );
    }
    case "running":
    case "finished": {
      const { run } = phase;
      const running = phase.kind === "running";
      return (
        <div className={BAR}>
          <div className="flex-1">
            <span>{runLabel(run)}</span>
            {run.failed > 0 && <span className={MUTED}> · {run.failed} failed</span>}
            {run.skipped > 0 && <span className={MUTED}> · {run.skipped} skipped</span>}
            <span className={MUTED}>
              {" "}
              · spent {formatUsd(run.spent_usd)}
              {run.unknown_cost_calls > 0 ? " (at least: some calls had no known cost)" : ""}
            </span>
            {running && (
              <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-border" aria-label="progress">
                <div className="h-full bg-primary" style={{ width: `${runPercent(run)}%` }} />
              </div>
            )}
          </div>
          {running ? (
            <Button size="sm" onClick={onStop}>
              Stop
            </Button>
          ) : (
            <Button variant="ghost" size="sm" onClick={onDismiss}>
              Dismiss
            </Button>
          )}
        </div>
      );
    }
  }
}
