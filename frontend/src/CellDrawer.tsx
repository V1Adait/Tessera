import { Pin } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import {
  CHECK_LABEL,
  CHECK_TONE,
  STATUS_LABEL,
  derivationFormula,
  evidenceWhere,
  statusTone,
} from "./matrixModel";
import type { Cell, Column, Entity } from "./matrixTypes";

interface Props {
  entity: Entity;
  column: Column;
  cell: Cell | undefined;
  busy: boolean;
  error: string | null;
  onFill: (refresh: boolean) => void;
  onClose: () => void;
}

/** One cell in full: its value, why it has that status, and the exact quotes behind it. */
export default function CellDrawer({ entity, column, cell, busy, error, onFill, onClose }: Props) {
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent side="right">
        <div>
          <div className="text-sm text-muted-foreground">{entity.name}</div>
          <DialogTitle className="text-lg font-semibold">{column.name}</DialogTitle>
        </div>
        <DialogDescription className="text-sm text-muted-foreground">{column.prompt}</DialogDescription>

        {cell === undefined ? (
          <p className="text-sm text-muted-foreground">Not filled yet.</p>
        ) : (
          <>
            <div className="text-xl">{cell.display_value ?? "—"}</div>
            <div className="flex flex-wrap items-center gap-1.5">
              <Badge tone={statusTone(cell.status)}>{STATUS_LABEL[cell.status]}</Badge>
              {cell.pinned && (
                <Badge>
                  <Pin className="mr-1 inline size-3" />
                  Set by a person
                </Badge>
              )}
            </div>
            {cell.stale && (
              <p className="text-sm text-muted-foreground">
                The question changed after this was filled. Fill it again to update.
              </p>
            )}
            {cell.status_detail && <p className="text-sm">{cell.status_detail}</p>}
            {cell.period && (
              <p className="text-sm text-muted-foreground">
                Period: <span className="text-foreground">{cell.period}</span>
              </p>
            )}

            {cell.derivation && (
              <div className="rounded-md border bg-muted/50 p-2 text-sm">
                <div className="font-medium">Computed in code, not read from a document</div>
                <div className="mt-0.5 text-muted-foreground">{derivationFormula(cell.derivation)}</div>
                <ul className="mt-1 flex flex-col gap-0.5 text-xs text-muted-foreground">
                  {cell.derivation.inputs.map((i) => (
                    <li key={i.column}>
                      {i.column}: {i.display ?? "—"}
                      {i.period ? ` (${i.period})` : ""}
                    </li>
                  ))}
                </ul>
                {cell.derivation.notes.map((n) => (
                  <p key={n} className="mt-1 text-xs text-[#cb912f]">
                    {n}
                  </p>
                ))}
              </div>
            )}

            {cell.checks && cell.checks.length > 0 && (
              <div className="flex flex-col gap-1.5">
                <h3 className="text-sm font-semibold">Tie-outs</h3>
                {cell.checks.map((c) => (
                  <div key={c.id} className="text-sm">
                    <Badge tone={CHECK_TONE[c.status]}>{CHECK_LABEL[c.status]}</Badge>{" "}
                    <span className="font-medium">{c.name}</span>
                    <p className="text-xs text-muted-foreground">{c.detail}</p>
                  </div>
                ))}
              </div>
            )}

            {cell.evidence.length > 0 && (
              <h3 className="text-sm font-semibold">
                {cell.derivation ? "Evidence (the quotes behind its inputs)" : "Evidence"}
              </h3>
            )}
            {cell.evidence.map((e) => (
              <blockquote key={e.id} className="border-l-2 border-primary/40 pl-3 text-sm">
                “{e.quote}”
                <footer className="mt-1 text-xs text-muted-foreground">
                  {[evidenceWhere(e), e.exact_range ? "" : "location approximate"]
                    .filter((s) => s !== "")
                    .join(" · ") || "from the row's documents"}
                </footer>
              </blockquote>
            ))}
            {cell.model && <div className="text-xs text-muted-foreground">Filled by {cell.model}</div>}
          </>
        )}

        {error && <div className="rounded-md bg-[#ffe2dd] px-3 py-2 text-sm text-[#5d1715]">{error}</div>}

        <div className="mt-auto flex gap-2 pt-2">
          <Button variant="default" disabled={busy} onClick={() => onFill(false)}>
            {busy ? "Working…" : cell === undefined ? "Fill this cell" : "Fill again"}
          </Button>
          {cell !== undefined && (
            <Button disabled={busy} onClick={() => onFill(true)} title="Ignore the saved model reply and ask again">
              Ask the model again
            </Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
