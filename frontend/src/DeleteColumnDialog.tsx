import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { deleteWarning, type ColumnImpact } from "./matrixModel";
import type { Column } from "./matrixTypes";

interface Props {
  column: Column;
  impact: ColumnImpact; // what removing it would throw away
  busy: boolean;
  error: string | null; // why the server refused, if it did
  blockedReason: string | null; // known in advance (a fill is going): the button is off and says why
  onConfirm: () => void;
  onClose: () => void;
}

/** The yes that removing a column needs. It says exactly what is lost, and nothing happens until the
 * person presses the red button: removing cannot be undone. */
export default function DeleteColumnDialog({ column, impact, busy, error, blockedReason, onConfirm, onClose }: Props) {
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogTitle className="text-lg font-semibold">Remove this column?</DialogTitle>
        <DialogDescription className="text-sm text-muted-foreground">
          {deleteWarning(column.name, impact)}
        </DialogDescription>

        {blockedReason && (
          <div className="rounded-md bg-[#fdecc8] px-3 py-2 text-sm text-[#402c1b]">{blockedReason}</div>
        )}
        {error && <div className="rounded-md bg-[#ffe2dd] px-3 py-2 text-sm text-[#5d1715]">{error}</div>}

        <div className="flex gap-2 pt-1">
          <Button
            variant="default"
            className="bg-destructive hover:bg-destructive/90"
            disabled={busy || blockedReason !== null}
            onClick={onConfirm}
          >
            {busy ? "Removing…" : "Remove column"}
          </Button>
          <Button onClick={onClose}>Keep it</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
