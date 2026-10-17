import { useState, type FormEvent, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Input, Select, Textarea } from "@/components/ui/input";
import { TYPE_LABEL, columnProblem, splitList } from "./matrixModel";
import { COLUMN_TYPES, type Column, type ColumnDraft, type ColumnType } from "./matrixTypes";

interface Props {
  column: Column | null; // null = a new column
  busy: boolean;
  error: string | null;
  onSave: (draft: ColumnDraft) => void;
  onClose: () => void;
  onDelete?: () => void; // only for an existing column: opens the confirmation to remove it
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span>{label}</span>
      {children}
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </label>
  );
}

/** Create or change a column: its name, the question it asks, and what kind of answer it takes. */
export default function ColumnEditor({ column, busy, error, onSave, onClose, onDelete }: Props) {
  const [name, setName] = useState(column?.name ?? "");
  const [prompt, setPrompt] = useState(column?.prompt ?? "");
  const [type, setType] = useState<ColumnType>(column?.type ?? "text");
  const [options, setOptions] = useState((column?.allowed_values ?? []).join(", "));
  const [applicability, setApplicability] = useState(column?.applicability ?? "");
  const [evidenceRequired, setEvidenceRequired] = useState(column?.evidence_required ?? true);
  const [problem, setProblem] = useState<string | null>(null);

  const takesOptions = type === "select" || type === "status";

  function submit(e: FormEvent): void {
    e.preventDefault();
    const draft: ColumnDraft = {
      name: name.trim(),
      prompt: prompt.trim(),
      type,
      allowed_values: takesOptions ? splitList(options) : [],
      applicability: applicability.trim() === "" ? null : applicability.trim(),
      evidence_required: evidenceRequired,
    };
    const found = columnProblem(draft);
    setProblem(found);
    if (found === null) onSave(draft);
  }

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <form className="flex flex-col gap-3" onSubmit={submit}>
          <DialogTitle className="text-lg font-semibold">{column ? "Edit column" : "New column"}</DialogTitle>

          <Field label="Name">
            <Input value={name} onChange={(e) => setName(e.target.value)} autoFocus placeholder="Revenue" />
          </Field>

          <Field label="Question" hint="Asked of every row. The answer comes only from that row's documents.">
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              placeholder="What was total net revenue for the latest fiscal year?"
            />
          </Field>

          <Field label="Answer type">
            <Select value={type} onChange={(e) => setType(e.target.value as ColumnType)}>
              {COLUMN_TYPES.map((t) => (
                <option key={t} value={t}>
                  {TYPE_LABEL[t]}
                </option>
              ))}
            </Select>
          </Field>

          {takesOptions && (
            <Field label="Options" hint="Separate with commas.">
              <Input
                value={options}
                onChange={(e) => setOptions(e.target.value)}
                placeholder="Buy-side, Sell-side, Other"
              />
            </Field>
          )}

          <Field label="Only applies when (optional)" hint="Left empty, the question is asked of every row.">
            <Input
              value={applicability}
              onChange={(e) => setApplicability(e.target.value)}
              placeholder="The company has outstanding debt"
            />
          </Field>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={evidenceRequired}
              onChange={(e) => setEvidenceRequired(e.target.checked)}
            />
            Require a quote from the documents
          </label>

          {column && (
            <p className="text-xs text-muted-foreground">
              Changing the question or type marks the answers already in this column as out of date.
            </p>
          )}

          {(problem ?? error) && (
            <div className="rounded-md bg-[#ffe2dd] px-3 py-2 text-sm text-[#5d1715]">{problem ?? error}</div>
          )}

          <div className="flex gap-2 pt-1">
            <Button type="submit" variant="default" disabled={busy}>
              {busy ? "Saving…" : "Save"}
            </Button>
            <Button onClick={onClose}>Cancel</Button>
            {column && onDelete && (
              <Button variant="ghost" className="ml-auto text-destructive hover:text-destructive" onClick={onDelete}>
                Delete column
              </Button>
            )}
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
