// Pure helpers for the grid: no React, no network, so each one is easy to test.

import type {
  Cell,
  CellCheck,
  CellDerivation,
  CheckStatus,
  CellStatus,
  ColumnDraft,
  ColumnType,
  Estimate,
  Evidence,
  Run,
  RunStatus,
} from "./matrixTypes";

export function cellKey(entityId: string, columnId: string): string {
  return `${entityId}:${columnId}`;
}

/** Look a cell up by (row, column) in constant time instead of scanning the whole list. */
export function indexCells(cells: Cell[]): Map<string, Cell> {
  return new Map(cells.map((c) => [cellKey(c.entity_id, c.column_id), c]));
}

export type Tone = "gray" | "green" | "amber" | "red" | "blue" | "purple" | "orange";

export const STATUS_LABEL: Record<CellStatus, string> = {
  answered: "Answered",
  not_applicable: "Not applicable",
  not_found: "Not found",
  conflicting: "Conflicting",
  low_confidence: "Low confidence",
  pending_or_error: "Error",
};

// Not applicable (the question does not apply) and not found (it applies, the documents do not
// answer) stay visibly different: never collapse them.
const STATUS_TONE: Record<CellStatus, Tone> = {
  answered: "green",
  not_applicable: "gray",
  not_found: "orange",
  conflicting: "purple",
  low_confidence: "amber",
  pending_or_error: "red",
};

export function statusTone(status: CellStatus): Tone {
  return STATUS_TONE[status];
}

export interface CellView {
  kind: "empty" | "value" | "status";
  text: string;
  tone: Tone;
  stale: boolean;
  pinned: boolean;
  period: string | null; // the reporting period the figure covers
  derived: boolean; // computed in code from other cells, not read from a document
  check: CheckStatus | null; // the worst tie-out result on this cell, if it has any
}

/** What a grid cell shows: nothing yet, a value, or a status pill. */
export function cellView(cell: Cell | undefined): CellView {
  if (cell === undefined) {
    return { kind: "empty", text: "", tone: "gray", stale: false, pinned: false, period: null, derived: false, check: null };
  }
  const base = {
    stale: cell.stale,
    pinned: cell.pinned,
    period: cell.period ?? null,
    derived: cell.derivation != null,
    check: worstCheck(cell.checks ?? []),
  };
  if (cell.status === "answered" && cell.display_value !== null) {
    return { kind: "value", text: cell.display_value, tone: "green", ...base };
  }
  // A value we are not sure of is still shown, and marked, so the person can judge it.
  if (cell.status === "low_confidence" && cell.display_value !== null) {
    return { kind: "value", text: cell.display_value, tone: "amber", ...base };
  }
  return { kind: "status", text: STATUS_LABEL[cell.status], tone: statusTone(cell.status), ...base };
}

const CHECK_RANK: Record<CheckStatus, number> = { does_not_tie: 2, cannot_check: 1, ties: 0 };

/** The most worrying result among a cell's tie-outs: a failed check beats one that could not run,
 * which beats one that passed. Null when the cell has none. */
export function worstCheck(checks: CellCheck[]): CheckStatus | null {
  let worst: CheckStatus | null = null;
  for (const c of checks) {
    if (worst === null || CHECK_RANK[c.status] > CHECK_RANK[worst]) worst = c.status;
  }
  return worst;
}

export const CHECK_LABEL: Record<CheckStatus, string> = {
  ties: "Ties out",
  does_not_tie: "Does not tie",
  cannot_check: "Could not check",
};

export const CHECK_TONE: Record<CheckStatus, Tone> = {
  ties: "green",
  does_not_tie: "amber",
  cannot_check: "gray",
};

const OP_FORMULA: Record<string, (names: string[]) => string> = {
  ratio: ([a, b]) => `${a} ÷ ${b}`,
  difference: ([a, b]) => `${a} − ${b}`,
  sum: (names) => names.join(" + "),
  growth: ([a, b]) => `${a} against ${b}, as growth`,
  net_ratio: ([a, b, c]) => `(${a} − ${b}) ÷ ${c}`,
};

/** "Operating income ÷ Revenue": how a derived cell was computed, in words a person can check. */
export function derivationFormula(derivation: CellDerivation): string {
  const names = derivation.inputs.map((i) => i.column);
  return (OP_FORMULA[derivation.op] ?? ((n: string[]) => n.join(", ")))(names);
}

export interface ColumnImpact {
  cells: number; // how many cells the column has
  answered: number; // how many of them hold an answer
}

/** What removing a column would throw away. */
export function columnImpact(cells: Cell[], columnId: string): ColumnImpact {
  const own = cells.filter((c) => c.column_id === columnId);
  return { cells: own.length, answered: own.filter((c) => c.status === "answered").length };
}

/** The sentence that asks for the yes. It names the column and says what is lost. */
export function deleteWarning(name: string, impact: ColumnImpact): string {
  if (impact.cells === 0) return `Remove “${name}”? It has no cells yet. This cannot be undone.`;
  const cells = `${impact.cells} cell${impact.cells === 1 ? "" : "s"}`;
  return `Remove “${name}”? This deletes the column and its ${cells} (${impact.answered} answered). This cannot be undone.`;
}

export const TYPE_LABEL: Record<ColumnType, string> = {
  text: "Text",
  number: "Number",
  date: "Date",
  currency: "Currency",
  select: "Select",
  yes_no: "Yes / No",
  status: "Status",
};

export const TYPE_ICON: Record<ColumnType, string> = {
  text: "≡",
  number: "#",
  date: "▦",
  currency: "$",
  select: "◉",
  yes_no: "☑",
  status: "◐",
};

export function isNumeric(type: ColumnType): boolean {
  return type === "number" || type === "currency";
}

const TAG_TONES: readonly Tone[] = ["blue", "purple", "orange", "green", "amber", "red"];

/** A stable colour for a tag value, so the same word is always the same colour. */
export function tagTone(value: string): Tone {
  const lower = value.trim().toLowerCase();
  if (lower === "yes") return "green";
  if (lower === "no") return "gray";
  let hash = 0;
  for (const ch of lower) hash = (hash * 31 + ch.codePointAt(0)!) >>> 0;
  return TAG_TONES[hash % TAG_TONES.length]!;
}

export function hasTags(type: ColumnType): boolean {
  return type === "select" || type === "status" || type === "yes_no";
}

/** "a, b\nc" -> ["a", "b", "c"]: how the allowed-values box is read. */
export function splitList(text: string): string[] {
  return text
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter((s) => s !== "");
}

/** The same rules the backend enforces on a column, so mistakes show before a round trip.
 * The backend still checks: this is a convenience, not the guard. */
export function columnProblem(draft: ColumnDraft): string | null {
  if (draft.name.trim() === "") return "Give the column a name.";
  if (draft.prompt.trim() === "") return "Write the question this column answers.";
  const takesValues = draft.type === "select" || draft.type === "status";
  if (draft.type === "select" && draft.allowed_values.length < 2) {
    return "A select column needs at least two options.";
  }
  if (!takesValues && draft.allowed_values.length > 0) {
    return "Options only apply to select and status columns.";
  }
  const seen = new Set<string>();
  for (const value of draft.allowed_values) {
    const key = value.toLowerCase();
    if (seen.has(key)) return `The option "${value}" is repeated.`;
    seen.add(key);
  }
  return null;
}

// --- runs ---------------------------------------------------------------------------------------

export function isFinished(status: RunStatus): boolean {
  return status !== "running";
}

const RUN_ENDING: Record<Exclude<RunStatus, "running">, string> = {
  completed: "Finished",
  stopped: "Stopped",
  cost_cap: "Stopped at the spending cap",
  interrupted: "Interrupted by a server restart",
  failed: "Failed",
};

export function runLabel(run: Run): string {
  const counts = `${run.done} of ${run.total} cells`;
  if (run.status === "running") return `Filling ${counts}`;
  return `${RUN_ENDING[run.status]}: ${counts}`;
}

export function runPercent(run: Run): number {
  if (run.total === 0) return 100;
  return Math.min(100, Math.round(((run.done + run.skipped) / run.total) * 100));
}

/** Money for display. The value arrives as an exact decimal string; this is only for reading. */
export function formatUsd(value: string | null): string {
  if (value === null) return "unknown";
  const n = Number(value);
  if (!Number.isFinite(n)) return "unknown";
  if (n === 0) return "$0";
  if (n < 0.0001) return "<$0.0001";
  return n < 0.01 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`;
}

export function describeEstimate(estimate: Estimate): string {
  if (estimate.expected_usd === null) return "cost can't be estimated yet";
  const basis =
    estimate.basis === "measured"
      ? `from ${estimate.samples} earlier calls`
      : "from the model's published price";
  return `about ${formatUsd(estimate.expected_usd)} (${basis})`;
}

export function evidenceWhere(e: Evidence): string {
  if (e.page_start === null) return "";
  return e.page_end !== null && e.page_end !== e.page_start
    ? `pages ${e.page_start}–${e.page_end}`
    : `page ${e.page_start}`;
}
