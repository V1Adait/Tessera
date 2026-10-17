// These mirror the backend models (backend/app/matrix/models.py, matrix/runs.py, db/records.py).
// If you change one side, change the other. Decimals (money) arrive as strings: JSON has no
// exact decimal, so the backend sends "0.0004" and we never do arithmetic on it with floats.

export interface Workspace {
  id: string;
  name: string;
}

export interface Entity {
  id: string;
  workspace_id: string;
  name: string;
  position: number;
  ticker?: string | null; // how intake recognises the company next time
  domain?: string | null; // the company's website, for its logo
}

export type ColumnType = "text" | "number" | "date" | "currency" | "select" | "yes_no" | "status";

export const COLUMN_TYPES: readonly ColumnType[] = [
  "text",
  "number",
  "date",
  "currency",
  "select",
  "yes_no",
  "status",
];

export interface Column {
  id: string;
  workspace_id: string;
  position: number;
  version: number;
  name: string;
  prompt: string;
  type: ColumnType;
  allowed_values: string[];
  applicability: string | null;
  evidence_required: boolean;
  derivation?: { op: string; as?: string } | null; // set for a column computed from other columns
  depends_on?: string[]; // the columns a derived column reads, in order
}

// What a person gives to create a column (backend ColumnDraft).
export interface ColumnDraft {
  name: string;
  prompt: string;
  type: ColumnType;
  allowed_values: string[];
  applicability: string | null;
  evidence_required: boolean;
}

export type CellStatus =
  | "answered"
  | "not_applicable"
  | "not_found"
  | "conflicting"
  | "low_confidence"
  | "pending_or_error";

export const CELL_STATUSES: readonly CellStatus[] = [
  "answered",
  "not_applicable",
  "not_found",
  "conflicting",
  "low_confidence",
  "pending_or_error",
];

export interface Evidence {
  id: string;
  document_id: string;
  page_start: number | null;
  page_end: number | null;
  char_start: number;
  char_end: number;
  exact_range: boolean;
  quote: string;
}

export type CheckStatus = "ties" | "does_not_tie" | "cannot_check";

// The result of a tie-out on a cell (Revenue - Cost of revenue = Gross profit), computed in code.
export interface CellCheck {
  id: string;
  name: string;
  status: CheckStatus;
  detail: string;
}

// How a derived cell was computed. Which columns it read, and what they said.
export interface CellDerivation {
  op: string;
  inputs: { column: string; display: string | null; period?: string | null }[];
  notes: string[];
}

export interface Cell {
  id: string;
  entity_id: string;
  column_id: string;
  column_version: number;
  status: CellStatus;
  status_detail: string | null;
  display_value: string | null;
  stale: boolean;
  pinned: boolean;
  model: string | null;
  evidence: Evidence[];
  period?: string | null; // the reporting period the figure covers, e.g. "Q2 2026"
  checks?: CellCheck[];
  derivation?: CellDerivation | null; // set when the cell was computed from other cells
}

export type RunStatus = "running" | "completed" | "stopped" | "cost_cap" | "interrupted" | "failed";

export interface Run {
  id: string;
  status: RunStatus;
  total: number;
  done: number;
  failed: number;
  skipped: number;
  spent_usd: string;
  unknown_cost_calls: number;
  cap_usd: string;
  estimate_usd: string | null;
  detail: string | null;
}

export interface Estimate {
  cells: number;
  per_cell_usd: string | null;
  expected_usd: string | null;
  basis: "measured" | "price_list" | "unknown";
  samples: number;
}

export interface Plan {
  cells: number;
  by_reason: Record<string, number>;
  skipped_pinned: number;
  estimate: Estimate;
  cap_usd: string;
  needs_confirmation: boolean;
  confirmation_reason: string | null;
  warnings: string[];
}

// Which cells a run may touch. Leave a list out to mean "all".
export interface RunScope {
  entity_ids?: string[];
  column_ids?: string[];
}

// --- checks for data that arrives from the network ----------------------------------------------
// Types are erased at runtime, so anything from the network is checked by hand before it is trusted.

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

const isString = (v: unknown): v is string => typeof v === "string";
const isNullableString = (v: unknown): v is string | null => v === null || typeof v === "string";

export function isWorkspace(v: unknown): v is Workspace {
  return isRecord(v) && isString(v.id) && isString(v.name);
}

export function isEntity(v: unknown): v is Entity {
  return isRecord(v) && isString(v.id) && isString(v.name) && typeof v.position === "number";
}

export function isColumn(v: unknown): v is Column {
  return (
    isRecord(v) &&
    isString(v.id) &&
    isString(v.name) &&
    isString(v.prompt) &&
    isString(v.type) &&
    (COLUMN_TYPES as readonly string[]).includes(v.type) &&
    Array.isArray(v.allowed_values) &&
    typeof v.version === "number"
  );
}

function isEvidence(v: unknown): v is Evidence {
  return (
    isRecord(v) &&
    isString(v.id) &&
    isString(v.document_id) &&
    isString(v.quote) &&
    typeof v.char_start === "number" &&
    typeof v.char_end === "number"
  );
}

export function isCell(v: unknown): v is Cell {
  return (
    isRecord(v) &&
    isString(v.id) &&
    isString(v.entity_id) &&
    isString(v.column_id) &&
    isString(v.status) &&
    (CELL_STATUSES as readonly string[]).includes(v.status) &&
    isNullableString(v.display_value) &&
    isNullableString(v.status_detail) &&
    typeof v.stale === "boolean" &&
    typeof v.pinned === "boolean" &&
    Array.isArray(v.evidence) &&
    v.evidence.every(isEvidence) &&
    (v.period === undefined || isNullableString(v.period)) &&
    (v.checks === undefined || (Array.isArray(v.checks) && v.checks.every(isCheck))) &&
    (v.derivation === undefined || v.derivation === null || isDerivation(v.derivation))
  );
}

const CHECK_STATUSES: readonly string[] = ["ties", "does_not_tie", "cannot_check"];

function isCheck(v: unknown): v is CellCheck {
  return (
    isRecord(v) &&
    isString(v.id) &&
    isString(v.name) &&
    isString(v.status) &&
    CHECK_STATUSES.includes(v.status) &&
    isString(v.detail)
  );
}

function isDerivation(v: unknown): v is CellDerivation {
  return (
    isRecord(v) &&
    isString(v.op) &&
    Array.isArray(v.inputs) &&
    v.inputs.every((i) => isRecord(i) && isString(i.column) && isNullableString(i.display)) &&
    Array.isArray(v.notes) &&
    v.notes.every(isString)
  );
}

export function isRun(v: unknown): v is Run {
  return (
    isRecord(v) &&
    isString(v.id) &&
    isString(v.status) &&
    typeof v.total === "number" &&
    typeof v.done === "number" &&
    typeof v.failed === "number" &&
    typeof v.skipped === "number" &&
    isString(v.spent_usd) &&
    isString(v.cap_usd)
  );
}

export function isPlan(v: unknown): v is Plan {
  return (
    isRecord(v) &&
    typeof v.cells === "number" &&
    isRecord(v.by_reason) &&
    typeof v.skipped_pinned === "number" &&
    isRecord(v.estimate) &&
    isString(v.cap_usd) &&
    typeof v.needs_confirmation === "boolean" &&
    Array.isArray(v.warnings)
  );
}

export function isListOf<T>(guard: (v: unknown) => v is T): (v: unknown) => v is T[] {
  return (v: unknown): v is T[] => Array.isArray(v) && v.every(guard);
}
