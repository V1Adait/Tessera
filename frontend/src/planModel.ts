// The plan card's data and wording. Pure: no React, no network, so each rule is easy to test.

import { formatUsd } from "./matrixModel";

export type PlanSource = "pack" | "new" | "existing";

export interface PlanColumn {
  name: string;
  type: string;
  prompt: string;
  allowed_values: string[];
  source: PlanSource;
  derived?: unknown; // set for a column computed in code from other columns
  inputs?: string[]; // the columns a computed column reads, by name
}

// What the confirmation card shows for a question the agent turned into columns. Mirrors the
// payload built in backend/app/agent/tools_plan.py.
export interface PlanPayload {
  question: string;
  columns: PlanColumn[];
  rows: number;
  cells: number;
  estimate_usd: string | null;
  basis: string;
  cap_usd: string;
  notes: string[];
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

function isPlanColumn(v: unknown): v is PlanColumn {
  return (
    isRecord(v) &&
    typeof v.name === "string" &&
    typeof v.type === "string" &&
    typeof v.prompt === "string" &&
    Array.isArray(v.allowed_values) &&
    (v.source === "pack" || v.source === "new" || v.source === "existing")
  );
}

export function isPlanPayload(v: unknown): v is PlanPayload {
  return (
    isRecord(v) &&
    typeof v.question === "string" &&
    Array.isArray(v.columns) &&
    v.columns.every(isPlanColumn) &&
    typeof v.rows === "number" &&
    typeof v.cells === "number" &&
    (v.estimate_usd === null || typeof v.estimate_usd === "string") &&
    typeof v.cap_usd === "string" &&
    Array.isArray(v.notes) &&
    v.notes.every((n) => typeof n === "string")
  );
}

/** The plan stored with a pending action (so the card comes back after a reload), or null. */
export function planOfArgs(args: unknown): PlanPayload | null {
  if (!isRecord(args)) return null;
  return isPlanPayload(args.plan) ? args.plan : null;
}

export const SOURCE_LABEL: Record<PlanSource, string> = {
  pack: "From the pack",
  new: "New",
  existing: "Already in the table",
};

export interface ColumnBadge {
  label: string;
  tone: "blue" | "green" | "gray" | "purple";
}

/** What a planned column is tagged with: a computed column says so, others say where they came from. */
export function columnBadge(column: PlanColumn): ColumnBadge {
  if (column.derived !== undefined && column.derived !== null && column.source !== "existing") {
    return { label: "Computed in code", tone: "purple" };
  }
  return { label: SOURCE_LABEL[column.source], tone: { pack: "blue", new: "green", existing: "gray" }[column.source] as ColumnBadge["tone"] };
}

export function planHeadline(plan: PlanPayload): string {
  const fresh = plan.columns.filter((c) => c.source !== "existing").length;
  const reused = plan.columns.length - fresh;
  const columns = `${fresh} new column${fresh === 1 ? "" : "s"}`;
  const kept = reused > 0 ? ` and ${reused} already in the table` : "";
  if (plan.rows === 0) return `${columns}${kept}. No companies yet, so nothing to fill.`;
  return `${columns}${kept}, ${plan.cells} cell${plan.cells === 1 ? "" : "s"} to fill across ${plan.rows} ${plan.rows === 1 ? "company" : "companies"}.`;
}

/** The cost line. An unknown cost is said to be unknown: it is never shown as free. */
export function planCost(plan: PlanPayload): string {
  if (plan.cells === 0) return "Nothing to fill, so nothing to pay.";
  if (plan.estimate_usd === null) {
    return `The cost cannot be estimated yet. It stops by itself at ${formatUsd(plan.cap_usd)}.`;
  }
  const basis = plan.basis === "measured" ? "from past runs" : "from the model's price";
  return `About ${formatUsd(plan.estimate_usd)} (${basis}). It stops by itself at ${formatUsd(plan.cap_usd)}.`;
}
