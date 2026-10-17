import { describe, expect, it } from "vitest";
import { isCell, isColumn, isListOf, isPlan, isRun } from "./matrixTypes";

const column = {
  id: "k",
  name: "Revenue",
  prompt: "What was revenue?",
  type: "currency",
  allowed_values: [],
  version: 1,
};

const cell = {
  id: "c",
  entity_id: "e",
  column_id: "k",
  status: "answered",
  display_value: "$1",
  status_detail: null,
  stale: false,
  pinned: false,
  evidence: [{ id: "x", document_id: "d", quote: "q", char_start: 0, char_end: 1 }],
};

describe("response checks", () => {
  it("accept the real shapes", () => {
    expect(isColumn(column)).toBe(true);
    expect(isCell(cell)).toBe(true);
    expect(isListOf(isColumn)([column, column])).toBe(true);
  });

  it("refuse an unknown column type or cell status", () => {
    expect(isColumn({ ...column, type: "matrix" })).toBe(false);
    expect(isCell({ ...cell, status: "great" })).toBe(false);
  });

  it("refuse a list with one bad item, and non-lists", () => {
    expect(isListOf(isColumn)([column, { id: "k" }])).toBe(false);
    expect(isListOf(isColumn)(null)).toBe(false);
  });

  it("refuse a cell with a malformed quote", () => {
    expect(isCell({ ...cell, evidence: [{ id: "x" }] })).toBe(false);
  });

  it("check runs and plans", () => {
    const run = { id: "r", status: "running", total: 1, done: 0, failed: 0, skipped: 0, spent_usd: "0", cap_usd: "1" };
    expect(isRun(run)).toBe(true);
    expect(isRun({ ...run, spent_usd: 0 })).toBe(false); // money must be an exact string
    const plan = {
      cells: 1,
      by_reason: {},
      skipped_pinned: 0,
      estimate: {},
      cap_usd: "1",
      needs_confirmation: false,
      warnings: [],
    };
    expect(isPlan(plan)).toBe(true);
    expect(isPlan({ ...plan, cells: "1" })).toBe(false);
  });
});

describe("the finance fields of a cell", () => {
  const checks = [{ id: "g", name: "Gross profit ties", status: "ties", detail: "ok" }];
  const derivation = { op: "ratio", inputs: [{ column: "Revenue", display: "$1" }], notes: [] };

  it("are optional, so an older server still works", () => {
    expect(isCell(cell)).toBe(true);
  });

  it("are accepted when well formed", () => {
    expect(isCell({ ...cell, period: "Q2 2026", checks, derivation })).toBe(true);
    expect(isCell({ ...cell, period: null, derivation: null })).toBe(true);
  });

  it("are refused when damaged", () => {
    expect(isCell({ ...cell, period: 5 })).toBe(false);
    expect(isCell({ ...cell, checks: [{ ...checks[0], status: "great" }] })).toBe(false);
    expect(isCell({ ...cell, checks: "ties" })).toBe(false);
    expect(isCell({ ...cell, derivation: { op: "ratio" } })).toBe(false);
  });
});
