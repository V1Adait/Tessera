import { describe, expect, it } from "vitest";
import {
  cellKey,
  cellView,
  columnImpact,
  columnProblem,
  deleteWarning,
  derivationFormula,
  describeEstimate,
  evidenceWhere,
  formatUsd,
  indexCells,
  isFinished,
  runLabel,
  runPercent,
  splitList,
  statusTone,
  tagTone,
  worstCheck,
} from "./matrixModel";
import type { Cell, CellCheck, ColumnDraft, Run } from "./matrixTypes";

const cell = (over: Partial<Cell> = {}): Cell => ({
  id: "c1",
  entity_id: "e1",
  column_id: "k1",
  column_version: 1,
  status: "answered",
  status_detail: null,
  display_value: "166,000",
  stale: false,
  pinned: false,
  model: null,
  evidence: [],
  ...over,
});

const run = (over: Partial<Run> = {}): Run => ({
  id: "r",
  status: "running",
  total: 10,
  done: 4,
  failed: 0,
  skipped: 0,
  spent_usd: "0.0004",
  unknown_cost_calls: 0,
  cap_usd: "0.5",
  estimate_usd: null,
  detail: null,
  ...over,
});

describe("indexCells", () => {
  it("finds a cell by row and column", () => {
    const index = indexCells([cell(), cell({ id: "c2", column_id: "k2" })]);
    expect(index.get(cellKey("e1", "k2"))?.id).toBe("c2");
    expect(index.get(cellKey("e9", "k1"))).toBeUndefined();
  });
});

describe("cellView", () => {
  it("shows nothing for a cell that does not exist yet", () => {
    expect(cellView(undefined).kind).toBe("empty");
  });

  it("shows an answered value", () => {
    expect(cellView(cell())).toMatchObject({ kind: "value", text: "166,000", tone: "green" });
  });

  it("keeps not applicable and not found visibly different", () => {
    const na = cellView(cell({ status: "not_applicable", display_value: null }));
    const nf = cellView(cell({ status: "not_found", display_value: null }));
    expect(na.text).toBe("Not applicable");
    expect(nf.text).toBe("Not found");
    expect(na.tone).not.toBe(nf.tone);
  });

  it("shows a low-confidence value, marked as uncertain, instead of hiding it", () => {
    expect(cellView(cell({ status: "low_confidence" }))).toMatchObject({ kind: "value", tone: "amber" });
  });

  it("shows a low-confidence cell with no value as a status", () => {
    const view = cellView(cell({ status: "low_confidence", display_value: null }));
    expect(view).toMatchObject({ kind: "status", text: "Low confidence" });
  });

  it("passes stale and pinned flags through", () => {
    expect(cellView(cell({ stale: true, pinned: true }))).toMatchObject({ stale: true, pinned: true });
  });

  it("gives an error its own tone", () => {
    expect(statusTone("pending_or_error")).toBe("red");
  });
});

describe("tagTone", () => {
  it("is stable: the same word is always the same colour", () => {
    expect(tagTone("Buy-Side")).toBe(tagTone("  buy-side "));
  });

  it("colours yes and no the way people expect", () => {
    expect(tagTone("Yes")).toBe("green");
    expect(tagTone("no")).toBe("gray");
  });
});

describe("splitList", () => {
  it("splits on commas and new lines and drops blanks", () => {
    expect(splitList("a, b\n c ,, ")).toEqual(["a", "b", "c"]);
    expect(splitList("")).toEqual([]);
  });
});

describe("columnProblem", () => {
  const ok: ColumnDraft = {
    name: "Revenue",
    prompt: "What was revenue?",
    type: "currency",
    allowed_values: [],
    applicability: null,
    evidence_required: true,
  };

  it("accepts a good column", () => {
    expect(columnProblem(ok)).toBeNull();
  });

  it("asks for a name and a question", () => {
    expect(columnProblem({ ...ok, name: "  " })).toMatch(/name/i);
    expect(columnProblem({ ...ok, prompt: "" })).toMatch(/question/i);
  });

  it("needs two options for a select, and refuses options on other types", () => {
    expect(columnProblem({ ...ok, type: "select", allowed_values: ["a"] })).toMatch(/two options/);
    expect(columnProblem({ ...ok, type: "select", allowed_values: ["a", "b"] })).toBeNull();
    expect(columnProblem({ ...ok, type: "number", allowed_values: ["a"] })).toMatch(/only apply/);
  });

  it("refuses a repeated option, ignoring case", () => {
    expect(columnProblem({ ...ok, type: "status", allowed_values: ["Open", "open"] })).toMatch(/repeated/);
  });
});

describe("runs", () => {
  it("knows when a run is over", () => {
    expect(isFinished("running")).toBe(false);
    expect(isFinished("cost_cap")).toBe(true);
  });

  it("describes progress and how a run ended", () => {
    expect(runLabel(run())).toBe("Filling 4 of 10 cells");
    expect(runLabel(run({ status: "cost_cap" }))).toMatch(/spending cap/);
  });

  it("computes a percentage that never passes 100 and handles an empty run", () => {
    expect(runPercent(run())).toBe(40);
    expect(runPercent(run({ done: 9, skipped: 9 }))).toBe(100);
    expect(runPercent(run({ total: 0, done: 0 }))).toBe(100);
  });
});

describe("formatUsd", () => {
  it("reads exact decimal strings", () => {
    expect(formatUsd("0")).toBe("$0");
    expect(formatUsd("0.0004000000")).toBe("$0.0004");
    expect(formatUsd("1.5")).toBe("$1.50");
    expect(formatUsd("0.00000001")).toBe("<$0.0001");
  });

  it("says unknown rather than showing a wrong number", () => {
    expect(formatUsd(null)).toBe("unknown");
    expect(formatUsd("abc")).toBe("unknown");
  });

  it("explains where an estimate comes from, or that there is none", () => {
    const base = { cells: 5, per_cell_usd: "0.001", expected_usd: "0.005", samples: 7 };
    expect(describeEstimate({ ...base, basis: "measured" })).toContain("7 earlier calls");
    expect(describeEstimate({ ...base, basis: "price_list" })).toContain("published price");
    expect(describeEstimate({ ...base, expected_usd: null, basis: "unknown" })).toMatch(/can't be estimated/);
  });
});

describe("evidenceWhere", () => {
  const e = { id: "1", document_id: "d", quote: "q", char_start: 0, char_end: 1, exact_range: true };
  it("names a page or a page range, or nothing", () => {
    expect(evidenceWhere({ ...e, page_start: 4, page_end: 4 })).toBe("page 4");
    expect(evidenceWhere({ ...e, page_start: 4, page_end: 6 })).toBe("pages 4–6");
    expect(evidenceWhere({ ...e, page_start: null, page_end: null })).toBe("");
  });
});

const check = (status: CellCheck["status"]): CellCheck => ({ id: status, name: "n", status, detail: "d" });

describe("worstCheck", () => {
  it("is null when a cell has no checks", () => {
    expect(worstCheck([])).toBeNull();
  });

  it("ranks a failed check above one that could not run, above one that passed", () => {
    expect(worstCheck([check("ties"), check("does_not_tie"), check("cannot_check")])).toBe("does_not_tie");
    expect(worstCheck([check("ties"), check("cannot_check")])).toBe("cannot_check");
    expect(worstCheck([check("ties"), check("ties")])).toBe("ties");
  });
});

describe("derivationFormula", () => {
  const inputs = (...names: string[]) => names.map((column) => ({ column, display: null }));

  it("writes each operation as a formula a person can check", () => {
    expect(derivationFormula({ op: "ratio", inputs: inputs("Operating income", "Revenue"), notes: [] })).toBe(
      "Operating income ÷ Revenue",
    );
    expect(derivationFormula({ op: "difference", inputs: inputs("Total debt", "Cash"), notes: [] })).toBe(
      "Total debt − Cash",
    );
    expect(derivationFormula({ op: "net_ratio", inputs: inputs("Debt", "Cash", "EBITDA"), notes: [] })).toBe(
      "(Debt − Cash) ÷ EBITDA",
    );
    expect(derivationFormula({ op: "sum", inputs: inputs("A", "B", "C"), notes: [] })).toBe("A + B + C");
  });

  it("falls back to listing the inputs for an operation it does not know", () => {
    expect(derivationFormula({ op: "mystery", inputs: inputs("A", "B"), notes: [] })).toBe("A, B");
  });
});

describe("cellView for the finance layer", () => {
  it("carries the period, whether it was computed, and the worst check", () => {
    const view = cellView(
      cell({
        period: "Q2 2026",
        derivation: { op: "ratio", inputs: [], notes: [] },
        checks: [check("ties"), check("does_not_tie")],
      }),
    );

    expect(view).toMatchObject({ period: "Q2 2026", derived: true, check: "does_not_tie" });
  });

  it("is plain for an ordinary cell", () => {
    expect(cellView(cell())).toMatchObject({ period: null, derived: false, check: null });
    expect(cellView(undefined)).toMatchObject({ period: null, derived: false, check: null });
  });
});

describe("columnImpact and deleteWarning", () => {
  it("counts the cells a column has and how many are answered, for that column only", () => {
    const cells = [
      cell({ id: "1", column_id: "k1", status: "answered" }),
      cell({ id: "2", column_id: "k1", status: "not_found" }),
      cell({ id: "3", column_id: "k1", status: "answered" }),
      cell({ id: "4", column_id: "k2", status: "answered" }),
    ];

    expect(columnImpact(cells, "k1")).toEqual({ cells: 3, answered: 2 });
    expect(columnImpact(cells, "k2")).toEqual({ cells: 1, answered: 1 });
    expect(columnImpact(cells, "nope")).toEqual({ cells: 0, answered: 0 });
  });

  it("asks for the yes by naming the column and what is lost", () => {
    expect(deleteWarning("Revenue", { cells: 3, answered: 2 })).toBe(
      "Remove “Revenue”? This deletes the column and its 3 cells (2 answered). This cannot be undone.",
    );
  });

  it("uses the singular for one cell and says plainly when there are none", () => {
    expect(deleteWarning("Revenue", { cells: 1, answered: 0 })).toContain("its 1 cell (0 answered)");
    expect(deleteWarning("Revenue", { cells: 0, answered: 0 })).toBe(
      "Remove “Revenue”? It has no cells yet. This cannot be undone.",
    );
  });
});
