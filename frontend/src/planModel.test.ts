import { describe, expect, it } from "vitest";
import { columnBadge, isPlanPayload, planCost, planHeadline, planOfArgs, type PlanPayload } from "./planModel";

const column = (name: string, source: "pack" | "new" | "existing" = "pack") => ({
  name,
  type: "currency",
  prompt: `What was ${name}?`,
  allowed_values: [],
  source,
});

const plan = (over: Partial<PlanPayload> = {}): PlanPayload => ({
  question: "Is it a good investment?",
  columns: [column("Revenue growth"), column("Free cash flow"), column("Total debt")],
  rows: 2,
  cells: 6,
  estimate_usd: "0.0042",
  basis: "price_list",
  cap_usd: "0.5",
  notes: [],
  ...over,
});

describe("isPlanPayload", () => {
  it("accepts what the server sends", () => {
    expect(isPlanPayload(plan())).toBe(true);
    expect(isPlanPayload(plan({ estimate_usd: null }))).toBe(true);
  });

  it("rejects a wrong shape, an unknown source and non-text notes", () => {
    expect(isPlanPayload(null)).toBe(false);
    expect(isPlanPayload({ ...plan(), columns: "nope" })).toBe(false);
    expect(isPlanPayload({ ...plan(), columns: [{ ...column("A"), source: "magic" }] })).toBe(false);
    expect(isPlanPayload({ ...plan(), notes: [1] })).toBe(false);
    expect(isPlanPayload({ ...plan(), cells: "6" })).toBe(false);
  });
});

describe("planOfArgs", () => {
  it("finds the plan stored with a pending action, so the card survives a reload", () => {
    expect(planOfArgs({ question: "q", plan: plan() })?.cells).toBe(6);
  });

  it("is null for an action that has no plan, or a damaged one", () => {
    expect(planOfArgs({ row: "Acme" })).toBeNull();
    expect(planOfArgs({ plan: { cells: 1 } })).toBeNull();
    expect(planOfArgs(undefined)).toBeNull();
    expect(planOfArgs("text")).toBeNull();
  });
});

describe("planHeadline", () => {
  it("counts new columns, cells and companies", () => {
    expect(planHeadline(plan())).toBe("3 new columns, 6 cells to fill across 2 companies.");
  });

  it("says how many columns were already there", () => {
    const p = plan({ columns: [column("A"), column("B", "existing")], cells: 3, rows: 3 });

    expect(planHeadline(p)).toBe("1 new column and 1 already in the table, 3 cells to fill across 3 companies.");
  });

  it("uses singular nouns for one", () => {
    expect(planHeadline(plan({ columns: [column("A")], cells: 1, rows: 1 }))).toBe(
      "1 new column, 1 cell to fill across 1 company.",
    );
  });

  it("says plainly that there is nothing to fill when there are no rows", () => {
    expect(planHeadline(plan({ rows: 0, cells: 0 }))).toContain("No companies yet");
  });
});

describe("planCost", () => {
  it("shows the estimate, where it came from, and the cap", () => {
    const text = planCost(plan());

    expect(text).toContain("About $0.0042");
    expect(text).toContain("from the model's price");
    expect(text).toContain("stops by itself at $0.50");
  });

  it("says measured when it comes from past runs", () => {
    expect(planCost(plan({ basis: "measured" }))).toContain("from past runs");
  });

  it("never shows an unknown cost as free", () => {
    const text = planCost(plan({ estimate_usd: null }));

    expect(text).toContain("cannot be estimated");
    expect(text).not.toContain("$0.00");
  });

  it("has nothing to pay when there is nothing to fill", () => {
    expect(planCost(plan({ cells: 0 }))).toBe("Nothing to fill, so nothing to pay.");
  });
});

describe("columnBadge", () => {
  it("says a computed column is computed in code", () => {
    expect(columnBadge({ ...column("Operating margin"), derived: { op: "ratio" } })).toEqual({
      label: "Computed in code",
      tone: "purple",
    });
  });

  it("says where an ordinary column came from", () => {
    expect(columnBadge(column("Revenue", "pack")).label).toBe("From the pack");
    expect(columnBadge(column("Backlog", "new")).label).toBe("New");
  });

  it("calls a computed column that already exists what it is: already in the table", () => {
    expect(columnBadge({ ...column("Operating margin", "existing"), derived: { op: "ratio" } }).label).toBe(
      "Already in the table",
    );
  });
});
