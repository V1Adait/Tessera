import { describe, expect, it } from "vitest";
import { groupSteps, stepsHeadline, type UiTool } from "./stepsModel";

const step = (id: string, label: string, status: UiTool["status"], summary = ""): UiTool => ({
  id,
  label,
  status,
  summary,
});

describe("groupSteps", () => {
  it("merges the same step repeated in a row, and counts it", () => {
    const groups = groupSteps([
      step("1", "Reading the matrix", "ok", "Read the matrix"),
      step("2", "Searching the documents", "ok", "Found 6 passages"),
      step("3", "Searching the documents", "ok", "Found 6 passages"),
      step("4", "Searching the documents", "ok", "Found 6 passages"),
    ]);

    expect(groups.map((g) => [g.label, g.count])).toEqual([
      ["Reading the matrix", 1],
      ["Searching the documents", 3],
    ]);
    expect(groups[0]?.summary).toBe("Read the matrix"); // a single step keeps its summary
    expect(groups[1]?.summary).toBe(""); // a repeated one does not
  });

  it("does not merge steps that are not neighbours", () => {
    const groups = groupSteps([step("1", "A", "ok"), step("2", "B", "ok"), step("3", "A", "ok")]);

    expect(groups).toHaveLength(3);
  });

  it("a group shows its worst step: running beats a failure beats a wait beats ok", () => {
    expect(groupSteps([step("1", "A", "ok"), step("2", "A", "running")])[0]?.status).toBe("running");
    expect(groupSteps([step("1", "A", "ok"), step("2", "A", "error")])[0]?.status).toBe("error");
    expect(groupSteps([step("1", "A", "ok"), step("2", "A", "needs_confirmation")])[0]?.status).toBe(
      "needs_confirmation",
    );
    expect(groupSteps([step("1", "A", "ok"), step("2", "A", "ok")])[0]?.status).toBe("ok");
  });

  it("returns nothing for no steps", () => {
    expect(groupSteps([])).toEqual([]);
  });
});

describe("stepsHeadline", () => {
  const groups = groupSteps([step("1", "Reading the matrix", "ok"), step("2", "Searching the documents", "running")]);

  it("names what it is doing right now while it works", () => {
    expect(stepsHeadline(groups, true)).toBe("Searching the documents");
  });

  it("counts the tools used once it is done, counting repeats", () => {
    const repeated = groupSteps([step("1", "A", "ok"), step("2", "A", "ok"), step("3", "B", "ok")]);

    expect(stepsHeadline(repeated, false)).toBe("Used 3 tools");
    expect(stepsHeadline(groupSteps([step("1", "A", "ok")]), false)).toBe("Used 1 tool");
  });
});
