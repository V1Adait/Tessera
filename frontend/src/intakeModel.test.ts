import { describe, expect, it } from "vitest";
import type { IntakeView, WorkspaceDocument } from "./documentsApi";
import { awaitsIntake, describeIntake, documentsByRow, unplacedDocuments } from "./intakeModel";

const doc = (id: string, entity: string | null): WorkspaceDocument => ({
  id,
  filename: `${id}.pdf`,
  status: "parsed",
  entity_id: entity,
});

const view = (over: Partial<IntakeView>): IntakeView => ({
  outcome: "created",
  entity_id: "r1",
  entity_name: "Palantir",
  detected: {},
  detail: null,
  candidates: [],
  ...over,
});

describe("documentsByRow", () => {
  it("groups files by row in upload order, and leaves out files with no row", () => {
    const grouped = documentsByRow([doc("a", "r1"), doc("b", null), doc("c", "r1"), doc("d", "r2")]);

    expect(grouped.get("r1")?.map((d) => d.id)).toEqual(["a", "c"]);
    expect(grouped.get("r2")?.map((d) => d.id)).toEqual(["d"]);
    expect(grouped.size).toBe(2);
  });

  it("is empty with no files", () => {
    expect(documentsByRow([]).size).toBe(0);
  });
});

describe("unplacedDocuments", () => {
  it("returns only the files that belong to no row", () => {
    expect(unplacedDocuments([doc("a", "r1"), doc("b", null)]).map((d) => d.id)).toEqual(["b"]);
  });
});

describe("describeIntake", () => {
  it("says a row was created, or a file was added to one", () => {
    expect(describeIntake(view({})).text).toBe("New row: Palantir");
    expect(describeIntake(view({ outcome: "assigned" })).text).toBe("Added to Palantir");
  });

  it("asks for a choice when intake was unsure, and says why", () => {
    const line = describeIntake(
      view({
        outcome: "needs_review",
        entity_id: null,
        entity_name: null,
        detail: "not sure which company this is",
      }),
    );

    expect(line).toMatchObject({ tone: "amber", needsChoice: true, text: "not sure which company this is" });
  });

  it("stops asking once a person has attached the file", () => {
    const line = describeIntake(view({ outcome: "needs_review", entity_name: "Pick Me" }));

    expect(line).toEqual({ tone: "green", text: "Attached to Pick Me", needsChoice: false });
  });

  it("falls back to a plain question when no reason was given", () => {
    const line = describeIntake(view({ outcome: "needs_review", entity_id: null, entity_name: null }));

    expect(line.text).toBe("Which row is this file for?");
  });
});

describe("awaitsIntake", () => {
  it("is true only for a read file whose placement has not arrived", () => {
    expect(awaitsIntake("parsed", undefined)).toBe(true);
    expect(awaitsIntake("parsed", null)).toBe(true);
    expect(awaitsIntake("parsing", undefined)).toBe(false);
    expect(awaitsIntake("parsed", view({}))).toBe(false);
  });
});
