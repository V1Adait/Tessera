// Pure helpers for showing documents and intake decisions. No React, no network: easy to test.

import type { IntakeView, WorkspaceDocument } from "./documentsApi";
import type { Tone } from "./matrixModel";

/** The files of each row, in upload order, keyed by row id. Files with no row are not in the map. */
export function documentsByRow(documents: WorkspaceDocument[]): Map<string, WorkspaceDocument[]> {
  const byRow = new Map<string, WorkspaceDocument[]>();
  for (const doc of documents) {
    if (doc.entity_id === null) continue;
    byRow.set(doc.entity_id, [...(byRow.get(doc.entity_id) ?? []), doc]);
  }
  return byRow;
}

export function unplacedDocuments(documents: WorkspaceDocument[]): WorkspaceDocument[] {
  return documents.filter((d) => d.entity_id === null);
}

export interface IntakeLine {
  tone: Tone;
  text: string;
  needsChoice: boolean; // a person has to say which row this file belongs to
}

/** One line telling the person what intake did. The wording says WHAT happened, and why when unsure. */
export function describeIntake(view: IntakeView): IntakeLine {
  // A person may have attached the file since intake ran: where it is NOW wins over what it decided.
  if (view.outcome === "needs_review" && view.entity_name !== null) {
    return { tone: "green", text: `Attached to ${view.entity_name}`, needsChoice: false };
  }
  switch (view.outcome) {
    case "created":
      return { tone: "green", text: `New row: ${view.entity_name ?? "created"}`, needsChoice: false };
    case "assigned":
      return { tone: "green", text: `Added to ${view.entity_name ?? "its row"}`, needsChoice: false };
    case "skipped":
      return { tone: "gray", text: `In ${view.entity_name ?? "its row"}`, needsChoice: false };
    case "needs_review":
      return { tone: "amber", text: view.detail ?? "Which row is this file for?", needsChoice: true };
  }
}

/** A read file whose placement has not been fetched yet needs a look at intake. */
export function awaitsIntake(status: string, view: IntakeView | null | undefined): boolean {
  return (status === "parsed" || status === "needs_review") && !view;
}
