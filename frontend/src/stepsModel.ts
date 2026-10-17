// What the agent did while answering, shaped for the screen. Pure functions, easy to test.

import type { ToolStatus } from "./types";

// One step the agent took: it starts "running" and is ticked off when it ends.
export interface UiTool {
  id: string;
  label: string;
  status: ToolStatus;
  summary: string;
}

export interface StepGroup {
  key: string;
  label: string;
  count: number; // how many times in a row the agent did this
  status: ToolStatus;
  summary: string; // only meaningful for a single step
}

// A group is as "bad" as its worst step: something still running, then a failure, then a wait.
const WORST: ToolStatus[] = ["running", "error", "needs_confirmation", "ok"];

/** Merge steps that repeat one after another ("Searching ×4") into one line. */
export function groupSteps(tools: UiTool[]): StepGroup[] {
  const groups: StepGroup[] = [];
  for (const t of tools) {
    const last = groups[groups.length - 1];
    if (last !== undefined && last.label === t.label) {
      last.count += 1;
      last.summary = "";
      if (WORST.indexOf(t.status) < WORST.indexOf(last.status)) last.status = t.status;
    } else {
      groups.push({ key: t.id, label: t.label, count: 1, status: t.status, summary: t.summary });
    }
  }
  return groups;
}

/** One line that says what happened, for the folded view. */
export function stepsHeadline(groups: StepGroup[], working: boolean): string {
  const total = groups.reduce((n, g) => n + g.count, 0);
  if (working) return groups[groups.length - 1]?.label ?? "Working";
  return total === 1 ? "Used 1 tool" : `Used ${total} tools`;
}
