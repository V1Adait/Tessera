// Confirming or cancelling what the agent was not allowed to do on its own.
// Mirrors backend/app/api_agent.py. Every response is checked before it is trusted.

import { API_URL, ApiError } from "./api";

export interface PendingAction {
  id: string;
  tool: string;
  risk: "costly" | "destructive";
  summary: string;
  args?: unknown; // what the action will run with; for a column plan it holds the plan card's data
}

export interface ActionResult {
  ok: boolean;
  message: string;
  changed_matrix: boolean;
  runId: string | null; // set when confirming started a fill: the answer is written when it ends
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

export function isPendingAction(v: unknown): v is PendingAction {
  return (
    isRecord(v) &&
    typeof v.id === "string" &&
    typeof v.tool === "string" &&
    (v.risk === "costly" || v.risk === "destructive") &&
    typeof v.summary === "string"
  );
}

export function isActionResult(v: unknown): v is Omit<ActionResult, "runId"> & { data?: unknown } {
  return (
    isRecord(v) &&
    typeof v.ok === "boolean" &&
    typeof v.message === "string" &&
    (v.changed_matrix === undefined || typeof v.changed_matrix === "boolean")
  );
}

/** The id of the run that confirming started, if any (the server puts it in `data`). */
export function runIdOf(body: { data?: unknown }): string | null {
  const data = body.data;
  if (!isRecord(data)) return null;
  return typeof data.run_id === "string" ? data.run_id : null;
}

async function failure(response: Response): Promise<ApiError> {
  let text = `The server answered ${response.status}.`;
  try {
    const body: unknown = await response.json();
    if (isRecord(body) && typeof body.detail === "string") text = body.detail;
  } catch {
    // keep the generic text
  }
  return new ApiError(response.status, text);
}

async function resolve(actionId: string, verb: "confirm" | "cancel"): Promise<ActionResult> {
  const response = await fetch(`${API_URL}/actions/${actionId}/${verb}`, { method: "POST" });
  if (!response.ok) throw await failure(response);
  const body: unknown = await response.json();
  if (!isActionResult(body)) throw new Error("The server sent data in an unexpected shape.");
  return {
    ok: body.ok,
    message: body.message,
    changed_matrix: body.changed_matrix === true,
    runId: runIdOf(body),
  };
}

export const confirmAction = (id: string): Promise<ActionResult> => resolve(id, "confirm");
export const cancelAction = (id: string): Promise<ActionResult> => resolve(id, "cancel");

/** Actions of a conversation that are still waiting for a yes (to show them again after a refresh). */
export async function fetchPendingActions(conversationId: string): Promise<PendingAction[]> {
  const response = await fetch(`${API_URL}/conversations/${conversationId}/actions`);
  if (!response.ok) throw await failure(response);
  const body: unknown = await response.json();
  if (!Array.isArray(body) || !body.every(isPendingAction)) {
    throw new Error("The server sent data in an unexpected shape.");
  }
  return body;
}
