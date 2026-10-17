// Uploading documents and watching them get parsed. Mirrors backend/app/api_documents.py.

import { API_URL, ApiError } from "./api";

export type DocumentStatus = "uploaded" | "parsing" | "parsed" | "needs_review" | "needs_ocr" | "failed";

export interface DocumentInfo {
  id: string;
  filename: string;
  status: DocumentStatus;
  status_detail: string | null;
  entity_id: string | null;
  duplicate: boolean;
}

const STATUSES: readonly string[] = ["uploaded", "parsing", "parsed", "needs_review", "needs_ocr", "failed"];

export function isDocumentInfo(v: unknown): v is DocumentInfo {
  if (typeof v !== "object" || v === null) return false;
  const d = v as Record<string, unknown>;
  return (
    typeof d.id === "string" &&
    typeof d.filename === "string" &&
    typeof d.status === "string" &&
    STATUSES.includes(d.status) &&
    (d.status_detail === null || typeof d.status_detail === "string") &&
    (d.entity_id === null || typeof d.entity_id === "string")
  );
}

/** A document is still being worked on while the server has not finished parsing it. */
export function isWorking(status: DocumentStatus): boolean {
  return status === "uploaded" || status === "parsing";
}

export const STATUS_TEXT: Record<DocumentStatus, string> = {
  uploaded: "Waiting",
  parsing: "Reading…",
  parsed: "Ready",
  needs_review: "Needs a look",
  needs_ocr: "Scanned: no text found",
  failed: "Failed",
};

async function problem(response: Response): Promise<ApiError> {
  let text = `The server answered ${response.status}.`;
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") text = detail;
    }
  } catch {
    // keep the generic text
  }
  return new ApiError(response.status, text);
}

export async function uploadDocument(
  file: File,
  workspaceId: string,
  entityId: string | null,
): Promise<DocumentInfo> {
  const form = new FormData(); // the browser sets the multipart Content-Type itself
  form.append("file", file);
  form.append("workspace_id", workspaceId);
  if (entityId !== null) form.append("entity_id", entityId);
  const response = await fetch(`${API_URL}/documents`, { method: "POST", body: form });
  if (!response.ok) throw await problem(response);
  const body: unknown = await response.json();
  if (!isDocumentInfo(body)) throw new Error("The server sent data in an unexpected shape.");
  return body;
}

export async function getDocument(id: string): Promise<DocumentInfo> {
  const response = await fetch(`${API_URL}/documents/${id}`);
  if (!response.ok) throw await problem(response);
  const body: unknown = await response.json();
  if (!isDocumentInfo(body)) throw new Error("The server sent data in an unexpected shape.");
  return body;
}

// --- what happened to a file after it was read (intake) -------------------------------------------

export type IntakeOutcome = "assigned" | "created" | "needs_review" | "skipped";

export interface IntakeView {
  outcome: IntakeOutcome;
  entity_id: string | null; // the row the file is attached to NOW (a person may have changed it)
  entity_name: string | null;
  detected: { company?: string | null; ticker?: string | null; doc_type?: string | null; period?: string | null };
  detail: string | null;
  candidates: string[];
}

const OUTCOMES: readonly string[] = ["assigned", "created", "needs_review", "skipped"];

export function isIntakeView(v: unknown): v is IntakeView {
  if (typeof v !== "object" || v === null) return false;
  const d = v as Record<string, unknown>;
  return (
    typeof d.outcome === "string" &&
    OUTCOMES.includes(d.outcome) &&
    (d.entity_id === null || typeof d.entity_id === "string") &&
    (d.entity_name === null || typeof d.entity_name === "string") &&
    typeof d.detected === "object" &&
    d.detected !== null &&
    (d.detail === null || typeof d.detail === "string") &&
    Array.isArray(d.candidates) &&
    d.candidates.every((c) => typeof c === "string")
  );
}

/** Intake's decision for a file, or null while intake has not looked at it yet (the server says 404). */
export async function getIntake(id: string): Promise<IntakeView | null> {
  const response = await fetch(`${API_URL}/documents/${id}/intake`);
  if (response.status === 404) return null;
  if (!response.ok) throw await problem(response);
  const body: unknown = await response.json();
  if (!isIntakeView(body)) throw new Error("The server sent data in an unexpected shape.");
  return body;
}

/** Attach a file to a row by hand (null detaches it). A person's choice always wins over intake. */
export async function assignDocument(id: string, entityId: string | null): Promise<void> {
  const response = await fetch(`${API_URL}/documents/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entity_id: entityId }),
  });
  if (!response.ok) throw await problem(response);
}

// --- every file of a workspace, for the grid's documents column ---------------------------------

export interface WorkspaceDocument {
  id: string;
  filename: string;
  status: DocumentStatus;
  entity_id: string | null;
}

export function isWorkspaceDocument(v: unknown): v is WorkspaceDocument {
  if (typeof v !== "object" || v === null) return false;
  const d = v as Record<string, unknown>;
  return (
    typeof d.id === "string" &&
    typeof d.filename === "string" &&
    typeof d.status === "string" &&
    STATUSES.includes(d.status) &&
    (d.entity_id === null || typeof d.entity_id === "string")
  );
}

export async function listDocuments(workspaceId: string): Promise<WorkspaceDocument[]> {
  const response = await fetch(`${API_URL}/documents?workspace_id=${workspaceId}&limit=200`);
  if (!response.ok) throw await problem(response);
  const body: unknown = await response.json();
  if (!Array.isArray(body) || !body.every(isWorkspaceDocument)) {
    throw new Error("The server sent data in an unexpected shape.");
  }
  return body;
}
