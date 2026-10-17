// The calls the grid makes. Every response is checked before it is trusted (see matrixTypes.ts).

import { API_URL, ApiError } from "./api";
import {
  isCell,
  isColumn,
  isEntity,
  isListOf,
  isPlan,
  isRun,
  isWorkspace,
  type Cell,
  type Column,
  type ColumnDraft,
  type Entity,
  type Plan,
  type Run,
  type RunScope,
  type Workspace,
} from "./matrixTypes";

const REQUEST_TIMEOUT_MS = 15_000;

// FastAPI reports a problem as {"detail": "text"} (or a list for a malformed body). Show the text.
function messageFrom(body: unknown, status: number): string {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      const first: unknown = detail[0];
      if (typeof first === "object" && first !== null && "msg" in first) {
        return String((first as { msg: unknown }).msg);
      }
    }
  }
  return `The server answered ${status}.`;
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function request<T>(
  path: string,
  guard: (value: unknown) => value is T,
  init?: RequestInit,
): Promise<T> {
  // A server that accepts the connection but never answers would leave the screen on "Loading…"
  // for ever, so every call has a time limit.
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "TimeoutError") {
      throw new Error("The server did not answer in time. Is the backend running?");
    }
    throw err;
  }
  const body = await readJson(response);
  if (!response.ok) throw new ApiError(response.status, messageFrom(body, response.status));
  if (!guard(body)) throw new Error("The server sent data in an unexpected shape.");
  return body;
}

function post<T>(path: string, body: unknown, guard: (v: unknown) => v is T): Promise<T> {
  return request(path, guard, { method: "POST", body: JSON.stringify(body) });
}

// --- workspaces, rows, columns, cells -----------------------------------------------------------

export const listWorkspaces = (): Promise<Workspace[]> =>
  request("/workspaces", isListOf(isWorkspace));

export const createWorkspace = (name: string): Promise<Workspace> =>
  post("/workspaces", { name }, isWorkspace);

export const listEntities = (workspaceId: string): Promise<Entity[]> =>
  request(`/workspaces/${workspaceId}/entities`, isListOf(isEntity));

export const createEntity = (workspaceId: string, name: string): Promise<Entity> =>
  post(`/workspaces/${workspaceId}/entities`, { name }, isEntity);

/** Set or clear a company's website and/or ticker. A field left out is kept; null clears it. Rejects
 * with the server's reason (409: another row has that ticker; 422: not a valid domain or ticker). */
export const updateEntity = (
  workspaceId: string,
  entityId: string,
  details: { domain?: string | null; ticker?: string | null },
): Promise<Entity> =>
  request(`/workspaces/${workspaceId}/entities/${entityId}`, isEntity, {
    method: "PATCH",
    body: JSON.stringify(details),
  });

/** The workspace as a PDF. Rejects with the server's reason (422: too big to fit in one readable PDF).
 * Building it can take a few seconds for a large matrix, so the time limit is longer than usual. */
export async function exportPdf(workspaceId: string): Promise<Blob> {
  const response = await fetch(`${API_URL}/workspaces/${workspaceId}/export/pdf`, {
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) throw new ApiError(response.status, messageFrom(await readJson(response), response.status));
  return await response.blob();
}

export const listColumns = (workspaceId: string): Promise<Column[]> =>
  request(`/workspaces/${workspaceId}/columns`, isListOf(isColumn));

export const createColumn = (workspaceId: string, draft: ColumnDraft): Promise<Column> =>
  post(`/workspaces/${workspaceId}/columns`, draft, isColumn);

export const updateColumn = (
  workspaceId: string,
  columnId: string,
  patch: Partial<ColumnDraft> & { clear_applicability?: boolean },
): Promise<Column> =>
  request(`/workspaces/${workspaceId}/columns/${columnId}`, isColumn, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });

/** Remove a column and its cells. Rejects with the server's own reason when it refuses (409: a fill is
 * going, or computed columns are built on this one). Cannot be undone: the caller has asked first. */
export async function deleteColumn(workspaceId: string, columnId: string): Promise<void> {
  const response = await fetch(`${API_URL}/workspaces/${workspaceId}/columns/${columnId}`, {
    method: "DELETE",
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  if (response.status === 204) return;
  throw new ApiError(response.status, messageFrom(await readJson(response), response.status));
}

export const listCells = (workspaceId: string): Promise<Cell[]> =>
  request(`/workspaces/${workspaceId}/cells`, isListOf(isCell));

export const extractCell = (
  workspaceId: string,
  entityId: string,
  columnId: string,
  refresh: boolean,
): Promise<Cell> =>
  post(
    `/workspaces/${workspaceId}/cells/extract`,
    { entity_id: entityId, column_id: columnId, refresh },
    isCell,
  );

// --- runs ---------------------------------------------------------------------------------------

export const planRun = (workspaceId: string, scope: RunScope): Promise<Plan> =>
  post(`/workspaces/${workspaceId}/runs/plan`, { scope }, isPlan);

export type StartResult =
  | { kind: "started"; run: Run }
  | { kind: "needs_confirmation"; message: string; plan: Plan };

/** Start a run. A costly or large run is refused with its plan until `confirmed` is true: that
 * refusal is not an error, it is the question "do you want to spend this?". */
export async function startRun(
  workspaceId: string,
  scope: RunScope,
  confirmed: boolean,
): Promise<StartResult> {
  const response = await fetch(`${API_URL}/workspaces/${workspaceId}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scope, confirmed }),
  });
  const body = await readJson(response);
  if (response.ok) {
    if (!isRun(body)) throw new Error("The server sent data in an unexpected shape.");
    return { kind: "started", run: body };
  }
  if (response.status === 409 && typeof body === "object" && body !== null && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "object" && detail !== null && "plan" in detail) {
      const { message, plan } = detail as { message?: unknown; plan: unknown };
      if (isPlan(plan)) {
        return {
          kind: "needs_confirmation",
          message: typeof message === "string" ? message : "This run needs your confirmation.",
          plan,
        };
      }
    }
  }
  throw new ApiError(response.status, messageFrom(body, response.status));
}

export const getRun = (workspaceId: string, runId: string): Promise<Run> =>
  request(`/workspaces/${workspaceId}/runs/${runId}`, isRun);

export const listRuns = (workspaceId: string): Promise<Run[]> =>
  request(`/workspaces/${workspaceId}/runs`, isListOf(isRun));

export const stopRun = (workspaceId: string, runId: string): Promise<Run> =>
  post(`/workspaces/${workspaceId}/runs/${runId}/stop`, {}, isRun);
