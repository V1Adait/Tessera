import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cancelAction,
  confirmAction,
  fetchPendingActions,
  isActionResult,
  isPendingAction,
  runIdOf,
} from "./agentApi";

function respond(status: number, body: unknown): ReturnType<typeof vi.fn> {
  const mock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }),
  );
  vi.stubGlobal("fetch", mock);
  return mock;
}

afterEach(() => vi.unstubAllGlobals());

describe("the checks", () => {
  it("accept real shapes and reject wrong ones", () => {
    expect(isPendingAction({ id: "a", tool: "remove_row", risk: "costly", summary: "s" })).toBe(true);
    expect(isPendingAction({ id: "a", tool: "t", risk: "mild", summary: "s" })).toBe(false);
    expect(isActionResult({ ok: true, message: "m", changed_matrix: true })).toBe(true);
    expect(isActionResult({ ok: "yes", message: "m" })).toBe(false);
  });
});

describe("confirming and cancelling", () => {
  it("POSTs to the action and returns what happened", async () => {
    const mock = respond(200, { ok: true, message: "Removed row 'Acme'.", changed_matrix: true });

    const result = await confirmAction("a1");

    const [url, init] = mock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/actions\/a1\/confirm$/);
    expect(init.method).toBe("POST");
    expect(result).toEqual({ ok: true, message: "Removed row 'Acme'.", changed_matrix: true, runId: null });
  });

  it("cancel says the matrix did not change", async () => {
    respond(200, { ok: true, message: "Cancelled. Nothing was changed." });

    expect((await cancelAction("a1")).changed_matrix).toBe(false);
  });

  it("turns an already-answered action (409) into an error with the server's words", async () => {
    respond(409, { detail: "this action was already confirmed or cancelled" });

    await expect(confirmAction("a1")).rejects.toThrow("already confirmed or cancelled");
  });

  it("refuses data in an unexpected shape", async () => {
    respond(200, { nope: true });

    await expect(confirmAction("a1")).rejects.toThrow("unexpected shape");
  });
});

describe("fetchPendingActions", () => {
  it("returns the actions that still wait for a yes", async () => {
    respond(200, [{ id: "a", tool: "remove_row", risk: "destructive", summary: "s" }]);

    expect(await fetchPendingActions("c1")).toHaveLength(1);
  });
});

describe("the run a confirmed plan started", () => {
  it("is passed on so the answer can wait for it", async () => {
    respond(200, { ok: true, message: "Added 3 columns", changed_matrix: true, data: { run_id: "run-7" } });

    expect((await confirmAction("a1")).runId).toBe("run-7");
  });

  it("is null when nothing was started, or the data is odd", () => {
    expect(runIdOf({})).toBeNull();
    expect(runIdOf({ data: null })).toBeNull();
    expect(runIdOf({ data: { run_id: 5 } })).toBeNull();
    expect(runIdOf({ data: { columns: ["A"] } })).toBeNull();
  });
});
