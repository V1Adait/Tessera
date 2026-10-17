import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "./api";
import { deleteColumn, exportPdf, updateEntity } from "./matrixApi";

function respond(status: number, body?: unknown): ReturnType<typeof vi.fn> {
  const mock = vi.fn().mockResolvedValue(
    new Response(body === undefined ? null : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", mock);
  return mock;
}

afterEach(() => vi.unstubAllGlobals());

describe("deleteColumn", () => {
  it("sends DELETE to the column and resolves on 204", async () => {
    const mock = respond(204);

    await expect(deleteColumn("w1", "c1")).resolves.toBeUndefined();

    const [url, init] = mock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/workspaces\/w1\/columns\/c1$/);
    expect(init.method).toBe("DELETE");
  });

  it("rejects with the server's own reason when it refuses (409)", async () => {
    respond(409, { detail: "Cost ratio is computed from 'Revenue'. Remove or change it first." });

    const error = await deleteColumn("w1", "c1").catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(409);
    expect((error as ApiError).message).toContain("computed from 'Revenue'");
  });

  it("rejects a missing column (404) with a readable message", async () => {
    respond(404, { detail: "column not found" });

    await expect(deleteColumn("w1", "c1")).rejects.toThrow("column not found");
  });

  it("falls back to a plain message when the body is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("boom", { status: 500 })));

    await expect(deleteColumn("w1", "c1")).rejects.toThrow("The server answered 500.");
  });
});

describe("exportPdf", () => {
  it("returns the file the server built", async () => {
    const mock = vi.fn().mockResolvedValue(
      new Response("%PDF-1.4 fake", { status: 200, headers: { "Content-Type": "application/pdf" } }),
    );
    vi.stubGlobal("fetch", mock);

    const blob = await exportPdf("w1");

    expect(await blob.text()).toBe("%PDF-1.4 fake");
    expect((mock.mock.calls[0] as unknown as [string])[0]).toMatch(/\/workspaces\/w1\/export\/pdf$/);
  });

  it("rejects with the server's advice when the matrix is too big", async () => {
    respond(422, { detail: "This matrix has 300 companies and 5 questions. Remove some, or export fewer." });

    await expect(exportPdf("w1")).rejects.toThrow("Remove some, or export fewer.");
  });
});

describe("updateEntity", () => {
  const row = { id: "e1", workspace_id: "w1", name: "Palantir", position: 1, domain: "palantir.com", ticker: "PLTR" };

  it("sends only what changed and returns the updated company", async () => {
    const mock = respond(200, row);

    const updated = await updateEntity("w1", "e1", { domain: "palantir.com" });

    const [url, init] = mock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/workspaces\/w1\/entities\/e1$/);
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body as string)).toEqual({ domain: "palantir.com" });
    expect(updated.domain).toBe("palantir.com");
  });

  it("can clear a field by sending null", async () => {
    const mock = respond(200, { ...row, ticker: null });

    await updateEntity("w1", "e1", { ticker: null });

    expect(JSON.parse((mock.mock.calls[0] as unknown as [string, RequestInit])[1].body as string)).toEqual({ ticker: null });
  });

  it("rejects with the reason when another row has that ticker", async () => {
    respond(409, { detail: "the ticker PLTR already belongs to 'Palantir'" });

    await expect(updateEntity("w1", "e1", { ticker: "PLTR" })).rejects.toThrow("already belongs to 'Palantir'");
  });
});
