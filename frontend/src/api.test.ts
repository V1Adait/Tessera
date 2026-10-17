import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, deleteConversation, fetchConversation, fetchConversations, streamChat } from "./api";
import type { StreamEvent } from "./types";

const encoder = new TextEncoder();

// A fake backend response: the given text chunks arrive one at a time.
function sseResponse(chunks: string[], headers: Record<string, string>): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers });
}

const ids = { "X-Conversation-ID": "conv-1", "X-Request-ID": "req-1" };

function mockFetch(response: Response): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async () => response);
  vi.stubGlobal("fetch", fn);
  return fn;
}

function call(overrides: Partial<Parameters<typeof streamChat>[0]> = {}) {
  const events: StreamEvent[] = [];
  const starts: { conversationId: string; requestId: string }[] = [];
  const promise = streamChat({
    conversationId: null,
    message: "hi",
    onStart: (s) => starts.push(s),
    onEvent: (e) => events.push(e),
    ...overrides,
  });
  return { promise, events, starts };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("streamChat", () => {
  it("sends ONE message plus the conversation id, and nothing else", async () => {
    const fetchMock = mockFetch(
      sseResponse(['data: {"type":"done","request_id":"r","usage":{"tokens_in":1,"tokens_out":1}}\n\n'], ids),
    );

    await call({ conversationId: "conv-9", message: "hello" }).promise;

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/chat$/);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      conversation_id: "conv-9",
      message: "hello",
    });
  });

  it("names the workspace only when it starts a NEW conversation", async () => {
    const done = 'data: {"type":"done","request_id":"r","usage":{"tokens_in":1,"tokens_out":1}}\n\n';
    const fresh = mockFetch(sseResponse([done], ids));
    await call({ conversationId: null, workspaceId: "ws-1", message: "hi" }).promise;
    const sent = (m: typeof fresh): Record<string, unknown> =>
      JSON.parse((m.mock.calls[0] as unknown as [string, RequestInit])[1].body as string);
    expect(sent(fresh)).toEqual({ conversation_id: null, message: "hi", workspace_id: "ws-1" });

    const existing = mockFetch(sseResponse([done], ids));
    await call({ conversationId: "conv-9", workspaceId: "ws-1", message: "hi" }).promise;
    expect(sent(existing)).toEqual({ conversation_id: "conv-9", message: "hi" }); // never both
  });

  it("reports the conversation id from the header before any event", async () => {
    mockFetch(
      sseResponse(['data: {"type":"done","request_id":"r","usage":{"tokens_in":1,"tokens_out":1}}\n\n'], ids),
    );

    const { promise, starts } = call();
    await promise;

    expect(starts).toEqual([{ conversationId: "conv-1", requestId: "req-1" }]);
  });

  it("assembles events even when a chunk ends in the middle of a message", async () => {
    mockFetch(
      sseResponse(
        [
          'data: {"type":"delta","text":"Hel"}\n\ndata: {"type":"del',
          'ta","text":"lo"}\n\n',
          'data: {"type":"done","request_id":"r","usage":{"tokens_in":3,"tokens_out":2}}\n\n',
        ],
        ids,
      ),
    );

    const { promise, events } = call();
    await promise;

    expect(events.map((e) => e.type)).toEqual(["delta", "delta", "done"]);
    const text = events.map((e) => (e.type === "delta" ? e.text : "")).join("");
    expect(text).toBe("Hello");
  });

  it("ignores events with an unexpected shape instead of crashing", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    mockFetch(
      sseResponse(
        [
          'data: {"type":"mystery"}\n\ndata: not json\n\n',
          'data: {"type":"done","request_id":"r","usage":{"tokens_in":1,"tokens_out":1}}\n\n',
        ],
        ids,
      ),
    );

    const { promise, events } = call();
    await promise;

    expect(events.map((e) => e.type)).toEqual(["done"]);
    warn.mockRestore();
  });

  it("delivers an error event as a normal final event", async () => {
    mockFetch(
      sseResponse(
        ['data: {"type":"error","request_id":"r","code":"context_too_large","message":"too big","retryable":false}\n\n'],
        ids,
      ),
    );

    const { promise, events } = call();
    await promise;

    expect(events[0]).toMatchObject({ type: "error", code: "context_too_large" });
  });

  it("throws when the stream ends without a final event", async () => {
    mockFetch(sseResponse(['data: {"type":"delta","text":"half"}\n\n'], ids));

    await expect(call().promise).rejects.toThrow(/ended before the answer was complete/);
  });

  it("throws an ApiError carrying the status for an up-front rejection", async () => {
    mockFetch(new Response("{}", { status: 404 }));

    await expect(call().promise).rejects.toMatchObject({ name: "ApiError", status: 404 });
  });

  it("throws when the server does not name the conversation", async () => {
    mockFetch(sseResponse([], {}));

    await expect(call().promise).rejects.toThrow(/did not say which conversation/);
  });
});

describe("fetchConversation", () => {
  it("returns a valid conversation", async () => {
    const body = {
      id: "c",
      title: "t",
      messages: [{ id: "1", role: "user", content: "hi", status: "complete", tokens: null }],
    };
    mockFetch(new Response(JSON.stringify(body), { status: 200 }));

    expect(await fetchConversation("c")).toEqual(body);
  });

  it("rejects a response in an unexpected shape", async () => {
    mockFetch(new Response(JSON.stringify({ nope: true }), { status: 200 }));

    await expect(fetchConversation("c")).rejects.toThrow(/unexpected shape/);
  });

  it("throws an ApiError for a missing conversation", async () => {
    mockFetch(new Response("{}", { status: 404 }));

    const error = await fetchConversation("c").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(404);
  });
});

describe("the chat history calls", () => {
  const item = { id: "c1", title: "Palantir leverage", updated_at: "2026-10-17T11:00:00Z", message_count: 4 };

  function reply(status: number, body?: unknown) {
    const mock = vi.fn().mockResolvedValue(
      new Response(body === undefined ? null : JSON.stringify(body), { status }),
    );
    vi.stubGlobal("fetch", mock);
    return mock;
  }

  afterEach(() => vi.unstubAllGlobals());

  it("lists the workspace's chats", async () => {
    const mock = reply(200, [item]);

    const found = await fetchConversations("w1");

    expect(found).toEqual([item]);
    expect((mock.mock.calls[0] as unknown as [string])[0]).toMatch(/\/workspaces\/w1\/conversations$/);
  });

  it("refuses a list in the wrong shape", async () => {
    reply(200, [{ id: "c1" }]);

    await expect(fetchConversations("w1")).rejects.toThrow("unexpected shape");
  });

  it("turns a server error into an ApiError with the status", async () => {
    reply(404, { detail: "workspace not found" });

    const error = await fetchConversations("w1").catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(404);
  });

  it("deletes a chat with DELETE and resolves on 204", async () => {
    const mock = reply(204);

    await expect(deleteConversation("c1")).resolves.toBeUndefined();

    const [url, init] = mock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/conversations\/c1$/);
    expect(init.method).toBe("DELETE");
  });

  it("rejects when the chat cannot be deleted", async () => {
    reply(404);

    await expect(deleteConversation("c1")).rejects.toBeInstanceOf(ApiError);
  });
});
