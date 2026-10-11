import { describe, expect, it } from "vitest";
import { parseSseChunk } from "./sse";

describe("parseSseChunk", () => {
  it("reads one complete message", () => {
    const r = parseSseChunk('data: {"type":"delta","text":"hi"}\n\n');
    expect(r.payloads).toEqual(['{"type":"delta","text":"hi"}']);
    expect(r.rest).toBe("");
  });

  it("reads several messages in one chunk", () => {
    const r = parseSseChunk("data: a\n\ndata: b\n\ndata: c\n\n");
    expect(r.payloads).toEqual(["a", "b", "c"]);
    expect(r.rest).toBe("");
  });

  it("keeps an unfinished message for the next chunk", () => {
    const r = parseSseChunk("data: a\n\ndata: par");
    expect(r.payloads).toEqual(["a"]);
    expect(r.rest).toBe("data: par");
  });

  it("completes a message that was split across two chunks", () => {
    const first = parseSseChunk("data: hel");
    expect(first.payloads).toEqual([]);
    const second = parseSseChunk(first.rest + "lo\n\n");
    expect(second.payloads).toEqual(["hello"]);
  });

  it("handles a split exactly between the two newlines", () => {
    const first = parseSseChunk("data: x\n");
    expect(first.payloads).toEqual([]);
    const second = parseSseChunk(first.rest + "\n");
    expect(second.payloads).toEqual(["x"]);
  });

  it("ignores lines that are not data lines", () => {
    const r = parseSseChunk(": keep-alive\n\ndata: ok\n\n");
    expect(r.payloads).toEqual(["ok"]);
  });

  it("returns nothing for an empty buffer", () => {
    expect(parseSseChunk("")).toEqual({ payloads: [], rest: "" });
  });
});
