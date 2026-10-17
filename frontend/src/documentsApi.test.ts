import { describe, expect, it } from "vitest";
import { isDocumentInfo, isWorking } from "./documentsApi";

const doc = { id: "d", filename: "a.pdf", status: "parsed", status_detail: null, entity_id: null, duplicate: false };

describe("isDocumentInfo", () => {
  it("accepts a real upload response", () => {
    expect(isDocumentInfo(doc)).toBe(true);
    expect(isDocumentInfo({ ...doc, entity_id: "row-1", status_detail: "ok" })).toBe(true);
  });

  it("rejects unknown statuses and missing parts", () => {
    expect(isDocumentInfo({ ...doc, status: "weird" })).toBe(false);
    expect(isDocumentInfo({ ...doc, filename: 5 })).toBe(false);
    expect(isDocumentInfo(null)).toBe(false);
  });
});

describe("isWorking", () => {
  it("is true only until the server has finished reading the file", () => {
    expect(isWorking("uploaded")).toBe(true);
    expect(isWorking("parsing")).toBe(true);
    for (const done of ["parsed", "needs_review", "needs_ocr", "failed"] as const) {
      expect(isWorking(done)).toBe(false);
    }
  });
});
