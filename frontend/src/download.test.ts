import { describe, expect, it } from "vitest";
import { pdfFileName } from "./download";

const DAY = new Date("2026-10-17T12:30:00Z");

describe("pdfFileName", () => {
  it("names the file after the workspace and the day", () => {
    expect(pdfFileName("PE diligence", DAY)).toBe("Tessera - PE diligence - 2026-10-17.pdf");
  });

  it("replaces characters a file system forbids, so the download never fails on the name", () => {
    expect(pdfFileName('Fund A/B: "Q3" <draft>?', DAY)).toBe("Tessera - Fund A-B- -Q3- -draft - 2026-10-17.pdf");
  });

  it("never lets a name climb out of the downloads folder", () => {
    const name = pdfFileName("../../etc/passwd", DAY);

    expect(name).not.toContain("/");
    expect(name).not.toContain("..");
    expect(name.endsWith(".pdf")).toBe(true);
  });

  it("falls back to a plain name when nothing usable is left", () => {
    expect(pdfFileName("///", DAY)).toBe("Tessera - matrix - 2026-10-17.pdf");
    expect(pdfFileName("   ", DAY)).toBe("Tessera - matrix - 2026-10-17.pdf");
  });

  it("keeps a very long name short", () => {
    expect(pdfFileName("x".repeat(300), DAY).length).toBeLessThan(100);
  });

  it("keeps ordinary non-English letters", () => {
    expect(pdfFileName("Société Générale", DAY)).toContain("Société Générale");
  });
});

describe("pdfFileName and control characters", () => {
  it("drops control characters such as newlines and tabs", () => {
    expect(pdfFileName("Fund\nA\tB\u0000", DAY)).toBe("Tessera - FundAB - 2026-10-17.pdf");
  });
});
