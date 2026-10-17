import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_LOGO_TEMPLATE,
  avatarTone,
  initialsOf,
  logoUrl,
  readLogosSetting,
  writeLogosSetting,
} from "./logoModel";

describe("logoUrl", () => {
  it("puts a plain domain into the template", () => {
    expect(logoUrl(DEFAULT_LOGO_TEMPLATE, "palantir.com")).toBe(
      "https://www.google.com/s2/favicons?domain=palantir.com&sz=64",
    );
  });

  it("is null when there is no domain, so nothing is ever looked up", () => {
    expect(logoUrl(DEFAULT_LOGO_TEMPLATE, null)).toBeNull();
    expect(logoUrl(DEFAULT_LOGO_TEMPLATE, undefined)).toBeNull();
    expect(logoUrl(DEFAULT_LOGO_TEMPLATE, "")).toBeNull();
  });

  it("refuses anything that is not a plain domain: no path, query, host or scheme gets into the URL", () => {
    for (const bad of ["palantir.com/x", "a.com?x=1", "evil.com#", "https://a.com", "a b.com", "a.com:80", "palantir", "-.com", "../x"]) {
      expect(logoUrl(DEFAULT_LOGO_TEMPLATE, bad)).toBeNull();
    }
  });

  it("refuses a template that has nowhere to put the domain", () => {
    expect(logoUrl("https://example.com/logo.png", "palantir.com")).toBeNull();
  });
});

describe("initialsOf", () => {
  it("takes the first letter of the first two words, or two letters of one word", () => {
    expect(initialsOf("Palantir Technologies")).toBe("PT");
    expect(initialsOf("Meta Platforms Inc")).toBe("MP");
    expect(initialsOf("Visa")).toBe("VI");
    expect(initialsOf("A")).toBe("A");
  });

  it("copes with punctuation, case and other alphabets", () => {
    expect(initialsOf("  coca-cola co. ")).toBe("CC");
    expect(initialsOf("société générale")).toBe("SG");
    expect(initialsOf("3M")).toBe("3M");
  });

  it("never returns nothing", () => {
    expect(initialsOf("")).toBe("?");
    expect(initialsOf("  ...  ")).toBe("?");
  });
});

describe("avatarTone", () => {
  it("is stable: the same company is always the same colour, whatever its case", () => {
    expect(avatarTone("Palantir")).toBe(avatarTone("palantir "));
    expect(avatarTone("Palantir")).toBe(avatarTone("Palantir"));
  });
});

describe("the logos switch", () => {
  afterEach(() => vi.unstubAllGlobals());

  function fakeStorage(initial: Record<string, string> = {}) {
    const data = { ...initial };
    vi.stubGlobal("localStorage", {
      getItem: (k: string) => data[k] ?? null,
      setItem: (k: string, v: string) => {
        data[k] = v;
      },
    });
    return data;
  }

  it("is off until the person turns it on", () => {
    fakeStorage();

    expect(readLogosSetting()).toBe(false);
  });

  it("remembers the choice", () => {
    const data = fakeStorage();

    writeLogosSetting(true);
    expect(readLogosSetting()).toBe(true);
    writeLogosSetting(false);
    expect(readLogosSetting()).toBe(false);
    expect(data["tessera.logos"]).toBe("off");
  });

  it("stays off, without throwing, when storage is blocked", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    });

    expect(readLogosSetting()).toBe(false);
    expect(() => writeLogosSetting(true)).not.toThrow();
  });
});
