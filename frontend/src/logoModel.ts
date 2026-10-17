// Company logos: pure helpers. No React, no network, so each rule is easy to test.
//
// A logo is looked up from the company's website domain on an OUTSIDE icon service. That tells the
// service which companies are being researched, which can matter in diligence, so it is OFF until the
// person switches it on, and a company without a domain just gets an initials avatar.

import type { Tone } from "./matrixModel";
import { tagTone } from "./matrixModel";

// `{domain}` is replaced by the company's domain. Change it with VITE_LOGO_URL to use another service
// (or a proxy of your own) without touching the code.
export const DEFAULT_LOGO_TEMPLATE = "https://www.google.com/s2/favicons?domain={domain}&sz=64";

export const LOGO_TEMPLATE: string = import.meta.env.VITE_LOGO_URL ?? DEFAULT_LOGO_TEMPLATE;

/** The URL of a company's logo, or null when there is nothing safe to look up. */
export function logoUrl(template: string, domain: string | null | undefined): string | null {
  if (!domain) return null;
  // Only a plain domain goes into the URL: nothing a person typed can add a path, a query or a host.
  if (!/^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9][a-z0-9-]*[a-z0-9]$/.test(domain)) return null;
  if (!template.includes("{domain}")) return null;
  return template.replace("{domain}", encodeURIComponent(domain));
}

/** Up to two letters for the avatar: "Palantir Technologies" -> "PT", "Visa" -> "VI". */
export function initialsOf(name: string): string {
  const words = name
    .split(/[^\p{L}\p{N}]+/u)
    .filter((w) => w !== "")
    .map((w) => Array.from(w));
  if (words.length === 0) return "?";
  const first = words[0]!;
  const letters = words.length >= 2 ? [first[0]!, words[1]![0]!] : first.slice(0, 2);
  return letters.join("").toUpperCase();
}

/** The same company is always the same colour. */
export function avatarTone(name: string): Tone {
  return tagTone(name.trim().toLowerCase() || "?");
}

const KEY = "tessera.logos";

// localStorage can throw (private windows, blocked storage): the setting then simply is not kept.
export function readLogosSetting(): boolean {
  try {
    return localStorage.getItem(KEY) === "on";
  } catch {
    return false;
  }
}

export function writeLogosSetting(on: boolean): void {
  try {
    localStorage.setItem(KEY, on ? "on" : "off");
  } catch {
    // not remembering the choice is acceptable
  }
}
