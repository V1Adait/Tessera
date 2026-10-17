import type { Tone } from "@/matrixModel";

// Soft tag colours: a tinted background with darker text of the same hue. Shared by the status
// badges and the company avatars, so one company or one status is always the same colour.
export const TONE_CLASSES: Record<Tone, string> = {
  gray: "bg-[#e3e2e0] text-[#32302c]",
  green: "bg-[#dbeddb] text-[#1c3829]",
  amber: "bg-[#fdecc8] text-[#402c1b]",
  red: "bg-[#ffe2dd] text-[#5d1715]",
  blue: "bg-[#d3e5ef] text-[#183347]",
  purple: "bg-[#e8deee] text-[#412454]",
  orange: "bg-[#fadec9] text-[#49290e]",
};
