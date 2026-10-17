import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";
import type { Tone } from "@/matrixModel";
import { TONE_CLASSES } from "./tones";

export function Badge({
  tone = "gray",
  className,
  ...rest
}: HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) {
  return (
    <span
      className={cn("inline-block max-w-full truncate rounded-sm px-1.5 py-0.5 text-xs leading-4", TONE_CLASSES[tone], className)}
      {...rest}
    />
  );
}
