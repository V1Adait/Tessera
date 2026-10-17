import { Dialog as D } from "radix-ui";
import { X } from "lucide-react";
import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

export const Dialog = D.Root;
export const DialogTitle = D.Title;
export const DialogDescription = D.Description;

/** A centred modal (`side="center"`) or a panel sliding in from the right (`side="right"`).
 * Radix supplies the hard parts: focus trap, Escape to close, scroll lock and screen-reader roles. */
export function DialogContent({
  side = "center",
  className,
  children,
  ...rest
}: ComponentProps<typeof D.Content> & { side?: "center" | "right" }) {
  return (
    <D.Portal>
      <D.Overlay className="fixed inset-0 z-40 bg-black/20" />
      <D.Content
        className={cn(
          "fixed z-50 flex flex-col gap-3 overflow-y-auto border bg-background p-5 shadow-xl focus:outline-none",
          side === "center"
            ? "left-1/2 top-1/2 max-h-[90vh] w-[min(480px,92vw)] -translate-x-1/2 -translate-y-1/2 rounded-md"
            : "inset-y-0 right-0 w-[min(420px,100vw)] border-l",
          className,
        )}
        {...rest}
      >
        {children}
        <D.Close
          className="absolute right-3 top-3 rounded-md p-1 text-muted-foreground hover:bg-muted"
          aria-label="Close"
        >
          <X className="size-4" />
        </D.Close>
      </D.Content>
    </D.Portal>
  );
}
