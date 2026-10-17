import { Check, ChevronRight, Loader2, ShieldAlert, X } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { groupSteps, stepsHeadline, type UiTool } from "./stepsModel";
import type { ToolStatus } from "./types";

function Icon({ status }: { status: ToolStatus }) {
  if (status === "running") return <Loader2 className="size-3 animate-spin" />;
  if (status === "ok") return <Check className="size-3 text-[#1c3829]" />;
  if (status === "needs_confirmation") return <ShieldAlert className="size-3 text-[#cb912f]" />;
  return <X className="size-3 text-destructive" />;
}

interface Props {
  tools: UiTool[];
  working: boolean; // the agent is still busy and has not started writing its answer
}

/** What the agent did, shown ABOVE its answer. Open while it works, folded once the answer starts;
 * the person can open or close it at any time. */
export default function ToolSteps({ tools, working }: Props) {
  const [chosen, setChosen] = useState<boolean | null>(null); // null: follow `working`
  const open = chosen ?? working;
  const groups = groupSteps(tools);

  return (
    <div className="mb-1.5 text-xs text-muted-foreground">
      <button
        className="flex items-center gap-1 rounded px-1 py-0.5 hover:bg-border/50"
        onClick={() => setChosen(!open)}
        aria-expanded={open}
      >
        <ChevronRight className={cn("size-3 transition-transform", open && "rotate-90")} />
        {stepsHeadline(groups, working)}
      </button>
      {open && (
        <ul className="ml-4 mt-0.5 flex flex-col gap-0.5">
          {groups.map((g) => (
            <li key={g.key} className="flex items-center gap-1.5" title={g.summary}>
              <Icon status={g.status} />
              <span>
                {g.label}
                {g.count > 1 && ` ×${g.count}`}
              </span>
              {g.summary && g.status !== "running" && <span className="truncate">· {g.summary}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
