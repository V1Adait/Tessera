import { Columns3 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { columnBadge, planCost, planHeadline, type PlanPayload } from "./planModel";
import { TYPE_ICON, TYPE_LABEL } from "./matrixModel";
import type { ColumnType } from "./matrixTypes";

/** What a question turned into: the columns it will add, how many cells that fills and what it
 * costs. Nothing has been created yet: the buttons next to it decide. */
export default function PlanCard({ plan }: { plan: PlanPayload }) {
  return (
    <div className="flex flex-col gap-2 text-sm">
      <div className="flex items-center gap-1.5 font-medium">
        <Columns3 className="size-4 text-primary" />
        Plan for: “{plan.question}”
      </div>
      <p className="text-muted-foreground">{planHeadline(plan)}</p>
      <ul className="flex flex-col divide-y rounded-md border">
        {plan.columns.map((c) => (
          <li key={c.name} className="flex items-start gap-2 px-2 py-1.5" title={c.prompt}>
            <span className="mt-0.5 w-4 shrink-0 text-center text-xs text-muted-foreground">
              {c.derived ? "ƒ" : (TYPE_ICON[c.type as ColumnType] ?? "·")}
            </span>
            <span className="min-w-0 flex-1">
              <span className="font-medium">{c.name}</span>
              <span className="ml-1.5 text-xs text-muted-foreground">
                {TYPE_LABEL[c.type as ColumnType] ?? c.type}
              </span>
              <span className="block truncate text-xs text-muted-foreground">
                {c.inputs && c.inputs.length > 0 ? `Computed from ${c.inputs.join(", ")}` : c.prompt}
              </span>
              {c.allowed_values.length > 0 && (
                <span className="mt-0.5 flex flex-wrap gap-1">
                  {c.allowed_values.map((v) => (
                    <Badge key={v}>{v}</Badge>
                  ))}
                </span>
              )}
            </span>
            <Badge tone={columnBadge(c).tone}>{columnBadge(c).label}</Badge>
          </li>
        ))}
      </ul>
      <p className="text-muted-foreground">{planCost(plan)}</p>
      {plan.notes.length > 0 && (
        <ul className="text-xs text-[#cb912f]">
          {plan.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
