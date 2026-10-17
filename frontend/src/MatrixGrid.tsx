import { flexRender, getCoreRowModel, useReactTable, type ColumnDef } from "@tanstack/react-table";
import { Check, FileText, Play, Plus, Trash2, TriangleAlert } from "lucide-react";
import { useMemo, useState, type KeyboardEvent } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { STATUS_TEXT, type WorkspaceDocument } from "./documentsApi";
import EntityLogo from "./EntityLogo";
import { documentsByRow } from "./intakeModel";
import { TYPE_ICON, cellKey, cellView, hasTags, isNumeric, tagTone } from "./matrixModel";
import type { Cell, Column, Entity } from "./matrixTypes";

export interface Selected {
  entityId: string;
  columnId: string;
}

interface Props {
  entities: Entity[];
  columns: Column[];
  cells: Map<string, Cell>;
  documents: WorkspaceDocument[]; // every file of the workspace: the second column lists each row's
  selected: Selected | null;
  busy: boolean; // a run is going: its buttons are off
  onSelect: (s: Selected) => void;
  onEditColumn: (c: Column) => void;
  onAddColumn: () => void;
  onFillColumn: (c: Column) => void;
  onDeleteColumn: (c: Column) => void;
  onEditEntity: (e: Entity) => void;
  showLogos: boolean;
  onFillRow: (e: Entity) => void;
  onAddRow: (name: string) => Promise<void>;
}

const DOCUMENTS_COLUMN = "documents"; // the built-in column's id; a real column id is a UUID

// What every column's header and cells have in common.
const TH = "h-8 whitespace-nowrap border-b border-r bg-background px-2 text-left text-sm font-normal text-muted-foreground";
const TD = "h-9 max-w-80 truncate border-b border-r px-2 text-sm";

/** The table. TanStack Table owns the structure (which columns exist, in what order, how a row is
 * turned into cells); this file owns how it looks. */
export default function MatrixGrid({
  entities,
  columns,
  cells,
  documents,
  selected,
  busy,
  onSelect,
  onEditColumn,
  onAddColumn,
  onFillColumn,
  onDeleteColumn,
  onEditEntity,
  showLogos,
  onFillRow,
  onAddRow,
}: Props) {
  const [adding, setAdding] = useState(false);
  const [rowName, setRowName] = useState("");
  const filesByRow = useMemo(() => documentsByRow(documents), [documents]);

  const defs = useMemo<ColumnDef<Entity>[]>(
    () => [
      {
        id: "name",
        header: () => "Row",
        cell: ({ row }) => (
          <span className="group/row flex items-center gap-1.5">
            <span className="w-5 text-right text-xs text-muted-foreground">{row.index + 1}</span>
            <button
              className="flex min-w-0 items-center gap-1.5 text-left"
              onClick={() => onEditEntity(row.original)}
              title="Edit the company's website and ticker"
            >
              <EntityLogo name={row.original.name} domain={row.original.domain} showLogos={showLogos} />
              <span className="truncate font-medium">{row.original.name}</span>
            </button>
            {row.original.ticker && (
              <span className="shrink-0 text-xs text-muted-foreground">{row.original.ticker}</span>
            )}
            <Button
              variant="ghost"
              size="icon"
              className="ml-auto size-5 opacity-0 group-hover/row:opacity-100"
              disabled={busy}
              onClick={() => onFillRow(row.original)}
              title={`Fill the row “${row.original.name}”`}
            >
              <Play className="size-3" />
            </Button>
          </span>
        ),
      },
      {
        // Built in: not a question and not filled by a model. It lists the files each row's answers
        // come from, so it can never disagree with the documents themselves.
        id: DOCUMENTS_COLUMN,
        header: () => (
          <span className="flex items-center gap-1">
            <FileText className="size-3.5" /> Documents
          </span>
        ),
        cell: ({ row }) => {
          const files = filesByRow.get(row.original.id) ?? [];
          if (files.length === 0) {
            return <span className="text-xs text-muted-foreground">No files yet</span>;
          }
          return (
            <span className="flex flex-wrap gap-1 py-1">
              {files.map((f) => (
                <span
                  key={f.id}
                  className="flex max-w-44 items-center gap-1 rounded-sm border bg-background px-1.5 py-0.5 text-xs"
                  title={`${f.filename} · ${STATUS_TEXT[f.status]}`}
                >
                  <FileText className="size-3 shrink-0 text-muted-foreground" />
                  <span className="truncate">{f.filename}</span>
                  {f.status !== "parsed" && <span className="shrink-0 text-muted-foreground">· {STATUS_TEXT[f.status]}</span>}
                </span>
              ))}
            </span>
          );
        },
      },
      ...columns.map<ColumnDef<Entity>>((col) => ({
        id: col.id,
        header: () => (
          <span className="group/col flex items-center gap-1" title={col.prompt}>
            <button className="flex items-center gap-1 hover:text-foreground" onClick={() => onEditColumn(col)}>
              <span className="text-xs" title={col.derivation ? "Computed in code from other columns" : undefined}>
                {col.derivation ? "ƒ" : TYPE_ICON[col.type]}
              </span>
              {col.name}
            </button>
            <Button
              variant="ghost"
              size="icon"
              className="size-5 opacity-0 group-hover/col:opacity-100"
              disabled={busy}
              onClick={() => onFillColumn(col)}
              title={`Fill the “${col.name}” column`}
            >
              <Play className="size-3" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="size-5 text-muted-foreground opacity-0 group-hover/col:opacity-100"
              onClick={() => onDeleteColumn(col)}
              title={`Remove the “${col.name}” column`}
              aria-label={`Remove the ${col.name} column`}
            >
              <Trash2 className="size-3" />
            </Button>
          </span>
        ),
        cell: ({ row }) => {
          const view = cellView(cells.get(cellKey(row.original.id, col.id)));
          return (
            <span className={cn("flex items-center gap-1.5", isNumeric(col.type) && "justify-end")}>
              {view.kind === "status" && <Badge tone={view.tone}>{view.text}</Badge>}
              {view.kind === "value" &&
                (hasTags(col.type) ? (
                  <Badge tone={tagTone(view.text)}>{view.text}</Badge>
                ) : (
                  <span className={cn("truncate", view.tone === "amber" && "text-[#cb912f]")}>{view.text}</span>
                ))}
              {view.period && (
                <span className="shrink-0 text-xs text-muted-foreground" title="The period this figure covers">
                  {view.period}
                </span>
              )}
              {view.derived && (
                <span className="shrink-0 text-xs italic text-muted-foreground" title="Computed in code from other cells">
                  ƒ
                </span>
              )}
              {view.check === "ties" && (
                <span className="shrink-0 text-[#1c3829]" title="Ties out with the other figures">
                  <Check className="size-3" />
                </span>
              )}
              {view.check === "does_not_tie" && (
                <span className="shrink-0 text-[#cb912f]" title="Does not tie out: open the cell to see why">
                  <TriangleAlert className="size-3" />
                </span>
              )}
              {view.stale && (
                <span className="text-[#cb912f]" title="The question changed after this was filled">
                  ●
                </span>
              )}
              {view.pinned && (
                <span className="text-muted-foreground" title="Set by a person">
                  ⚑
                </span>
              )}
            </span>
          );
        },
      })),
    ],
    [columns, cells, filesByRow, busy, showLogos, onEditColumn, onFillColumn, onDeleteColumn, onEditEntity, onFillRow],
  );

  const table = useReactTable({ data: entities, columns: defs, getCoreRowModel: getCoreRowModel() });

  function onRowKey(e: KeyboardEvent<HTMLInputElement>): void {
    if (e.key === "Enter" && rowName.trim() !== "") {
      void onAddRow(rowName.trim()).then(() => setRowName(""));
    }
    if (e.key === "Escape") {
      setAdding(false);
      setRowName("");
    }
  }

  return (
    <div className="flex-1 overflow-auto">
      <table className="w-max min-w-full border-separate border-spacing-0">
        <thead className="sticky top-0 z-10">
          {table.getHeaderGroups().map((group) => (
            <tr key={group.id}>
              {group.headers.map((header, i) => (
                <th key={header.id} className={cn(TH, i === 0 && "sticky left-0 z-20 min-w-52")}>
                  {flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
              <th className={cn(TH, "w-10 px-1")}>
                <Button variant="ghost" size="icon" onClick={onAddColumn} aria-label="Add column">
                  <Plus className="size-4" />
                </Button>
              </th>
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="hover:bg-muted/60">
              {row.getVisibleCells().map((cell, i) => {
                // The row name and the documents column are not questions: nothing to open.
                const builtIn = cell.column.id === "name" || cell.column.id === DOCUMENTS_COLUMN;
                const isSelected =
                  !builtIn && selected !== null && selected.entityId === row.original.id && selected.columnId === cell.column.id;
                return (
                  <td
                    key={cell.id}
                    className={cn(
                      TD,
                      i === 0 && "sticky left-0 z-[5] bg-background",
                      cell.column.id === DOCUMENTS_COLUMN && "h-auto max-w-96 whitespace-normal",
                      !builtIn && "cursor-pointer",
                      isSelected && "outline-2 -outline-offset-2 outline-primary",
                    )}
                    onClick={builtIn ? undefined : () => onSelect({ entityId: row.original.id, columnId: cell.column.id })}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                );
              })}
              <td className="border-b" />
            </tr>
          ))}
          <tr>
            <td className="sticky left-0 border-b bg-background px-2 py-1" colSpan={1}>
              {adding ? (
                <Input
                  value={rowName}
                  onChange={(e) => setRowName(e.target.value)}
                  onKeyDown={onRowKey}
                  onBlur={() => rowName.trim() === "" && setAdding(false)}
                  placeholder="Company, vendor or case — Enter to add"
                  autoFocus
                />
              ) : (
                <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={() => setAdding(true)}>
                  <Plus className="size-4" /> New
                </Button>
              )}
            </td>
            <td className="border-b" colSpan={columns.length + 2} />
          </tr>
        </tbody>
      </table>
    </div>
  );
}
