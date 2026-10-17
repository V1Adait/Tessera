import { Columns3, FileDown, Image as ImageIcon, Play } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input, Select } from "@/components/ui/input";
import CellDrawer from "./CellDrawer";
import ColumnEditor from "./ColumnEditor";
import DeleteColumnDialog from "./DeleteColumnDialog";
import EntityDialog from "./EntityDialog";
import { pdfFileName, saveBlob } from "./download";
import { readLogosSetting, writeLogosSetting } from "./logoModel";
import MatrixGrid, { type Selected } from "./MatrixGrid";
import { isWorking, listDocuments, type WorkspaceDocument } from "./documentsApi";
import { unplacedDocuments } from "./intakeModel";
import RunBar from "./RunBar";
import {
  createColumn,
  createEntity,
  createWorkspace,
  deleteColumn,
  exportPdf,
  extractCell,
  listCells,
  listColumns,
  listEntities,
  listRuns,
  listWorkspaces,
  updateColumn,
  updateEntity,
} from "./matrixApi";
import { cellKey, columnImpact, indexCells } from "./matrixModel";
import type { Cell, Column, ColumnDraft, Entity, Workspace } from "./matrixTypes";
import { useRunner } from "./useRunner";

const STORAGE_KEY = "tessera.workspaceId";

// localStorage can throw (private windows, blocked storage), so every use is wrapped.
function readStoredId(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function writeStoredId(id: string): void {
  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch {
    // not remembering the workspace is acceptable
  }
}

const CENTER = "flex flex-1 items-center justify-center p-6";

// The documents column is a convenience: if the list cannot be read, the rest of the grid still loads.
function filesOrNone(workspaceId: string): Promise<WorkspaceDocument[]> {
  return listDocuments(workspaceId).catch(() => []);
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong.";
}

interface MatrixProps {
  onContext: (workspaceId: string | null, entities: Entity[]) => void; // tells the chat where uploads go
  refreshSignal: number; // goes up when the agent changed the matrix: reload everything
}

export default function Matrix({ onContext, refreshSignal }: MatrixProps) {
  const [workspaces, setWorkspaces] = useState<Workspace[] | null>(null);
  const [workspaceId, setWorkspaceId] = useState<string | null>(null);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [columns, setColumns] = useState<Column[]>([]);
  const [cells, setCells] = useState<Cell[]>([]);
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const [selected, setSelected] = useState<Selected | null>(null);
  const [drawerBusy, setDrawerBusy] = useState(false);
  const [drawerError, setDrawerError] = useState<string | null>(null);

  const [editing, setEditing] = useState<{ column: Column | null } | null>(null);
  const [deleting, setDeleting] = useState<Column | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [editorBusy, setEditorBusy] = useState(false);
  const [editorError, setEditorError] = useState<string | null>(null);

  const [newWorkspace, setNewWorkspace] = useState("");

  // Company logos are looked up on an outside service, so they are OFF until the person turns them on.
  const [showLogos, setShowLogos] = useState<boolean>(() => readLogosSetting());
  const [exporting, setExporting] = useState(false);
  const [editingEntity, setEditingEntity] = useState<Entity | null>(null);
  const [entityBusy, setEntityBusy] = useState(false);
  const [entityError, setEntityError] = useState<string | null>(null);

  // --- loading -------------------------------------------------------------------------------

  useEffect(() => {
    listWorkspaces()
      .then((list) => {
        setWorkspaces(list);
        const remembered = readStoredId();
        const pick = list.find((w) => w.id === remembered) ?? list[0];
        if (pick) setWorkspaceId(pick.id);
      })
      .catch((err: unknown) => {
        setWorkspaces([]);
        setProblem(`Could not reach the server: ${messageOf(err)}`);
      });
  }, []);

  useEffect(() => {
    if (workspaceId === null) return;
    let cancelled = false;
    setLoading(true);
    setSelected(null);
    Promise.all([
      listEntities(workspaceId),
      listColumns(workspaceId),
      listCells(workspaceId),
      filesOrNone(workspaceId),
    ])
      .then(([e, c, k, d]) => {
        if (cancelled) return;
        setEntities(e);
        setColumns(c);
        setCells(k);
        setDocuments(d);
        setProblem(null);
      })
      .catch((err: unknown) => {
        if (!cancelled) setProblem(messageOf(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [workspaceId]);

  const reloadCells = useCallback(() => {
    if (workspaceId === null) return;
    listCells(workspaceId)
      .then(setCells)
      .catch(() => undefined); // the next poll tries again
  }, [workspaceId]);

  // The agent changed something (a row, a column, or it started a fill). Reload the grid now, and
  // keep reloading its cells every 2 s while a fill it started is still going.
  useEffect(() => {
    if (refreshSignal === 0 || workspaceId === null) return;
    let stopped = false;
    void (async () => {
      for (let tick = 0; tick < 300 && !stopped; tick++) {
        try {
          const [e, c, k, d] = await Promise.all([
            listEntities(workspaceId),
            listColumns(workspaceId),
            listCells(workspaceId),
            filesOrNone(workspaceId),
          ]);
          if (stopped) return;
          setEntities(e);
          setColumns(c);
          setCells(k);
          setDocuments(d);
          const [latest] = await listRuns(workspaceId);
          if (latest === undefined || latest.status !== "running") return;
        } catch {
          return; // the grid still has what it had; a manual reload fixes the rest
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    })();
    return () => {
      stopped = true;
    };
  }, [refreshSignal, workspaceId]);

  // While a file is still being read, look again every 2 s so its status chip updates.
  const readingFiles = documents.some((d) => isWorking(d.status));
  useEffect(() => {
    if (!readingFiles || workspaceId === null) return;
    const timer = setInterval(() => {
      listDocuments(workspaceId)
        .then(setDocuments)
        .catch(() => undefined); // keep what is shown; the next tick tries again
    }, 2000);
    return () => clearInterval(timer);
  }, [readingFiles, workspaceId]);

  const runner = useRunner(workspaceId, reloadCells);
  const runBusy = runner.phase.kind === "running" || runner.phase.kind === "working" || runner.phase.kind === "confirm";

  useEffect(() => {
    onContext(workspaceId, entities);
  }, [onContext, workspaceId, entities]);

  const cellIndex = useMemo(() => indexCells(cells), [cells]);

  // --- actions -------------------------------------------------------------------------------

  async function addWorkspace(): Promise<void> {
    const name = newWorkspace.trim();
    if (name === "") return;
    try {
      const created = await createWorkspace(name);
      setWorkspaces((prev) => [...(prev ?? []), created]);
      setNewWorkspace("");
      pickWorkspace(created.id);
    } catch (err) {
      setProblem(messageOf(err));
    }
  }

  function pickWorkspace(id: string): void {
    setWorkspaceId(id);
    writeStoredId(id);
  }

  async function addRow(name: string): Promise<void> {
    if (workspaceId === null) return;
    try {
      const created = await createEntity(workspaceId, name);
      setEntities((prev) => [...prev, created]);
      setProblem(null);
    } catch (err) {
      setProblem(messageOf(err));
    }
  }

  async function downloadPdf(): Promise<void> {
    if (workspaceId === null) return;
    setExporting(true);
    try {
      const blob = await exportPdf(workspaceId);
      const name = workspaces?.find((w) => w.id === workspaceId)?.name ?? "matrix";
      saveBlob(blob, pdfFileName(name, new Date()));
      setProblem(null);
    } catch (err) {
      setProblem(`Could not export the PDF: ${messageOf(err)}`);
    } finally {
      setExporting(false);
    }
  }

  function toggleLogos(): void {
    const next = !showLogos;
    setShowLogos(next);
    writeLogosSetting(next);
  }

  async function saveEntity(details: { domain: string | null; ticker: string | null }): Promise<void> {
    if (workspaceId === null || editingEntity === null) return;
    setEntityBusy(true);
    setEntityError(null);
    try {
      const updated = await updateEntity(workspaceId, editingEntity.id, details);
      setEntities((prev) => prev.map((e) => (e.id === updated.id ? updated : e)));
      setEditingEntity(null);
    } catch (err) {
      setEntityError(messageOf(err)); // the server's reason, e.g. "the ticker PLTR already belongs to ..."
    } finally {
      setEntityBusy(false);
    }
  }

  function askToRemove(column: Column): void {
    setDeleteError(null);
    setDeleting(column);
  }

  async function removeColumn(): Promise<void> {
    if (workspaceId === null || deleting === null) return;
    setDeleteBusy(true);
    setDeleteError(null);
    try {
      await deleteColumn(workspaceId, deleting.id);
      const gone = deleting.id;
      setColumns((prev) => prev.filter((c) => c.id !== gone));
      setCells((prev) => prev.filter((c) => c.column_id !== gone));
      setSelected((prev) => (prev?.columnId === gone ? null : prev));
      setDeleting(null);
      reloadCells(); // tie-outs that named it are recomputed on the server
    } catch (err) {
      setDeleteError(messageOf(err)); // the server's own reason, e.g. "Cost ratio is computed from it"
    } finally {
      setDeleteBusy(false);
    }
  }

  async function saveColumn(draft: ColumnDraft): Promise<void> {
    if (workspaceId === null || editing === null) return;
    setEditorBusy(true);
    setEditorError(null);
    try {
      const existing = editing.column;
      if (existing === null) {
        const created = await createColumn(workspaceId, draft);
        setColumns((prev) => [...prev, created]);
      } else {
        // An emptied rule must be CLEARED explicitly; leaving the field out means "keep it".
        const { applicability, ...rest } = draft;
        const patch =
          applicability === null
            ? { ...rest, clear_applicability: existing.applicability !== null }
            : { ...rest, applicability };
        await updateColumn(workspaceId, existing.id, patch);
        // A change can mark answers stale, so re-read both.
        const [c, k] = await Promise.all([listColumns(workspaceId), listCells(workspaceId)]);
        setColumns(c);
        setCells(k);
      }
      setEditing(null);
    } catch (err) {
      setEditorError(messageOf(err));
    } finally {
      setEditorBusy(false);
    }
  }

  async function fillSelected(refresh: boolean): Promise<void> {
    if (workspaceId === null || selected === null) return;
    setDrawerBusy(true);
    setDrawerError(null);
    try {
      const cell = await extractCell(workspaceId, selected.entityId, selected.columnId, refresh);
      setCells((prev) => [
        ...prev.filter((c) => cellKey(c.entity_id, c.column_id) !== cellKey(cell.entity_id, cell.column_id)),
        cell,
      ]);
    } catch (err) {
      setDrawerError(messageOf(err));
    } finally {
      setDrawerBusy(false);
    }
  }

  // --- render --------------------------------------------------------------------------------

  if (workspaces === null) return <section className={CENTER}>Loading…</section>;

  if (workspaces.length === 0) {
    return (
      <section className={CENTER}>
        <div className="flex max-w-sm flex-col items-center gap-3 text-center">
          <h2 className="text-xl font-semibold">Create your first workspace</h2>
          <p className="text-sm text-muted-foreground">
            A workspace holds one matrix: its rows, its columns and their documents.
          </p>
          <div className="flex w-full gap-2">
            <Input
              value={newWorkspace}
              onChange={(e) => setNewWorkspace(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void addWorkspace()}
              placeholder="Workspace name"
              autoFocus
            />
            <Button variant="default" onClick={() => void addWorkspace()} disabled={newWorkspace.trim() === ""}>
              Create
            </Button>
          </div>
          {problem && <div className="text-sm text-destructive">{problem}</div>}
        </div>
      </section>
    );
  }

  const selectedEntity = selected && entities.find((e) => e.id === selected.entityId);
  const selectedColumn = selected && columns.find((c) => c.id === selected.columnId);

  function openEditor(column: Column | null): void {
    setEditorError(null);
    setEditing({ column });
  }

  return (
    <section className="relative flex h-full min-w-0 flex-1 flex-col overflow-hidden bg-background text-foreground">
      <div className="flex items-center gap-3 px-5 pb-2 pt-4">
        <Select
          className="h-9 w-auto border-0 bg-transparent px-1 text-lg font-semibold hover:bg-muted"
          value={workspaceId ?? ""}
          onChange={(e) => pickWorkspace(e.target.value)}
          aria-label="Workspace"
        >
          {workspaces.map((w) => (
            <option key={w.id} value={w.id}>
              {w.name}
            </option>
          ))}
        </Select>
        <Input
          className="ml-auto w-44"
          value={newWorkspace}
          onChange={(e) => setNewWorkspace(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void addWorkspace()}
          placeholder="New workspace…"
        />
      </div>

      <div className="flex items-center gap-1 border-b px-4 pb-2">
        <Button variant="ghost" size="sm" onClick={() => openEditor(null)}>
          <Columns3 className="size-4" /> Add column
        </Button>
        <Button
          variant={showLogos ? "outline" : "ghost"}
          size="sm"
          aria-pressed={showLogos}
          onClick={toggleLogos}
          title="Show company logos. Each logo is looked up from the company's website on an outside icon service, which then knows which companies you are researching. Off by default."
        >
          <ImageIcon className="size-4" /> Logos
        </Button>
        <Button
          variant="ghost"
          size="sm"
          disabled={exporting || entities.length === 0 || columns.length === 0}
          onClick={() => void downloadPdf()}
          title="Download the table as a PDF: a summary, then the quote, file and page behind every figure"
        >
          <FileDown className="size-4" /> {exporting ? "Building…" : "Export PDF"}
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="ml-auto"
          disabled={runBusy || entities.length === 0 || columns.length === 0}
          onClick={() => void runner.begin({})}
          title="Fill every cell that has no answer yet, is out of date, or ended in an error"
        >
          <Play className="size-3.5" /> Fill empty cells
        </Button>
      </div>

      {unplacedDocuments(documents).length > 0 && (
        <div className="bg-[#fdecc8] px-5 py-2 text-sm text-[#402c1b]">
          {unplacedDocuments(documents).length} file(s) are not in a row yet. Choose their row in the chat, or ask the
          agent to assign them.
        </div>
      )}

      {problem && <div className="bg-[#ffe2dd] px-5 py-2 text-sm text-[#5d1715]">{problem}</div>}

      <MatrixGrid
        entities={entities}
        columns={columns}
        cells={cellIndex}
        documents={documents}
        selected={selected}
        busy={runBusy}
        onSelect={(s) => {
          setSelected(s);
          setDrawerError(null);
        }}
        onEditColumn={openEditor}
        onAddColumn={() => openEditor(null)}
        onFillColumn={(c) => void runner.begin({ column_ids: [c.id] })}
        onDeleteColumn={askToRemove}
        onEditEntity={(e) => {
          setEntityError(null);
          setEditingEntity(e);
        }}
        showLogos={showLogos}
        onFillRow={(e) => void runner.begin({ entity_ids: [e.id] })}
        onAddRow={addRow}
      />
      {loading && <div className="px-5 py-3 text-sm text-muted-foreground">Loading…</div>}
      {!loading && entities.length === 0 && columns.length === 0 && (
        <div className="px-5 py-3 text-sm text-muted-foreground">
          Add a row for each company, and a column for each question you want answered.
        </div>
      )}

      <RunBar
        phase={runner.phase}
        onConfirm={() => void runner.confirm()}
        onStop={() => void runner.stop()}
        onDismiss={runner.dismiss}
      />

      <div className="border-t px-5 py-1.5 text-xs tracking-wide text-muted-foreground">
        COUNT <strong className="text-sm text-foreground">{entities.length}</strong> rows ·{" "}
        <strong className="text-sm text-foreground">{columns.length}</strong> columns
      </div>

      {selectedEntity && selectedColumn && (
        <CellDrawer
          entity={selectedEntity}
          column={selectedColumn}
          cell={cellIndex.get(cellKey(selectedEntity.id, selectedColumn.id))}
          busy={drawerBusy}
          error={drawerError}
          onFill={(refresh) => void fillSelected(refresh)}
          onClose={() => setSelected(null)}
        />
      )}

      {editing && (
        <ColumnEditor
          column={editing.column}
          busy={editorBusy}
          error={editorError}
          onSave={(draft) => void saveColumn(draft)}
          onClose={() => setEditing(null)}
          onDelete={
            editing.column
              ? () => {
                  const column = editing.column;
                  setEditing(null);
                  if (column) askToRemove(column);
                }
              : undefined
          }
        />
      )}

      {editingEntity && (
        <EntityDialog
          entity={editingEntity}
          busy={entityBusy}
          error={entityError}
          onSave={(details) => void saveEntity(details)}
          onClose={() => setEditingEntity(null)}
        />
      )}

      {deleting && (
        <DeleteColumnDialog
          column={deleting}
          impact={columnImpact(cells, deleting.id)}
          busy={deleteBusy}
          error={deleteError}
          blockedReason={runBusy ? "A fill is going. Stop it, or wait, before removing a column." : null}
          onConfirm={() => void removeColumn()}
          onClose={() => setDeleting(null)}
        />
      )}
    </section>
  );
}
