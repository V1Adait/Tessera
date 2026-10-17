import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import type { Entity } from "./matrixTypes";

interface Props {
  entity: Entity;
  busy: boolean;
  error: string | null; // the server's reason, e.g. a ticker another row already has
  onSave: (details: { domain: string | null; ticker: string | null }) => void;
  onClose: () => void;
}

/** A company's website (for its logo) and ticker (how new files are matched to it). A blank field
 * clears it. The server checks both: this form only saves a round trip. */
export default function EntityDialog({ entity, busy, error, onSave, onClose }: Props) {
  const [domain, setDomain] = useState(entity.domain ?? "");
  const [ticker, setTicker] = useState(entity.ticker ?? "");

  function submit(e: FormEvent): void {
    e.preventDefault();
    onSave({ domain: domain.trim() || null, ticker: ticker.trim() || null });
  }

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <form className="flex flex-col gap-3" onSubmit={submit}>
          <DialogTitle className="text-lg font-semibold">{entity.name}</DialogTitle>
          <DialogDescription className="text-sm text-muted-foreground">
            The website is used for the company&rsquo;s logo (when logos are switched on). The ticker helps match new
            files to this company.
          </DialogDescription>

          <label className="flex flex-col gap-1 text-sm">
            <span>Website</span>
            <Input value={domain} onChange={(e) => setDomain(e.target.value)} placeholder="palantir.com" autoFocus />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span>Ticker</span>
            <Input value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="PLTR" />
          </label>

          {error && <div className="rounded-md bg-[#ffe2dd] px-3 py-2 text-sm text-[#5d1715]">{error}</div>}

          <div className="flex gap-2 pt-1">
            <Button type="submit" variant="default" disabled={busy}>
              {busy ? "Saving…" : "Save"}
            </Button>
            <Button onClick={onClose}>Cancel</Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
