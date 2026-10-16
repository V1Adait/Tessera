-- Matrix runs: one request to fill many cells.
--
-- A run is a batch with a scope (some rows, some columns), a cost cap, and live progress, so a
-- person can watch it, stop it, and see afterwards what it did and what it cost. The cells it fills
-- are ordinary cells; the run only keeps the books.
--
--   running      cells are being filled
--   completed    every planned cell was attempted
--   stopped      a person stopped it; the rest were not started
--   cost_cap     it reached its spending limit; the rest were not started
--   interrupted  the server stopped while it ran (shutdown or crash); unfinished cells are untouched
--   failed       the run itself broke (not a single cell: those count under `failed`)

CREATE TABLE matrix_runs (
    id                  uuid          PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id        uuid          NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    status              text          NOT NULL
                        CHECK (status IN ('running', 'completed', 'stopped', 'cost_cap', 'interrupted', 'failed')),
    scope               jsonb         NOT NULL,
    total               integer       NOT NULL CHECK (total >= 0),
    done                integer       NOT NULL DEFAULT 0 CHECK (done >= 0),      -- cells attempted
    failed              integer       NOT NULL DEFAULT 0 CHECK (failed >= 0),    -- of those, ended in error
    skipped             integer       NOT NULL DEFAULT 0 CHECK (skipped >= 0),   -- never started
    spent_usd           numeric(18, 10) NOT NULL DEFAULT 0 CHECK (spent_usd >= 0),
    unknown_cost_calls  integer       NOT NULL DEFAULT 0 CHECK (unknown_cost_calls >= 0),
    cap_usd             numeric(18, 10) NOT NULL CHECK (cap_usd >= 0),
    estimate_usd        numeric(18, 10) CHECK (estimate_usd >= 0),
    detail              text,
    created_at          timestamptz   NOT NULL DEFAULT now(),
    finished_at         timestamptz,
    CONSTRAINT matrix_runs_counts_ok CHECK (done + skipped <= total)
);
CREATE INDEX matrix_runs_workspace_idx ON matrix_runs (workspace_id, created_at DESC);
-- at most one running run per workspace: two at once would fight over the same cells and the same cap
CREATE UNIQUE INDEX matrix_runs_one_running_uq ON matrix_runs (workspace_id) WHERE status = 'running';
