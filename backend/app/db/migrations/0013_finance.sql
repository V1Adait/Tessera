-- The finance layer: a figure's period, columns computed in code, and tie-out checks.
--
--   cells.period               the reporting period the figure covers ("FY2025", "Q2 2026"), as the
--                              document states it and code has normalised it. NULL when not stated.
--   cells.checks               the tie-out results shown on this cell (see backend/app/finance/checks.py):
--                              [{id, name, status: ties | does_not_tie | cannot_check, detail}]
--   column_versions.derivation how a COMPUTED column works: {"op": "ratio", "as": "percent"}. NULL for an
--                              ordinary column that is read from documents. The columns it reads are
--                              the version's depends_on, in order.
--
-- (cells.derivation already exists: for a computed cell it records the inputs it was computed from.)

ALTER TABLE cells
    ADD COLUMN period text CHECK (period IS NULL OR char_length(period) BETWEEN 4 AND 12),
    ADD COLUMN checks jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(checks) = 'array');

ALTER TABLE column_versions
    ADD COLUMN derivation jsonb CHECK (derivation IS NULL OR jsonb_typeof(derivation) = 'object');
