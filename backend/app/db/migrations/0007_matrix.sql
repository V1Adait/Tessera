-- The matrix: columns (typed questions), cells (one answer per row x column) and their evidence.
--
--   matrix_columns   one row per column: the identity and which version is current
--   column_versions  every version of a column's definition, never edited. A cell remembers which
--                    version it answered, so changing a column marks its old answers stale and
--                    the history stays readable.
--   cells            one per (entity, column): the status, the normalised value, and flags
--   cell_evidence    the verbatim quotes behind a cell, with where in the document they sit
--
-- Rules the database enforces itself (the application checks them too, this is the second defence):
--   * a cell belongs to an entity AND a column of the SAME workspace
--   * one cell per entity and column
--   * an 'answered' cell has a value
--   * a select column has at least two allowed values
-- "Every answered cell has evidence" spans two tables; the application enforces it (and a test
-- proves it), because a plain CHECK cannot look at another table.

CREATE TABLE matrix_columns (
    id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id     uuid        NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    -- column order in the grid; an identity column is handed out atomically
    position         bigint      GENERATED ALWAYS AS IDENTITY,
    current_version  integer     NOT NULL DEFAULT 1 CHECK (current_version >= 1),
    -- the current name, lower-cased and trimmed: unique within a workspace
    name_key         text        NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT matrix_columns_id_workspace_uq UNIQUE (id, workspace_id)
);
CREATE UNIQUE INDEX matrix_columns_name_uq ON matrix_columns (workspace_id, name_key);
CREATE INDEX matrix_columns_position_idx ON matrix_columns (workspace_id, position);

CREATE TABLE column_versions (
    column_id          uuid        NOT NULL REFERENCES matrix_columns (id) ON DELETE CASCADE,
    version            integer     NOT NULL CHECK (version >= 1),
    name               text        NOT NULL CHECK (char_length(btrim(name)) BETWEEN 1 AND 120),
    prompt             text        NOT NULL CHECK (char_length(btrim(prompt)) BETWEEN 1 AND 2000),
    type               text        NOT NULL
                       CHECK (type IN ('text', 'number', 'date', 'currency', 'select', 'yes_no', 'status')),
    allowed_values     text[]      NOT NULL DEFAULT '{}',
    -- when the question does not apply to a row, in words. NULL means it applies to every row.
    applicability      text        CHECK (applicability IS NULL OR char_length(applicability) <= 1000),
    evidence_required  boolean     NOT NULL DEFAULT true,
    -- columns whose answers this one is derived from (computed in code, not by the model)
    depends_on         uuid[]      NOT NULL DEFAULT '{}',
    created_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (column_id, version),
    CONSTRAINT column_versions_select_needs_values
        CHECK (type <> 'select' OR cardinality(allowed_values) >= 2)
);

CREATE TABLE cells (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    uuid        NOT NULL,
    entity_id       uuid        NOT NULL,
    column_id       uuid        NOT NULL,
    column_version  integer     NOT NULL,
    -- answered: found, with evidence.  not_applicable: the question does not apply to this row.
    -- not_found: it applies but the documents do not answer it.  conflicting: sources disagree.
    -- low_confidence: an answer exists but cannot be trusted (see status_detail).
    -- pending_or_error: not done yet, or it failed (see status_detail).
    status          text        NOT NULL
                    CHECK (status IN ('answered', 'not_applicable', 'not_found', 'conflicting',
                                      'low_confidence', 'pending_or_error')),
    status_detail   text,
    value           jsonb,          -- normalised: numbers, dates and currency are parsed in code
    display_value   text,           -- how the value reads in a grid
    as_of           date,
    derivation      jsonb,          -- set when the value was computed from other cells
    stale           boolean     NOT NULL DEFAULT false,
    pinned          boolean     NOT NULL DEFAULT false,   -- a person set it: never overwritten silently
    model           text,
    prompt_version  text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT cells_entity_fk FOREIGN KEY (entity_id, workspace_id)
        REFERENCES entities (id, workspace_id) ON DELETE CASCADE,
    CONSTRAINT cells_column_fk FOREIGN KEY (column_id, workspace_id)
        REFERENCES matrix_columns (id, workspace_id) ON DELETE CASCADE,
    CONSTRAINT cells_version_fk FOREIGN KEY (column_id, column_version)
        REFERENCES column_versions (column_id, version),
    CONSTRAINT cells_one_per_pair UNIQUE (entity_id, column_id),
    CONSTRAINT cells_answered_has_value CHECK (status <> 'answered' OR value IS NOT NULL)
);
CREATE INDEX cells_workspace_idx ON cells (workspace_id, column_id);

CREATE TABLE cell_evidence (
    id            uuid     PRIMARY KEY DEFAULT gen_random_uuid(),
    cell_id       uuid     NOT NULL REFERENCES cells (id) ON DELETE CASCADE,
    seq           integer  NOT NULL CHECK (seq >= 0),
    document_id   uuid     NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    -- the chunk it was found in; kept as NULL if the chunk is rebuilt (the quote stays valid)
    chunk_id      uuid     REFERENCES document_chunks (id) ON DELETE SET NULL,
    page_start    integer,
    page_end      integer,
    -- where the quote sits in the document's canonical text. exact_range says whether that is the
    -- quote itself (true) or the whole chunk it came from (false: a split table repeats its header
    -- in each piece, so the quote cannot be placed to the character)
    char_start    integer  NOT NULL CHECK (char_start >= 0),
    char_end      integer  NOT NULL,
    exact_range   boolean  NOT NULL,
    quote         text     NOT NULL CHECK (char_length(quote) BETWEEN 1 AND 2000),
    CONSTRAINT cell_evidence_range_ok CHECK (char_end >= char_start),
    CONSTRAINT cell_evidence_order_uq UNIQUE (cell_id, seq)
);
CREATE INDEX cell_evidence_document_idx ON cell_evidence (document_id);
