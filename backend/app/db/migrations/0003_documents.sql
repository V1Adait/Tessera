-- Documents and their parsed blocks: the hand-off to chunking.
--
--   documents        one row per uploaded file (the bytes live on disk under their SHA-256 hash)
--   document_blocks  what the parser read, in reading order: headings, paragraphs, list items,
--                    tables, figures. Each block keeps its verbatim text, where it came from,
--                    its character offsets in the document's canonical text, a confidence and
--                    flags. Chunking and retrieval read from here.
--
-- A document belongs to a workspace and may belong to an entity (a row of the matrix). It can be
-- uploaded before its entity exists and assigned later. The database itself guarantees that an
-- entity belongs to the same workspace as its documents.

-- the composite key that lets documents reference (entity, workspace) together
ALTER TABLE entities ADD CONSTRAINT entities_id_workspace_uq UNIQUE (id, workspace_id);

CREATE TABLE documents (
    id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id  uuid        NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    entity_id     uuid,
    -- a label for display only. It is NEVER used as a path.
    filename      text        NOT NULL CHECK (char_length(filename) BETWEEN 1 AND 255),
    kind          text        NOT NULL
                  CHECK (kind IN ('pdf', 'docx', 'xlsx', 'pptx', 'html', 'csv', 'txt', 'md')),
    sha256        text        NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes    bigint      NOT NULL CHECK (size_bytes > 0),
    -- uploaded: stored, not parsed yet.  parsing: a worker has it.
    -- parsed: read cleanly.  needs_review: read, but some blocks or pages are suspect.
    -- needs_ocr: scanned, no text could be read.  failed: could not be read (see status_detail).
    status        text        NOT NULL DEFAULT 'uploaded'
                  CHECK (status IN ('uploaded', 'parsing', 'parsed', 'needs_review', 'needs_ocr', 'failed')),
    status_detail text,
    parser        text,
    page_count    integer     CHECK (page_count >= 0),
    char_count    integer     CHECK (char_count >= 0),
    report        jsonb       NOT NULL DEFAULT '{}'::jsonb,   -- per page/sheet/slide reports + flags
    created_at    timestamptz NOT NULL DEFAULT now(),
    parsed_at     timestamptz,
    -- the same bytes uploaded twice to one workspace are one document
    CONSTRAINT documents_workspace_sha_uq UNIQUE (workspace_id, sha256),
    -- an entity must be in the document's own workspace; deleting the entity un-assigns the
    -- document (entity_id becomes NULL) and keeps the document and its workspace
    CONSTRAINT documents_entity_fk FOREIGN KEY (entity_id, workspace_id)
        REFERENCES entities (id, workspace_id) ON DELETE SET NULL (entity_id)
);
CREATE INDEX documents_workspace_idx ON documents (workspace_id, created_at DESC);
CREATE INDEX documents_entity_idx ON documents (entity_id) WHERE entity_id IS NOT NULL;
CREATE INDEX documents_status_idx ON documents (status) WHERE status IN ('uploaded', 'parsing');

CREATE TABLE document_blocks (
    id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   uuid        NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    seq           integer     NOT NULL CHECK (seq >= 0),         -- reading order
    type          text        NOT NULL
                  CHECK (type IN ('heading', 'paragraph', 'list_item', 'table', 'figure')),
    text          text        NOT NULL,                          -- verbatim; quotes are checked against it
    level         smallint    CHECK (level BETWEEN 0 AND 6),     -- heading level / list depth
    location      jsonb       NOT NULL,                          -- page, sheet + cell range, slide, section path
    data          jsonb,                                         -- tables: the structured cells; figures: labels
    confidence    real        NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    flags         text[]      NOT NULL DEFAULT '{}',
    char_start    integer     NOT NULL CHECK (char_start >= 0),  -- offsets into the canonical text
    char_end      integer     NOT NULL,
    CONSTRAINT document_blocks_offsets_ok CHECK (char_end >= char_start),
    CONSTRAINT document_blocks_order_uq UNIQUE (document_id, seq)
);
