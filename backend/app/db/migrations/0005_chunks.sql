-- Chunks: what gets searched, embedded and cited. Built from document_blocks after a parse.
--
--   document_sections  the PARENTS: a heading and what is under it (a huge section is cut into
--                      windows). This is what the model reads when one of its chunks is found.
--   document_chunks    the CHILDREN: small, one kind of content each. Two texts per chunk:
--                        text         the exact words. Quotes are checked against this only.
--                        search_text  a context line (document, section path, units) + text.
--                                     Only used to FIND chunks (full-text now, vectors later).
--
-- Both are derived data: deleting a document's chunks and rebuilding them from its blocks loses
-- nothing. Offsets are into the document's canonical text, like document_blocks.
-- Scope (workspace, entity) is NOT copied here: it comes from `documents` by join, so assigning
-- a document to another entity needs no chunk update.

CREATE TABLE document_sections (
    id           uuid     PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  uuid     NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal      integer  NOT NULL CHECK (ordinal >= 0),   -- position in the document
    path         text[]   NOT NULL DEFAULT '{}',           -- headings, outermost first
    char_start   integer  NOT NULL CHECK (char_start >= 0),
    char_end     integer  NOT NULL,
    page_start   integer,
    page_end     integer,
    token_count  integer  NOT NULL CHECK (token_count >= 0),
    CONSTRAINT document_sections_span_ok CHECK (char_end >= char_start),
    CONSTRAINT document_sections_order_uq UNIQUE (document_id, ordinal)
);

CREATE TABLE document_chunks (
    id            uuid     PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   uuid     NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    section_id    uuid     NOT NULL REFERENCES document_sections (id) ON DELETE CASCADE,
    ordinal       integer  NOT NULL CHECK (ordinal >= 0),
    kind          text     NOT NULL CHECK (kind IN ('prose', 'table', 'figure')),
    text          text     NOT NULL CHECK (text <> ''),
    search_text   text     NOT NULL,
    section_path  text[]   NOT NULL DEFAULT '{}',
    block_seqs    integer[] NOT NULL,                      -- document_blocks.seq it came from
    char_start    integer  NOT NULL CHECK (char_start >= 0),
    char_end      integer  NOT NULL,
    token_count   integer  NOT NULL CHECK (token_count > 0),
    page_start    integer,
    page_end      integer,
    sheet         text,
    slide         integer,
    cell_range    text,
    units         text,
    confidence    real     NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    flags         text[]   NOT NULL DEFAULT '{}',
    -- sha256 of search_text: the same value means the same thing would be embedded
    content_hash  text     NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    -- full-text index over the search text; the database keeps it in step with the text
    search_tsv    tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
    CONSTRAINT document_chunks_span_ok CHECK (char_end >= char_start),
    CONSTRAINT document_chunks_order_uq UNIQUE (document_id, ordinal)
);
CREATE INDEX document_chunks_section_idx ON document_chunks (section_id);
CREATE INDEX document_chunks_hash_idx ON document_chunks (content_hash);
CREATE INDEX document_chunks_fts_idx ON document_chunks USING gin (search_tsv);
