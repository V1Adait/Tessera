-- Embeddings. The vectors themselves live in Qdrant; Postgres stays the source of truth for
-- chunks and now also records, per document, whether its chunks are in the vector index.
--
--   index_status  pending: chunks changed (or are new), not indexed yet
--                 indexing: a worker is embedding them right now
--                 indexed: every chunk of the document is in the vector index
--                 failed: indexing did not finish (see index_detail); full-text search still works
--   indexed_model which embedding model produced the vectors, so a model change is visible
--
-- Embedding calls are traced like every other model call, so 'embed' joins the trace kinds.

ALTER TABLE documents
    ADD COLUMN index_status  text NOT NULL DEFAULT 'pending'
        CHECK (index_status IN ('pending', 'indexing', 'indexed', 'failed')),
    ADD COLUMN index_detail  text,
    ADD COLUMN indexed_model text,
    ADD COLUMN indexed_at    timestamptz;

CREATE INDEX documents_index_idx ON documents (index_status)
    WHERE index_status IN ('pending', 'indexing');

ALTER TABLE traces DROP CONSTRAINT traces_kind_check;
ALTER TABLE traces
    ADD CONSTRAINT traces_kind_check
    CHECK (kind IN ('chat', 'summary', 'classify', 'extract', 'plan', 'embed'));
