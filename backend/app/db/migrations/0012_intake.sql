-- Intake: what happens to a file after it is parsed.
--
-- A model reads the first page and says which company the file is about. Code then finds that
-- company's row (or makes it) and attaches the file. A person who attaches a file to a row
-- themselves skips all of this.
--
--   entities.ticker / aliases   how a company is recognised the next time: "Palantir" and
--                               "Palantir Technologies" and PLTR are one row, not three
--   intake_results              what intake decided for each file, and why, so the screen can say
--                               "created row Palantir" or ask "which row is this?"

ALTER TABLE entities
    ADD COLUMN ticker  text CHECK (ticker IS NULL OR ticker ~ '^[A-Z0-9][A-Z0-9.\-]{0,9}$'),
    ADD COLUMN aliases text[] NOT NULL DEFAULT '{}';

CREATE TABLE intake_results (
    document_id uuid        PRIMARY KEY REFERENCES documents (id) ON DELETE CASCADE,
    outcome     text        NOT NULL
                CHECK (outcome IN ('assigned', 'created', 'needs_review', 'skipped')),
    entity_id   uuid        REFERENCES entities (id) ON DELETE SET NULL,
    detected    jsonb       NOT NULL DEFAULT '{}',
    detail      text,
    candidates  text[]      NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- Intake's model call is traced like every other paid call.
ALTER TABLE traces DROP CONSTRAINT traces_kind_check;
ALTER TABLE traces
    ADD CONSTRAINT traces_kind_check
    CHECK (kind IN ('chat', 'summary', 'classify', 'extract', 'plan', 'embed', 'agent', 'moderation', 'intake'));
