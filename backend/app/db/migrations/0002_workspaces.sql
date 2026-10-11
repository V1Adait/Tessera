-- Workspaces and entities (the "rows" of the matrix).
--
--   workspace  a project, for example "Payments coverage"
--   entity     one row of the matrix: a company, vendor or case. It will own documents.
--   conversations now belong to a workspace, so the chat knows which documents it may use.
--
-- A single-user app has no workspace picker yet, so this migration also creates a default
-- workspace and moves every existing conversation into it.
-- (The domain "profile" and its version are added in their own migration, with personalisation.)

CREATE TABLE workspaces (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text        NOT NULL CHECK (char_length(btrim(name)) BETWEEN 1 AND 120),
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE entities (
    id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id  uuid        NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    name          text        NOT NULL CHECK (char_length(btrim(name)) BETWEEN 1 AND 200),
    -- Row order in the matrix. An identity column is handed out atomically, so two inserts
    -- at the same moment can never share a position.
    position      bigint      GENERATED ALWAYS AS IDENTITY,
    created_at    timestamptz NOT NULL DEFAULT now()
);
-- One entity per name within a workspace, ignoring case ("Visa" and "visa" are the same row).
CREATE UNIQUE INDEX entities_workspace_name_uq ON entities (workspace_id, lower(btrim(name)));
CREATE INDEX entities_workspace_position_idx ON entities (workspace_id, position);

ALTER TABLE conversations
    ADD COLUMN workspace_id uuid REFERENCES workspaces (id) ON DELETE CASCADE;

INSERT INTO workspaces (name) SELECT 'Default' WHERE NOT EXISTS (SELECT 1 FROM workspaces);

UPDATE conversations
SET workspace_id = (SELECT id FROM workspaces ORDER BY created_at, id LIMIT 1)
WHERE workspace_id IS NULL;

ALTER TABLE conversations ALTER COLUMN workspace_id SET NOT NULL;
CREATE INDEX conversations_workspace_idx ON conversations (workspace_id, updated_at DESC);
