-- The chat agent: tools it may call, and the actions that wait for a person's yes.
--
--   agent_actions   A tool call the model asked for that costs money or cannot be undone. The tool is
--                   NOT run: it is stored here, validated, with a plain-words summary, and a person
--                   confirms or cancels it. Confirming runs the stored call in code; the model is not
--                   asked again, so what was shown is exactly what runs.
--   tool_calls      An audit line for every tool call the model made: what, with what, and how it ended.
--
-- One trace kind joins the list: 'agent' (one model call inside an agent turn: a turn can make several).

ALTER TABLE traces DROP CONSTRAINT traces_kind_check;
ALTER TABLE traces
    ADD CONSTRAINT traces_kind_check
    CHECK (kind IN ('chat', 'summary', 'classify', 'extract', 'plan', 'embed', 'agent'));

CREATE TABLE agent_actions (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id uuid        NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    workspace_id    uuid        NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
    request_id      uuid        NOT NULL,
    tool            text        NOT NULL,
    args            jsonb       NOT NULL,
    risk            text        NOT NULL CHECK (risk IN ('costly', 'destructive')),
    summary         text        NOT NULL,
    status          text        NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'confirmed', 'cancelled', 'failed')),
    result          text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    resolved_at     timestamptz
);

CREATE INDEX agent_actions_conversation_idx ON agent_actions (conversation_id, created_at DESC);

CREATE TABLE tool_calls (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id uuid        NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    request_id      uuid        NOT NULL,
    step            int         NOT NULL,
    tool            text        NOT NULL,
    args            jsonb       NOT NULL,
    outcome         text        NOT NULL
                    CHECK (outcome IN ('ok', 'error', 'invalid', 'unknown_tool', 'needs_confirmation')),
    summary         text        NOT NULL,
    latency_ms      int         NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX tool_calls_conversation_idx ON tool_calls (conversation_id, created_at);
