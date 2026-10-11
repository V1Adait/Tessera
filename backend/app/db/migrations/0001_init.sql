-- Tessera schema, version 1: conversations, messages, summaries, trim events, traces.
--
-- Rules of thumb used here:
--   * The database enforces what must always be true (CHECK, NOT NULL, FOREIGN KEY),
--     so a bug in the Python code cannot silently corrupt the data.
--   * We never delete history. A summary only moves a pointer (summaries.summarised_through_
--     message_id); the full conversation stays in `messages`.
--   * Times are timestamptz (timezone-aware).

CREATE TABLE conversations (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    title       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per chat message. The server owns the history; the system prompt is NOT stored
-- here (it lives in code, versioned by prompt_version), so only user/assistant roles exist.
CREATE TABLE messages (
    id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  uuid        NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    -- Insertion order. An identity column is handed out atomically, so concurrent inserts
    -- can never get the same number (unlike "SELECT max(position) + 1").
    seq              bigint      GENERATED ALWAYS AS IDENTITY,
    role             text        NOT NULL CHECK (role IN ('user', 'assistant')),
    content          text        NOT NULL CHECK (octet_length(content) <= 1000000),
    -- complete: finished. streaming: still being written. stopped: user pressed Stop.
    -- error: the model call failed (content holds whatever arrived first).
    status           text        NOT NULL DEFAULT 'complete'
                                 CHECK (status IN ('streaming', 'complete', 'stopped', 'error')),
    tokens           integer     CHECK (tokens >= 0),
    -- The chat call this message belongs to; joins to traces.request_id.
    request_id       uuid,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX messages_conversation_seq_idx ON messages (conversation_id, seq);

-- One row per summary round. The newest row (highest seq) is the active summary; older rows
-- are kept so a bad summary can be inspected or regenerated.
CREATE TABLE summaries (
    id                           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    seq                          bigint      GENERATED ALWAYS AS IDENTITY,
    conversation_id              uuid        NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    -- Summaries stack: each one is built from the previous summary plus newer messages.
    previous_summary_id          uuid        REFERENCES summaries (id) ON DELETE SET NULL,
    -- THE POINTER: the last message this summary covers. Messages after it are still sent
    -- word for word. No ON DELETE action: a summarised message cannot be deleted on its own
    -- (a conversation delete still works, because the summary is deleted in the same statement).
    summarised_through_message_id uuid       NOT NULL REFERENCES messages (id),
    content                      text        NOT NULL CHECK (octet_length(content) <= 100000),
    tokens                       integer     CHECK (tokens >= 0),
    prompt_version               text        NOT NULL,
    model                        text,
    -- The LLM call that wrote this summary; joins to traces.request_id.
    request_id                   uuid,
    created_at                   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX summaries_conversation_seq_idx ON summaries (conversation_id, seq DESC);

-- A moment when the emergency brake (trim_history) had to drop messages from what we SEND.
-- Normally rare: a frequent row here means summaries run too late or fail.
CREATE TABLE trim_events (
    id                          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id             uuid        NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    request_id                  uuid        NOT NULL,
    reason                      text        NOT NULL CHECK (reason IN ('budget')),
    messages_dropped            integer     NOT NULL CHECK (messages_dropped > 0),
    tokens_dropped              integer     CHECK (tokens_dropped >= 0),
    budget_tokens               integer     NOT NULL CHECK (budget_tokens >= 0),
    dropped_through_message_id  uuid        REFERENCES messages (id) ON DELETE SET NULL,
    created_at                  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX trim_events_conversation_idx ON trim_events (conversation_id, created_at DESC);

-- One row per request we handled, successful or not (the "trace every call" rule).
-- Kept even if its conversation is deleted (audit and cost history), hence SET NULL.
CREATE TABLE traces (
    request_id       uuid          PRIMARY KEY,
    conversation_id  uuid          REFERENCES conversations (id) ON DELETE SET NULL,
    kind             text          NOT NULL CHECK (kind IN ('chat', 'summary')),
    provider         text,
    model            text,
    prompt_version   text,
    tokens_in        integer       CHECK (tokens_in >= 0),
    tokens_out       integer       CHECK (tokens_out >= 0),
    cost_usd         numeric(12, 6) CHECK (cost_usd >= 0),
    latency_ms       integer       CHECK (latency_ms >= 0),
    first_token_ms   integer       CHECK (first_token_ms >= 0),
    -- ok: finished. error: the call failed. rejected: refused before calling the model
    -- (guardrails, context too large). cancelled: the client went away or pressed Stop.
    status           text          NOT NULL CHECK (status IN ('ok', 'error', 'rejected', 'cancelled')),
    error_code       text,
    created_at       timestamptz   NOT NULL DEFAULT now()
);
CREATE INDEX traces_created_idx ON traces (created_at DESC);
CREATE INDEX traces_conversation_idx ON traces (conversation_id, created_at);
