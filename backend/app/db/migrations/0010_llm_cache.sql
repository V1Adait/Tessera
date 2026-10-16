-- Replies of paid model calls, found again by a fingerprint of everything that decided them.
-- key        sha256 of (model, prompt version, sampling settings, the exact messages sent)
-- reply      the model's raw text; verification and parsing run again on every use
-- cost_usd   what the original call cost: what each later hit saves
CREATE TABLE llm_cache (
    key          text PRIMARY KEY,
    kind         text NOT NULL,
    model        text NOT NULL,
    reply        text NOT NULL,
    cost_usd     numeric(18,10),
    hits         integer NOT NULL DEFAULT 0,
    created_at   timestamptz NOT NULL DEFAULT now(),
    last_hit_at  timestamptz
);
