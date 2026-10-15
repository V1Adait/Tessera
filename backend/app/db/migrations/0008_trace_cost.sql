-- Cost tracing: every call that costs money records what it cost and where that number came from.
--
--   cost_source  provider: the provider reported it on the response
--                computed: tokens x the model's published price (an estimate)
--                unknown:  a paid call whose cost could not be worked out. Counted separately in
--                          the cost report, never treated as free.
--                NULL:     no paid call was made (a request refused before the model, an error
--                          before any billing)
--
-- cost_usd also gets room for the cheapest calls. numeric(12,6) rounds $0.00000086 (one embedding
-- call) up to $0.000001, and anything under half a millionth to zero, so a sum over thousands of
-- calls could be wrong by a large factor. Ten decimal places keeps them exact.

ALTER TABLE traces ALTER COLUMN cost_usd TYPE numeric(18, 10);

ALTER TABLE traces
    ADD COLUMN cost_source text
        CHECK (cost_source IN ('provider', 'computed', 'unknown'));

-- a cost with no source, or a source with no cost (except 'unknown'), would be a bookkeeping bug
ALTER TABLE traces
    ADD CONSTRAINT traces_cost_has_source
        CHECK ((cost_usd IS NULL) = (cost_source IS NULL OR cost_source = 'unknown'));

CREATE INDEX traces_kind_created_idx ON traces (kind, created_at DESC);
