-- Query handling makes model calls outside a chat turn: classify the question, pull out its
-- metrics, plan an open question. Each one is traced like any other call, so the allowed
-- trace kinds grow.
ALTER TABLE traces DROP CONSTRAINT traces_kind_check;
ALTER TABLE traces
    ADD CONSTRAINT traces_kind_check
    CHECK (kind IN ('chat', 'summary', 'classify', 'extract', 'plan'));
