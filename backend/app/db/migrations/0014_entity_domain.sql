-- A company's website domain ("palantir.com"), used to show its logo.
--
-- Intake fills it when the filing's first page prints the website (and only then: the domain must
-- appear in the text, like every other claim a model makes). A person can also set or change it.
-- A logo is only ever looked up for a company that has one, and only when the person switched logos
-- on in the screen: a lookup tells an outside service which companies are being researched.
--
-- The pattern matches backend/app/intake/models.py: labels start and end with a letter or digit,
-- and the last label has at least two characters.

ALTER TABLE entities
    ADD COLUMN domain text
    CHECK (
        domain IS NULL OR (
            char_length(domain) <= 100
            AND domain ~ '^([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z0-9][a-z0-9-]*[a-z0-9]$'
        )
    );
