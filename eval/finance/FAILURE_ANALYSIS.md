# Finance eval: failure analysis

Written from [REPORT.md](REPORT.md), the output of `eval/finance/run_eval.py` on one 10-K each for Apple, Microsoft, NVIDIA, Palantir, Salesforce and Walmart. Gold values are the XBRL numbers each company filed. Six filings is a small sample: the patterns below matter more than the percentages. Statements marked *hypothesis* were not verified against the filing.

## The headline, and what it hides

| | result |
|---|---|
| Extracted values scored | 50 |
| Exactly correct | 36 |
| Within 0.5% but not exact ("rounded") | 3 |
| Wrong value | 3 |
| Missed (the cell said it could not answer) | 8 |
| Quotes that failed the check against the document | 0 of 64 |
| Answered cells without a quote | 0 of 51 |

The failures split into **missing an answer** (8 of 14) and **answering a slightly different question** (6 of 14). There was **no case of an invented number**: every answered cell had a quote that exists in the document, and the three wrong values are real figures from the filing that are the wrong line for the question.

## 1. Cells that refused to answer (8 missed, plus the computed cells that depend on them)

**Microsoft: four extracted cells and two computed ones.** Revenue, gross profit, cash and interest expense all came back `low_confidence` with the reason that the quoted passage was flagged as unreliable when the document was read (`table_total_mismatch`). The parser's check that a table's rows add up to its printed total fired on Microsoft's tables, and the extractor refused to build on them. This is the intended safe behaviour (refuse, say why), but on this filing it cost six of eleven scored cells. *Hypothesis:* the check is too strict for how this filing's tables are laid out. Not yet investigated.

**NVIDIA and Palantir: cost of revenue and operating income.** The model said the documents do not state the figure, for example that NVIDIA gives revenue and a gross margin but "never states the cost of revenue amount". The XBRL values exist, so the number is in the filing, but not in the passages retrieved. *Hypothesis:* a retrieval or table-reading gap, not a model-knowledge gap. Palantir's cost of revenue was refused for a different reason: the table's unit (such as "in millions") could not be found, so the scale was unknown and the cell was not filled.

**Salesforce, Apple:** no missed cells.

## 2. Wrong values (3), all a different line, not a misread

| company | cell | answered | gold | quoted line |
|---|---|---|---|---|
| Palantir | Net income | 1,634,644 | 1,625,033 | "Net income $1,634,644" |
| Walmart | Net income | 22,270 | 21,893 | "Consolidated net income $22,270" |
| Walmart | Interest expense | 2,431 | 2,318 | "Interest, net 2,431" |

The two net income answers are consolidated net income, while the gold tag is, *hypothesis*, net income attributable to the company (the two differ by non-controlling interests). The interest answer is net of interest income, while the gold is gross interest expense. In each case the question the column asks ("net income", "interest expense") is ambiguous, and the model read it one way and the gold the other. The pack prompts should say "attributable to the company" and "gross, before interest income". Palantir's net margin is wrong for the same reason (it was computed from the wrong-line net income); the computation itself was right.

## 3. Rounded values (3)

Salesforce operating income ($8.3 billion against 8,331), Microsoft operating cash flow ($182.9 billion against 182,935) and Walmart cash ($10.7 billion against 10,727). The model quoted a rounded figure from the prose instead of the exact figure in the table. The quote is real and the figure is within 0.5%, but it is less precise than the filing. The extractor should prefer the table value.

## 4. Periods

43 of 44 answered extracted cells carry the filing's fiscal year. The one exception is Walmart net income, labelled FY2025 for what the table shows as the first (latest) column. That error is why Walmart's net margin was **refused** ("the inputs cover different periods"): the guard against mixing periods stopped a wrong figure from appearing, which is what it is for, but the root cause is a wrong period on one cell.

## 5. Computed cells (12 scored)

Operating margin and net margin were computed from the cells above. Six were correct, five were missed because an input was missing or unreliable (Microsoft twice, NVIDIA, Palantir, and Walmart because of the period conflict above), and one was wrong because its input was the wrong line. The computation was never the failing step: the code did what it was given and refused when its inputs were missing, mismatched or uncertain.

## 6. Tie-out checks

Six checks ran. Two tied, three could not be checked because an input was missing or uncertain, and one did not tie: Walmart's gross profit differs from revenue minus cost of revenue by $6.75 billion (3.8%). *Hypothesis:* a definition difference (Walmart's revenue includes membership income, while its reported gross profit is on net sales). The check is shown on the cell with the gap and the note that companies classify costs differently.

## What I would change next, in order

1. Look at why Microsoft's tables were flagged, since that is the largest single loss (6 cells).
2. Tighten the pack prompts for net income (attributable to the company) and interest expense (gross).
3. Prefer a table figure to a rounded figure in the prose.
4. Look at the NVIDIA and Palantir cases where the figure exists but retrieval did not surface it.

## Limits of this evaluation

Six filings, one run, one model, no repeats, so run-to-run variation is not measured. Gold labels come from XBRL tags, which are sometimes a narrower definition than the question asks. Free cash flow, gross margin and free cash flow margin are not scored because companies define them themselves and XBRL holds no gold for them. The report was rescored once, offline, after two scoring mistakes of mine were found in the first output; no model call was repeated.
