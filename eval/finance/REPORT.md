# Finance eval report

> Rescored offline from the saved run `20261018-163052.json` (no model calls were repeated). Two fixes to the scoring, made after reading the first report: free cash flow, gross margin and free cash flow margin are no longer scored (they are extracted columns that companies state or define themselves, and XBRL has no gold for them; the first report counted them as computed misses by mistake), and the expected fiscal year is the year the period ends (XBRL labels Salesforce's year ending January 2026 as 2025). Six filings is too small a sample for a precise accuracy figure: read the patterns, not the decimals.

Generated 2026-10-18 16:32 UTC by `eval/finance/run_eval.py`. Extraction model `anthropic/claude-haiku-5.5`. Gold values are the XBRL numbers each company filed with its 10-K. Every figure below was produced by that run.

Filings run: 6 of 6

**Extracted values:** 36/50 (72%) correct | 3 rounded | 0 sign | **3 wrong values** | 8 missed

**Computed values (from gold inputs):** 6/12 (50%) correct | 0 rounded | 0 sign | **1 wrong values** | 5 missed

## By metric

|  | scored | correct | rounded | sign | wrong | missed |
|---|---|---|---|---|---|---|
| Revenue | 6 | 5 | 0 | 0 | 0 | 1 |
| Cost of revenue | 6 | 4 | 0 | 0 | 0 | 2 |
| Gross profit | 5 | 4 | 0 | 0 | 0 | 1 |
| Operating income | 6 | 3 | 1 | 0 | 0 | 2 |
| Net income | 6 | 4 | 0 | 0 | 2 | 0 |
| Cash and equivalents | 6 | 4 | 1 | 0 | 0 | 1 |
| Operating cash flow | 6 | 5 | 1 | 0 | 0 | 0 |
| Capital expenditures | 5 | 5 | 0 | 0 | 0 | 0 |
| Interest expense | 4 | 2 | 0 | 0 | 1 | 1 |
| Operating margin | 6 | 3 | 0 | 0 | 0 | 3 |
| Net margin | 6 | 3 | 0 | 0 | 1 | 2 |

## By company

|  | scored | correct | rounded | sign | wrong | missed |
|---|---|---|---|---|---|---|
| AAPL | 10 | 10 | 0 | 0 | 0 | 0 |
| CRM | 11 | 10 | 1 | 0 | 0 | 0 |
| MSFT | 11 | 4 | 1 | 0 | 0 | 6 |
| NVDA | 10 | 7 | 0 | 0 | 0 | 3 |
| PLTR | 10 | 5 | 0 | 0 | 2 | 3 |
| WMT | 10 | 6 | 1 | 0 | 2 | 1 |

## Periods

Answered extracted cells carrying the filing's fiscal-year label: 43/44 (98%) (cells with no period: 0).

## Tie-out checks

Results over all filings: {'ties': 2, 'cannot_check': 3, 'does_not_tie': 1}

- MSFT | Gross profit = revenue - cost of revenue | cannot_check | cannot check: 'Revenue' is uncertain; 'Gross profit' is uncertain
- NVDA | Gross profit = revenue - cost of revenue | cannot_check | cannot check: 'Cost of revenue' is not found
- PLTR | Gross profit = revenue - cost of revenue | cannot_check | cannot check: 'Cost of revenue' is uncertain
- WMT | Gross profit = revenue - cost of revenue | does_not_tie | Revenue - Cost of revenue = $177.77 billion, but 'Gross profit' says $171.02 billion: a gap of $6.75 billion (3.8%). Companies can classify costs differently, so a gap may be a definition, not an error.

## Evidence audit

Quotes checked to the character against the document text: 63; placed by chunk: 1; **failed: 0**. Answered cells without evidence: 0 of 51.

## Cost and speed

Extraction cost: $0.0811 over 77 model calls (0 errors). Tokens in/out: 639,760 / 34,198. Median ingest per filing: 23.8s; median fill of the extracted columns: 36.6s.

## Every cell that was not correct

| company | metric | verdict | expected | got | status / reason |
|---|---|---|---|---|---|
| CRM | Operating income | ROUNDED | 8331000000 | 8300000000 | answered:  |
| MSFT | Revenue | MISSED | 331839000000 | None | low_confidence: the quoted passage was flagged as unreliable when the document was read: table_total_mismatch:r5c2, table_total_mismatch:r5c4, table_total_m |
| MSFT | Gross profit | MISSED | 225465000000 | None | low_confidence: the quoted passage was flagged as unreliable when the document was read: table_total_mismatch:r5c2, table_total_mismatch:r5c4, table_total_m |
| MSFT | Cash and equivalents | MISSED | 20935000000 | None | low_confidence: the quoted passage was flagged as unreliable when the document was read: table_total_mismatch:r17c1, table_total_mismatch:r6c2, table_total_ |
| MSFT | Operating cash flow | ROUNDED | 182935000000 | 182900000000 | answered:  |
| MSFT | Interest expense | MISSED | 3051000000 | None | low_confidence: the quoted passage was flagged as unreliable when the document was read: table_total_mismatch:r8c1 |
| MSFT | Operating margin | MISSED | 46.78081840892722073053498835 | None | low_confidence: an input cannot be relied on: 'Revenue' is uncertain |
| MSFT | Net margin | MISSED | 40.30538905915217921944075290 | None | low_confidence: an input cannot be relied on: 'Revenue' is uncertain |
| NVDA | Cost of revenue | MISSED | 62475000000 | None | not_found: The documents give revenue ($215,938 million) and gross margin (71.1%) for fiscal 2026 but never state the cost of revenue amount; deriving  |
| NVDA | Operating income | MISSED | 130387000000 | None | not_found: The documents give only segment operating income (Total $139,297 million, which is the sum of reportable segments) and an operating income m |
| NVDA | Operating margin | MISSED | 60.38168363141272031786901796 | None | not_found: needs its inputs: 'Operating income' is not found in the documents |
| PLTR | Cost of revenue | MISSED | 789177000 | None | low_confidence: the amount comes from a table whose units (such as 'in millions') could not be found, so it is shown as printed and its scale is unknown |
| PLTR | Operating income | MISSED | 1414015000 | None | not_found: The documents reference income from operations and a reconciliation to it, but do not include a GAAP operating income figure. The only incom |
| PLTR | Net income | WRONG | 1625033000 | 1634644000 | answered:  |
| PLTR | Operating margin | MISSED | 31.59495165398040776271236431 | None | not_found: needs its inputs: 'Operating income' is not found in the documents |
| PLTR | Net margin | WRONG | 36.30996776634105293640008169 | 36.52 | answered:  |
| WMT | Net income | WRONG | 21893000000 | 22270000000 | answered:  |
| WMT | Cash and equivalents | ROUNDED | 10727000000 | 10700000000 | answered:  |
| WMT | Interest expense | WRONG | 2318000000 | 2431000000 | answered:  |
| WMT | Net margin | MISSED | 3.069845182658102004731036243 | None | low_confidence: the inputs cover different periods (FY2025, FY2026); they cannot be combined |

## Not scored (no XBRL value for that filing)

- AAPL: Interest expense
- NVDA: Capital expenditures
- PLTR: Interest expense
- WMT: Gross profit
