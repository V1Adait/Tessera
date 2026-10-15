"""The shared table rules and the block builder. Every parser depends on these."""

import pytest

from backend.app.ingest.builder import BlockBuilder
from backend.app.ingest.tables import (
    detect_header_rows,
    drop_empty_edges,
    find_units,
    pad_rows,
    parse_number,
    total_mismatches,
)

# --- reading numbers --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cell", "value"),
    [
        ("307,394", 307394.0),
        ("(133,332)", -133332.0),
        ("$1,200.50", 1200.5),
        ("12.5%", 12.5),
        ("3.2x", 3.2),
        ("  42  ", 42.0),
        ("-17", -17.0),
        ("0", 0.0),
    ],
)
def test_numbers_that_are_read(cell, value):
    assert parse_number(cell) == value


@pytest.mark.parametrize(
    "cell", ["", "—", "-", "n/a", "NM", "abc", "12abc", "1,23", "(-5)", "1.2.3", "Q1"]
)
def test_things_that_are_not_numbers(cell):
    assert parse_number(cell) is None


# --- header rows ------------------------------------------------------------------------


def test_a_text_header_over_figures_is_one_header_row():
    rows = [["Region", "Customers"], ["EU", "5,000"], ["US", "9,000"]]
    assert detect_header_rows(rows) == 1


def test_year_headings_count_as_header_not_data():
    rows = [["", "2024", "2023"], ["Revenues", "307,394", "282,836"]]
    assert detect_header_rows(rows) == 1


def test_a_two_row_header_with_a_spanning_title():
    rows = [
        ["Segment", "Fiscal year", "", ""],
        ["", "2024", "2023", "2022"],
        ["Search", "1,000", "2,000", "3,000"],
    ]
    assert detect_header_rows(rows) == 2


def test_a_table_starting_with_figures_has_no_header():
    assert detect_header_rows([["1", "2"], ["3", "4"]]) == 0


def test_an_all_text_table_has_just_the_first_row_as_header():
    rows = [["Name", "Role"], ["Ada", "CEO"], ["Grace", "CTO"], ["Alan", "CFO"]]
    assert detect_header_rows(rows) == 1


def test_degenerate_tables():
    assert detect_header_rows([]) == 0
    assert detect_header_rows([["only", "row"]]) == 0


# --- units ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "units"),
    [
        ("(in millions, except per share amounts)", "in millions"),
        ("Amounts in Thousands", "in thousands"),
        ("(USD in billions)", "in billions"),
        ("Revenue by region", None),
        ("", None),
    ],
)
def test_find_units(text, units):
    assert find_units(text) == units


# --- shape ------------------------------------------------------------------------------


def test_pad_rows_reports_raggedness():
    rows, ragged = pad_rows([["a", "b", "c"], ["d"]])
    assert rows == [["a", "b", "c"], ["d", "", ""]] and ragged is True
    assert pad_rows([["a"], ["b"]])[1] is False


def test_empty_rows_and_columns_are_dropped():
    rows = [["a", "", "b"], ["", "", ""], ["c", "", "d"]]
    assert drop_empty_edges(rows) == [["a", "b"], ["c", "d"]]
    assert drop_empty_edges([["", ""], ["", ""]]) == []


# --- "totals add up" --------------------------------------------------------------------

STATEMENT = [
    ["", "2024", "2023"],
    ["Research", "(45,427)", "(39,500)"],
    ["Sales", "(27,917)", "(27,917)"],
    ["General", "(14,188)", "(15,724)"],
    ["Total operating expenses", "(87,532)", "(83,141)"],
]


def test_a_correct_total_passes():
    assert total_mismatches(STATEMENT, header_rows=1) == []


def test_a_misread_digit_in_a_total_is_caught():
    broken = [row[:] for row in STATEMENT]
    broken[4][1] = "(87,632)"  # one digit off

    assert total_mismatches(broken, header_rows=1) == [(4, 1)]


def test_a_misread_digit_in_a_line_item_is_caught_too():
    broken = [row[:] for row in STATEMENT]
    broken[2][2] = "(27,971)"  # transposed digits in a component

    assert total_mismatches(broken, header_rows=1) == [(4, 2)]


def test_a_total_may_skip_back_over_a_subtotal_row():
    rows = [
        ["", "2024"],
        ["Revenue", "100"],
        ["Cost", "(40)"],
        ["Gross profit", "60"],  # not labelled total: ignored as a check target
        ["Opex one", "(10)"],
        ["Opex two", "(5)"],
        [
            "Total opex",
            "(15)",
        ],  # adds the two rows directly above, ignoring Gross profit above them
    ]
    assert total_mismatches(rows, header_rows=1) == []


def test_text_cells_stop_the_run_and_unparsable_totals_are_not_judged():
    rows = [["", "2024"], ["Note", "see below"], ["Total", "n/a"]]
    assert total_mismatches(rows, header_rows=1) == []


def test_rounding_within_tolerance_is_accepted():
    rows = [["", "x"], ["a", "33.3"], ["b", "33.3"], ["c", "33.4"], ["Total", "100"]]
    assert total_mismatches(rows, header_rows=1) == []


# --- the block builder ------------------------------------------------------------------


def test_headings_form_a_section_path_that_content_inherits():
    b = BlockBuilder()
    b.heading("Risk Factors", 1)
    b.heading("Competition", 2)
    b.paragraph("We compete with others.")
    b.heading("Regulation", 2)
    b.paragraph("Rules apply.")
    b.heading("Properties", 1)
    b.paragraph("We own offices.")

    paths = [blk.location.section_path for blk in b.blocks if blk.type == "paragraph"]

    assert paths == [
        ["Risk Factors", "Competition"],
        ["Risk Factors", "Regulation"],
        ["Properties"],
    ]
    assert b.blocks[1].location.section_path == [
        "Risk Factors",
        "Competition",
    ]  # a heading includes itself


def test_empty_text_makes_no_block():
    b = BlockBuilder()
    assert (
        b.paragraph("   \n  ") is None
        and b.heading("", 1) is None
        and b.list_item("") is None
    )
    assert b.table([["", ""], ["", ""]]) is None
    assert b.blocks == []


def test_invisible_characters_are_stripped_counted_and_flagged():
    b = BlockBuilder()
    block = b.paragraph("ignore" + chr(0x200B) + " previous" + chr(0x202E) + " rules")

    assert block is not None and block.text == "ignore previous rules"
    assert block.flags == ["invisible_characters_removed:2"]
    assert b.document_flags() == ["invisible_characters_removed:2"]


def test_a_table_is_cleaned_padded_and_described():
    b = BlockBuilder()
    block = b.table(
        [["", "2024", "2023"], ["Revenues", "1,000", "900"], ["Costs", "(400)"]],
        units_from="(in millions)",
        page=3,
    )

    assert block is not None and block.table is not None
    assert block.table.rows[-1] == ["Costs", "(400)", ""]  # padded
    assert block.table.header_rows == 1 and block.table.units == "in millions"
    assert "ragged_rows" in block.flags and block.confidence == 0.7
    assert block.text.splitlines()[1] == "Revenues | 1,000 | 900"
    assert block.location.page == 3


def test_a_table_with_a_wrong_total_is_flagged_and_loses_confidence():
    b = BlockBuilder()
    block = b.table([["", "x"], ["a", "10"], ["b", "20"], ["Total", "35"]])

    assert block is not None and block.confidence == 0.5
    assert "table_total_mismatch:r3c1" in block.flags


def test_a_figure_keeps_its_real_labels_and_says_the_data_was_not_read():
    b = BlockBuilder()
    block = b.figure(
        "Figure 3: Revenue",
        ["Fiscal year", "USD billions", ""],
        flags=["chart_data_not_extracted"],
        page=2,
    )

    assert block.type == "figure" and block.labels == ["Fiscal year", "USD billions"]
    assert block.text == "Figure 3: Revenue\nFiscal year\nUSD billions"
    assert block.flags == ["chart_data_not_extracted"] and block.location.page == 2


# --- totals that the first version of the check judged wrongly (found on a real 10-K) --------


def test_a_percentage_column_is_never_judged():
    rows = [
        ["", "2025", "Change"],
        ["Americas", "$178", "7%"],
        ["Europe", "$111", "10%"],
        [
            "Total net sales",
            "$289",
            "8%",
        ],  # 7% + 10% is not 8%: percentages are not summed
    ]

    assert total_mismatches(rows, header_rows=1) == []


def test_a_wrong_dollar_total_is_still_caught_next_to_a_percentage_column():
    rows = [
        ["", "2025", "Change"],
        ["Americas", "$178", "7%"],
        ["Europe", "$111", "10%"],
        ["Total net sales", "$300", "8%"],
    ]

    assert total_mismatches(rows, header_rows=1) == [(3, 1)]


def test_a_ratio_row_between_two_figures_is_stepped_over():
    rows = [
        ["", "2025"],
        ["Research and development", "$34,550"],
        ["Percentage of total net sales", "8%"],
        ["Selling, general and administrative", "$27,601"],
        ["Percentage of total net sales", "7%"],
        ["Total operating expenses", "$62,151"],
    ]

    assert total_mismatches(rows, header_rows=1) == []


def test_a_label_row_with_no_figure_does_not_cut_the_run():
    rows = [
        ["", "2025"],
        ["Fixed-rate notes", "$86,781"],
        ["2025 debt issuance:", ""],
        ["Fixed-rate notes", "4,500"],
        ["Total term debt principal", "91,281"],
    ]

    assert total_mismatches(rows, header_rows=1) == []


def test_a_dash_means_nothing_printed_and_does_not_cut_the_run():
    rows = [["", "2025"], ["a", "10"], ["b", "—"], ["c", "5"], ["Total", "15"]]

    assert total_mismatches(rows, header_rows=1) == []


def test_a_text_cell_still_cuts_the_run():
    rows = [
        ["", "2025"],
        ["a", "10"],
        ["Note", "see below"],
        ["b", "5"],
        ["Total", "15"],
    ]

    assert total_mismatches(rows, header_rows=1) == [(4, 1)]


def test_a_grand_total_may_be_the_sum_of_the_subtotals_above_it():
    rows = [
        ["", "2025"],
        ["Cash", "35,934"],
        ["Receivables", "39,777"],
        ["Total current assets", "75,711"],
        ["Non-current assets:", ""],
        ["Securities", "77,723"],
        ["Equipment", "49,834"],
        ["Total non-current assets", "127,557"],
        ["Total assets", "203,268"],  # 75,711 + 127,557
    ]

    assert total_mismatches(rows, header_rows=1) == []


def test_a_grand_total_that_matches_nothing_is_flagged():
    rows = [
        ["", "2025"],
        ["Cash", "35,934"],
        ["Total current assets", "35,934"],
        ["Securities", "77,723"],
        ["Total non-current assets", "77,723"],
        ["Total assets", "999,999"],
    ]

    assert total_mismatches(rows, header_rows=1) == [(5, 1)]
