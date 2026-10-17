"""Turning the model's proposed columns into a plan: pure rules, no model and no database."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from backend.app.matrix.models import ColumnRecord
from backend.app.packs.loader import get_packs
from backend.app.planning.plan import MAX_COLUMNS, resolve_plan

PACK = get_packs()["credit_pe_diligence"]


def existing(name: str) -> ColumnRecord:
    now = datetime.now(UTC)
    return ColumnRecord(
        id=uuid4(),
        workspace_id=uuid4(),
        position=1,
        version=1,
        name=name,
        prompt="Some question?",
        type="text",
        allowed_values=[],
        applicability=None,
        evidence_required=True,
        depends_on=[],
        created_at=now,
        updated_at=now,
    )


def plan(columns, *, pack=PACK, have=(), **kw):
    return resolve_plan(
        {"columns": columns}, question="q", pack=pack, existing=list(have), **kw
    )


def new(name="Backlog", prompt="What is the order backlog?", **kw) -> dict:
    return {"name": name, "prompt": prompt, "type": "currency", **kw}


# --- pack columns ---------------------------------------------------------------------------------


def test_a_pack_column_is_taken_from_the_pack_not_from_the_models_wording():
    result = plan(
        [
            {
                "pack_key": "revenue",
                "name": "Sales!!",
                "prompt": "tell me sales",
                "type": "text",
            }
        ]
    )

    column = result.columns[0]
    assert column.source == "pack" and column.pack_key == "revenue"
    assert column.name == "Revenue" and column.type == "currency"
    assert "total net revenue" in column.prompt


def test_an_unknown_pack_key_falls_back_to_the_models_column_and_says_so():
    result = plan([new() | {"pack_key": "no_such_key"}])

    assert result.columns[0].source == "new"
    assert any("no_such_key" in n for n in result.notes)


def test_an_unknown_pack_key_with_nothing_else_is_dropped():
    result = plan([{"pack_key": "no_such_key"}])

    assert result.columns == [] and result.notes


def test_without_a_pack_the_models_columns_still_work():
    result = plan([new()], pack=None)

    assert [c.source for c in result.columns] == ["new"]


# --- new columns follow the same rules as hand-made ones --------------------------------------------------


def test_a_new_column_is_validated_like_a_hand_made_one():
    result = plan([new(type="select", allowed_values=["only one"])])

    assert result.columns == []
    assert any("at least two" in n for n in result.notes)


def test_a_select_column_keeps_its_labels():
    result = plan([new(type="select", allowed_values=["High", "Medium", "Low"])])

    assert result.columns[0].allowed_values == ["High", "Medium", "Low"]


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "", "prompt": "q?"},
        {"name": "A", "prompt": "   "},
        {"name": "A", "prompt": "q?", "type": "spreadsheet"},
        {"prompt": "no name"},
        {"name": 5, "prompt": "q?"},
        "just a string",
        None,
    ],
)
def test_junk_columns_are_dropped_not_crashed_on(bad):
    assert plan([bad]).columns == []


def test_allowed_values_that_are_not_text_are_ignored():
    result = plan([new(type="select", allowed_values=["A", 7, None, "B"])])

    assert result.columns[0].allowed_values == ["A", "B"]


# --- reuse, repeats, limits -------------------------------------------------------------------------------


def test_a_column_that_already_exists_is_reused_not_created_again():
    have = existing("Revenue")

    result = plan([{"pack_key": "revenue"}], have=[have])

    column = result.columns[0]
    assert column.source == "existing" and column.column_id == have.id


def test_existing_columns_match_by_name_ignoring_case():
    have = existing("backlog")

    assert plan([new()], have=[have]).columns[0].column_id == have.id


def test_the_same_column_proposed_twice_is_kept_once():
    result = plan([new(), new(name="BACKLOG")])

    assert len(result.columns) == 1
    assert any("twice" in n for n in result.notes)


def test_too_many_columns_are_cut_and_the_cut_is_reported():
    many = [
        new(name=f"Metric {i}", prompt=f"What is metric {i}?")
        for i in range(MAX_COLUMNS + 4)
    ]

    result = plan(many)

    assert len(result.columns) == MAX_COLUMNS
    assert result.columns[0].name == "Metric 0"  # the first ones are kept
    assert any("kept the first" in n for n in result.notes)


# --- a reply of the wrong shape --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw", [None, "text", 5, [], {}, {"columns": "nope"}, {"columns": None}]
)
def test_a_reply_of_the_wrong_shape_gives_an_empty_plan_with_a_reason(raw):
    result = resolve_plan(raw, question="q", pack=PACK, existing=[])

    assert result.columns == [] and result.notes


def test_the_plan_remembers_the_question():
    assert resolve_plan(
        {"columns": []}, question="What is revenue?", pack=PACK, existing=[]
    ).question == ("What is revenue?")


# --- computed columns ---------------------------------------------------------------------------------------------------


def names(result) -> list[str]:
    return [c.name for c in result.columns]


def test_a_computed_column_brings_the_columns_it_reads_and_they_come_first():
    result = plan([{"pack_key": "operating_margin"}])

    assert names(result) == ["Operating income", "Revenue", "Operating margin"]
    margin = result.columns[-1]
    assert margin.derived is not None and margin.inputs == [
        "Operating income",
        "Revenue",
    ]
    assert any(
        "added 'Operating income' because 'Operating margin'" in n for n in result.notes
    )


def test_inputs_the_model_already_chose_are_not_added_twice_or_reported_as_added():
    result = plan([{"pack_key": "revenue"}, {"pack_key": "operating_margin"}])

    assert names(result) == ["Revenue", "Operating income", "Operating margin"]
    assert not any("added 'Revenue'" in n for n in result.notes)


def test_two_computed_columns_sharing_an_input_add_it_once():
    result = plan([{"pack_key": "operating_margin"}, {"pack_key": "net_margin"}])

    assert names(result).count("Revenue") == 1
    assert names(result) == [
        "Operating income",
        "Revenue",
        "Operating margin",
        "Net income",
        "Net margin",
    ]


def test_inputs_that_already_exist_are_reused_not_recreated():
    have = [existing("Operating income"), existing("Revenue")]

    result = plan([{"pack_key": "operating_margin"}], have=have)

    assert [(c.name, c.source) for c in result.columns] == [
        ("Operating income", "existing"),
        ("Revenue", "existing"),
        ("Operating margin", "pack"),
    ]


def test_a_computed_column_that_already_exists_is_reused():
    have = [existing("Operating margin")]

    result = plan([{"pack_key": "operating_margin"}], have=have)

    assert result.columns[-1].source == "existing"


def test_net_debt_to_ebitda_brings_all_three_inputs_in_order():
    result = plan([{"pack_key": "net_debt_to_ebitda"}])

    # the inputs come in the order the pack lists them: debt, cash, then the measure
    assert names(result) == [
        "Total debt",
        "Cash and equivalents",
        "Adjusted EBITDA",
        "Net debt to EBITDA",
    ]


def test_a_computed_column_cut_by_the_limit_is_left_out_and_the_note_names_it():
    result = plan([{"pack_key": "net_debt_to_ebitda"}], max_columns=3)

    assert "Net debt to EBITDA" not in names(result)
    assert any("left out 'Net debt to EBITDA'" in n for n in result.notes)


def test_a_computed_planned_column_can_be_turned_into_a_valid_draft_only_with_its_inputs_ids():
    from uuid import uuid4

    margin = plan([{"pack_key": "operating_margin"}]).columns[-1]

    assert margin.draft([uuid4(), uuid4()]).derivation is not None
    with pytest.raises(ValueError, match="needs 2 input columns"):
        margin.draft()


def test_a_column_the_model_invents_is_never_computed():
    result = plan([new() | {"derived": {"op": "ratio"}, "inputs": ["A", "B"]}])

    assert result.columns[0].derived is None and result.columns[0].inputs == []
