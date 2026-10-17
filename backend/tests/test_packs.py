"""Domain packs: the file format and the loader. Pure, no database."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.matrix.models import ColumnDraft
from backend.app.packs.loader import PackError, get_packs, load_packs
from backend.app.packs.models import Pack, PackColumn


def col(**over) -> dict:
    return {"key": "revenue", "name": "Revenue", "prompt": "What was revenue?", **over}


def pack(*columns: dict, **over) -> dict:
    return {
        "id": "demo",
        "name": "Demo",
        "description": "A test pack.",
        "columns": list(columns) or [col()],
        **over,
    }


# --- the shipped pack ---------------------------------------------------------------------------


def test_the_shipped_packs_are_valid_and_nonempty():
    packs = get_packs()

    assert "credit_pe_diligence" in packs
    assert len(packs["credit_pe_diligence"].columns) >= 12


def test_every_shipped_column_becomes_a_valid_matrix_column():
    for pack_ in get_packs().values():
        for column in pack_.columns:
            ids = [uuid4() for _ in column.derived.inputs] if column.derived else None
            assert isinstance(column.draft(ids), ColumnDraft)


def test_the_shipped_pack_has_labeled_and_typed_columns():
    columns = {c.key: c for c in get_packs()["credit_pe_diligence"].columns}

    assert columns["customer_concentration"].type == "select"
    assert len(columns["customer_concentration"].allowed_values) >= 2
    assert columns["going_concern"].type == "yes_no"
    assert columns["revenue"].type == "currency"


def test_no_shipped_prompt_asks_the_model_to_compare_companies():
    """A cell is answered from ONE company's documents. A prompt that mentions other companies could
    never be answered from them."""
    for column in get_packs()["credit_pe_diligence"].columns:
        text = column.prompt.casefold()
        assert "compared with other" not in text and "peers" not in text


# --- the format ---------------------------------------------------------------------------------


def test_a_valid_pack_loads():
    assert Pack.model_validate(pack()).column("revenue") is not None


def test_a_pack_needs_at_least_one_column():
    with pytest.raises(ValidationError):
        Pack.model_validate(pack(columns=[]))


def test_two_columns_cannot_share_a_key_or_a_name():
    with pytest.raises(ValidationError, match="share a key"):
        Pack.model_validate(pack(col(), col(name="Other")))
    with pytest.raises(ValidationError, match="share a name"):
        Pack.model_validate(pack(col(), col(key="rev2", name="REVENUE")))


@pytest.mark.parametrize("bad", ["Revenue", "1revenue", "has space", "", "x" * 60])
def test_a_column_key_must_be_a_plain_slug(bad):
    with pytest.raises(ValidationError):
        PackColumn.model_validate(col(key=bad))


def test_a_pack_column_follows_the_same_rules_as_a_hand_made_one():
    with pytest.raises(ValidationError, match="at least two"):
        PackColumn.model_validate(col(type="select", allowed_values=["only one"]))
    with pytest.raises(ValidationError, match="only apply to select"):
        PackColumn.model_validate(col(type="number", allowed_values=["a", "b"]))
    with pytest.raises(ValidationError):
        PackColumn.model_validate(col(type="spreadsheet"))


def test_a_blank_prompt_is_refused():
    with pytest.raises(ValidationError):
        PackColumn.model_validate(col(prompt="   "))


# --- the loader ---------------------------------------------------------------------------------


def write(directory: Path, name: str, body: object) -> None:
    (directory / name).write_text(
        json.dumps(body) if not isinstance(body, str) else body
    )


def test_the_loader_reads_every_json_file_by_pack_id(tmp_path):
    write(tmp_path, "a.json", pack(id="alpha"))
    write(tmp_path, "b.json", pack(id="beta"))
    write(tmp_path, "notes.txt", "ignored")

    assert set(load_packs(tmp_path)) == {"alpha", "beta"}


def test_one_bad_file_fails_the_whole_load_and_names_the_file(tmp_path):
    write(tmp_path, "good.json", pack(id="good"))
    write(tmp_path, "bad.json", pack(id="bad", columns=[col(type="select")]))

    with pytest.raises(PackError, match="bad.json"):
        load_packs(tmp_path)


def test_invalid_json_is_a_pack_error_not_a_crash(tmp_path):
    write(tmp_path, "broken.json", "{not json")

    with pytest.raises(PackError, match="broken.json"):
        load_packs(tmp_path)


def test_two_files_with_the_same_pack_id_are_refused(tmp_path):
    write(tmp_path, "a.json", pack(id="same"))
    write(tmp_path, "b.json", pack(id="same"))

    with pytest.raises(PackError, match="already used"):
        load_packs(tmp_path)


def test_a_missing_folder_is_a_pack_error(tmp_path):
    with pytest.raises(PackError, match="does not exist"):
        load_packs(tmp_path / "nope")


# --- computed columns and checks ----------------------------------------------------------------------------


def base(key, name, type_="currency"):
    return {"key": key, "name": name, "prompt": f"What was {name}?", "type": type_}


def derived(key, name, op, inputs, type_="number", as_=None):
    d = {"op": op, "inputs": inputs, **({"as": as_} if as_ else {})}
    return {
        "key": key,
        "name": name,
        "prompt": f"{name}, computed.",
        "type": type_,
        "derived": d,
    }


BASES = [base("revenue", "Revenue"), base("income", "Income"), base("cost", "Cost")]


def test_the_shipped_pack_has_computed_columns_and_checks():
    shipped = get_packs()["credit_pe_diligence"]

    computed = {c.key for c in shipped.columns if c.derived}
    assert {
        "operating_margin",
        "net_debt",
        "net_debt_to_ebitda",
        "interest_coverage",
    } <= computed
    assert {c.id for c in shipped.checks} >= {
        "gross_profit_ties",
        "gross_margin_ties",
        "fcf_ties",
    }


def test_a_computed_column_reads_other_columns_of_the_pack_by_key():
    result = Pack.model_validate(
        pack(
            *BASES,
            derived("margin", "Margin", "ratio", ["income", "revenue"], as_="percent"),
        )
    )

    margin = result.column("margin")
    assert margin is not None and margin.derived is not None
    assert margin.derived.names_in(result) == ["Income", "Revenue"]


def test_a_computed_column_that_reads_a_missing_key_is_refused():
    with pytest.raises(ValidationError, match="not in the pack"):
        Pack.model_validate(
            pack(
                *BASES,
                derived("mgn", "Mgn", "ratio", ["income", "ghost"], as_="percent"),
            )
        )


def test_a_computed_column_may_not_read_another_computed_column():
    with pytest.raises(ValidationError, match="itself computed"):
        Pack.model_validate(
            pack(
                *BASES,
                derived("m1", "M1", "ratio", ["income", "revenue"], as_="percent"),
                derived("m2", "M2", "ratio", ["m1", "revenue"], as_="percent"),
            )
        )


def test_a_computed_column_must_have_the_type_its_operation_gives():
    with pytest.raises(ValidationError, match="should be of type 'number'"):
        Pack.model_validate(
            pack(
                *BASES,
                derived(
                    "mgn",
                    "Mgn",
                    "ratio",
                    ["income", "revenue"],
                    type_="currency",
                    as_="percent",
                ),
            )
        )
    with pytest.raises(ValidationError, match="should be of type 'currency'"):
        Pack.model_validate(
            pack(
                *BASES,
                derived("dif", "Dif", "difference", ["income", "cost"], type_="number"),
            )
        )


def test_a_computed_column_needs_the_right_number_of_inputs_and_a_ratio_needs_its_display():
    with pytest.raises(ValidationError):
        Pack.model_validate(
            pack(
                *BASES,
                derived(
                    "mgn", "Mgn", "net_ratio", ["income", "revenue"], as_="multiple"
                ),
            )
        )
    with pytest.raises(ValidationError, match="percent or a multiple"):
        Pack.model_validate(
            pack(*BASES, derived("mgn", "Mgn", "ratio", ["income", "revenue"]))
        )


def test_with_inputs_adds_what_a_computed_column_reads_and_keeps_pack_order():
    shipped = get_packs()["credit_pe_diligence"]

    keys = [c.key for c in shipped.with_inputs(["net_debt_to_ebitda"])]

    assert keys == [
        c.key
        for c in shipped.columns
        if c.key in {"cash", "total_debt", "adjusted_ebitda", "net_debt_to_ebitda"}
    ]
    assert keys.index("total_debt") < keys.index("net_debt_to_ebitda")


def test_with_inputs_leaves_ordinary_columns_alone():
    shipped = get_packs()["credit_pe_diligence"]

    assert [c.key for c in shipped.with_inputs(["revenue"])] == ["revenue"]


def test_a_check_may_only_name_columns_of_the_pack():
    bad = {
        "id": "c",
        "name": "C",
        "kind": "difference_identity",
        "a": "revenue",
        "b": "ghost",
        "result": "income",
        "tolerance": "0.01",
    }

    with pytest.raises(ValidationError, match="not in the pack"):
        Pack.model_validate(pack(*BASES, checks=[bad]))


def test_two_checks_cannot_share_an_id():
    ok = {
        "id": "c",
        "name": "C",
        "kind": "difference_identity",
        "a": "revenue",
        "b": "cost",
        "result": "income",
        "tolerance": "0.01",
    }

    with pytest.raises(ValidationError, match="share an id"):
        Pack.model_validate(pack(*BASES, checks=[ok, ok]))


def test_check_specs_use_column_names_not_keys():
    specs = get_packs()["credit_pe_diligence"].check_specs()

    gross = next(s for s in specs if s.id == "gross_profit_ties")
    assert (gross.a, gross.b, gross.result) == (
        "Revenue",
        "Cost of revenue",
        "Gross profit",
    )


def test_adjusted_ebitda_never_accepts_another_measure_in_its_place():
    """Seen live: 'Adjusted EBITDA' was answered with adjusted operating income, which is a different
    measure and would have fed a wrong leverage ratio. The prompt now forbids substituting."""
    pack_ = get_packs()["credit_pe_diligence"]

    ebitda = pack_.column("adjusted_ebitda")
    other = pack_.column("adjusted_operating_income")
    assert ebitda is not None and "Never substitute" in ebitda.prompt
    assert "operating income" not in ebitda.prompt.replace(
        "such as adjusted operating income", ""
    )
    assert (
        other is not None and other.type == "currency"
    )  # the measure Palantir does report
