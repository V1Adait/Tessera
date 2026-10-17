"""Loading the packs that ship with Tessera from backend/app/packs/data/*.json."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from backend.app.finance.checks import CheckSpec
from backend.app.packs.models import Pack

DATA_DIR = Path(__file__).parent / "data"


class PackError(Exception):
    """A pack file is unreadable or invalid. The message names the file."""


def load_packs(directory: Path = DATA_DIR) -> dict[str, Pack]:
    """Every *.json file in `directory`, by pack id. One bad file fails the whole load, loudly:
    a half-loaded set of packs would be a silent bug."""
    if not directory.is_dir():
        raise PackError(f"the pack folder does not exist: {directory}")
    packs: dict[str, Pack] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            pack = Pack.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, ValidationError) as exc:
            raise PackError(f"{path.name}: {exc}") from exc
        if pack.id in packs:
            raise PackError(f"{path.name}: the pack id '{pack.id}' is already used")
        packs[pack.id] = pack
    return packs


@lru_cache
def get_packs() -> dict[str, Pack]:
    """The shipped packs, loaded once."""
    return load_packs()


def get_check_specs() -> list[CheckSpec]:
    """The tie-out checks of every shipped pack. A check only runs where all its columns exist."""
    return [spec for pack in get_packs().values() for spec in pack.check_specs()]
