"""Report invisible or unusual characters written literally in source files.

    uv run python -m scratch.find_invisible backend

Invisible characters in code are a classic way to hide things from reviewers, so this is a handy
check to run on any file that handles text security.
"""

import sys
import unicodedata
from pathlib import Path

SUSPICIOUS = {"Cf", "Zs", "Cc", "Zl", "Zp", "Co", "Cn"}


def scan(path: Path) -> list[tuple[int, str]]:
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for ch in line:
            if ord(ch) > 126 and unicodedata.category(ch) in SUSPICIOUS:
                found.append((number, f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}"))
    return found


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    total = 0
    for path in sorted(root.rglob("*.py")):
        if ".venv" in path.parts or "node_modules" in path.parts:
            continue
        for number, what in scan(path):
            print(f"{path}:{number}: {what}")
            total += 1
    print(f"{total} suspicious character(s) found")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
