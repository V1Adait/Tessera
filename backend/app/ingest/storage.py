"""Saving an upload to disk safely.

The file is streamed to a temporary file while it is hashed and counted, so an oversized upload
is refused as soon as it crosses the limit. Once it is validated it is moved to a path built from
its SHA-256 hash. The client's file name never touches the file system, which closes the whole
class of path-traversal tricks, and identical files share one copy.
"""

import asyncio
import hashlib
import os
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from backend.app.ingest.intake import (
    IntakeError,
    IntakeLimits,
    detect_kind,
    safe_display_name,
)
from backend.app.ingest.models import DocKind

_CHUNK = 1024 * 1024
_SHA_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class StoredFile:
    sha256: str
    size: int
    path: Path
    kind: DocKind
    display_name: str


def path_for(upload_dir: Path, sha256: str) -> Path:
    """Where a file with this hash lives. Refuses anything that is not a plain hash."""
    if not _SHA_RE.fullmatch(sha256):
        raise ValueError("not a sha256 hash")
    return upload_dir / sha256[:2] / sha256


async def save_upload(
    read: Callable[[int], Awaitable[bytes]],
    filename: str | None,
    upload_dir: Path,
    limits: IntakeLimits,
) -> StoredFile:
    """Stream, hash, validate and store an upload. Raises IntakeError if it is refused; nothing
    is left on disk in that case."""
    display_name = safe_display_name(filename)
    temp_dir = upload_dir / ".tmp"
    await asyncio.to_thread(temp_dir.mkdir, parents=True, exist_ok=True)
    temp_path = temp_dir / f"upload-{uuid.uuid4().hex}"
    hasher = hashlib.sha256()
    size = 0

    try:
        with temp_path.open("wb") as out:
            while chunk := await read(_CHUNK):
                size += len(chunk)
                if size > limits.max_bytes:
                    raise IntakeError(
                        413,
                        "too_large",
                        f"The file is larger than the {limits.max_bytes // (1024 * 1024)} MB limit.",
                    )
                hasher.update(chunk)
                await asyncio.to_thread(out.write, chunk)
        if size == 0:
            raise IntakeError(400, "empty_file", "The file is empty.")

        kind = await asyncio.to_thread(detect_kind, temp_path, display_name, limits)
        sha256 = hasher.hexdigest()
        final = path_for(upload_dir, sha256)
        await asyncio.to_thread(final.parent.mkdir, parents=True, exist_ok=True)
        if final.exists():
            temp_path.unlink(missing_ok=True)  # the same bytes are already stored
        else:
            await asyncio.to_thread(os.replace, temp_path, final)
        return StoredFile(
            sha256=sha256, size=size, path=final, kind=kind, display_name=display_name
        )
    finally:
        temp_path.unlink(missing_ok=True)  # no-op after a successful move
