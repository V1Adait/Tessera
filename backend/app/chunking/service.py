"""The glue around the pure splitter: settings to limits, stored rows back to blocks, and the
"rebuild this document's chunks from its stored blocks" operation.

Why rebuild from stored blocks and not re-parse: assigning a document to an entity changes the
search text (it names the entity), but the file did not change, so parsing again would waste
time and could even change the blocks.
"""

import asyncio
from uuid import UUID

import asyncpg

from backend.app.chunking.models import ChunkLimits, ChunkSet, DocContext
from backend.app.chunking.splitter import build_chunks
from backend.app.config import Settings
from backend.app.db import repository as repo
from backend.app.db.records import BlockRecord, DocumentRecord
from backend.app.ingest.models import Block, Location, TableData


def limits_from_settings(settings: Settings) -> ChunkLimits:
    return ChunkLimits(
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        parent_max_tokens=settings.chunk_parent_max_tokens,
    )


def block_from_record(record: BlockRecord) -> Block:
    """A stored block back into the parser's Block (only what chunking needs is restored)."""
    table = None
    labels: list[str] = []
    if record.data:
        if "rows" in record.data:
            table = TableData(**record.data)
        labels = list(record.data.get("labels", []))
    return Block(
        type=record.type,
        text=record.text,
        location=Location(**record.location),
        level=record.level,
        table=table,
        labels=labels,
        confidence=record.confidence,
        flags=list(record.flags),
    )


async def document_context(pool: asyncpg.Pool, document: DocumentRecord) -> DocContext:
    return DocContext(
        label=document.filename, entity=await repo.entity_name(pool, document.entity_id)
    )


async def make_chunks(
    blocks: list[Block], context: DocContext, settings: Settings
) -> ChunkSet:
    """Chunking is quick, but a huge document is still CPU work: keep it off the event loop."""
    return await asyncio.to_thread(
        build_chunks, blocks, context, limits_from_settings(settings)
    )


async def rechunk(
    pool: asyncpg.Pool, settings: Settings, document_id: UUID
) -> int | None:
    """Rebuild one document's chunks from its stored blocks. Returns the chunk count, or None
    if the document does not exist. A document with no blocks ends up with no chunks."""
    document = await repo.get_document(pool, document_id)
    if document is None:
        return None
    blocks: list[Block] = []
    offset = 0
    while True:
        page = await repo.list_blocks(pool, document_id, limit=500, offset=offset)
        blocks += [block_from_record(r) for r in page]
        if len(page) < 500:
            break
        offset += 500
    chunk_set = await make_chunks(
        blocks, await document_context(pool, document), settings
    )
    await repo.save_chunks(pool, document_id, chunk_set)
    return len(chunk_set.chunks)
