"""Retrieval eval: for each question, does a search put the passage that answers it near the top?

    .venv/Scripts/python eval/retrieval/run_eval.py FILING [--questions FILE] [--entity NAME]

What it does (nothing touches your real database or your real Qdrant collections):
  1. makes a throwaway Postgres schema and applies the migrations in it
  2. uploads FILING through the real DocumentService (real parser, real chunker)
  3. embeds the chunks with the real embedding model into throwaway Qdrant collections
  4. asks every question three ways and scores each against the gold passages
  5. drops the schema and the collections

The gold for a question is a list of short strings copied VERBATIM from the filing. A retrieved
chunk is a hit if it contains any of them (whitespace and case ignored). The same check is applied
to parents (a hit's whole section), which is what the extractor would actually read.

Metrics: recall@k (is a hit among the top k chunks?), MRR (1 / rank of the first hit), and parent
recall@5 (is a hit inside the top-5 parents?). Every number printed comes from this run.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from backend.app.chunking.expand import expand_to_parents
from backend.app.config import Settings, get_settings
from backend.app.db import repository as repo
from backend.app.db.migrate import migrate
from backend.app.db.records import ChunkHit
from backend.app.documents_service import DocumentService
from backend.app.embeddings.embedder import (
    OpenRouterEmbedder,
    make_embedding_client,
)
from backend.app.embeddings.indexer import Indexer
from backend.app.embeddings.store import QdrantStore, make_qdrant_client
from backend.app.tracing import db_recorder

KS = (1, 3, 5, 10)
PARENTS = 5
FETCH = max(KS)
_WORD = re.compile(r"[A-Za-z0-9$%.,'&-]+")
_STOP = {
    "what", "was", "were", "is", "are", "the", "a", "an", "of", "in", "on", "for", "to", "and",
    "or", "does", "do", "did", "how", "much", "many", "which", "who", "when", "where", "its",
    "it", "as", "by", "at", "with", "from", "that", "this", "has", "have", "had", "be", "been",
}  # fmt: skip


def squash(text: str) -> str:
    """Lower-case, one space between words, and one separator between table cells however many
    empty cells sat between them: a table's column layout is not part of what it says."""
    text = re.sub(r"\s*\|(?:\s*\|)*\s*", " | ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def or_query(question: str) -> str:
    """The question's content words joined with OR: the fair full-text baseline. The search as
    built ANDs every word, so a long natural-language question usually matches nothing."""
    words = [w.strip(".,'-").lower() for w in _WORD.findall(question)]
    words = [w for w in words if w and w not in _STOP]
    return " or ".join(dict.fromkeys(words))


def is_hit(text: str, gold: list[str]) -> bool:
    haystack = squash(text)
    return any(squash(g) in haystack for g in gold)


@dataclass
class Outcome:
    method: str
    first_rank: int | None  # 1-based rank of the first chunk that contains the gold
    parent_hit: bool
    seconds: float
    returned: int
    parent_tokens: int = 0  # what reading the top-5 parents would put in a prompt


async def run_method(
    method: str, question: dict, search, pool: asyncpg.Pool
) -> Outcome:
    started = time.perf_counter()
    hits: list[ChunkHit] = await search(question["question"])
    seconds = time.perf_counter() - started
    gold = question["gold"]
    first = next(
        (i + 1 for i, h in enumerate(hits) if is_hit(h.chunk.text, gold)), None
    )
    parents = await expand_to_parents(pool, hits, PARENTS)
    parent_hit = any(is_hit(p.text, gold) for p in parents)
    read = sum(c.token_count for p in parents for c in p.chunks)
    return Outcome(method, first, parent_hit, seconds, len(hits), read)


def summarise(name: str, outcomes: list[Outcome]) -> dict:
    n = len(outcomes)
    row: dict = {"method": name, "n": n}
    for k in KS:
        row[f"recall@{k}"] = sum(
            o.first_rank is not None and o.first_rank <= k for o in outcomes
        )
    row["mrr"] = sum(1 / o.first_rank for o in outcomes if o.first_rank) / n
    row[f"parent_recall@{PARENTS}"] = sum(o.parent_hit for o in outcomes)
    row["no_results"] = sum(o.returned == 0 for o in outcomes)
    row["median_s"] = sorted(o.seconds for o in outcomes)[n // 2]
    tokens = sorted(o.parent_tokens for o in outcomes)
    row["parent_tokens_median"] = tokens[n // 2]
    row["parent_tokens_max"] = tokens[-1]
    return row


def print_table(rows: list[dict]) -> None:
    n = rows[0]["n"]
    heads = [f"R@{k}" for k in KS] + [f"parentR@{PARENTS}", "MRR", "empty", "med s"]
    print(f"\n{'method':26s}" + "".join(f"{h:>10s}" for h in heads) + f"     (n = {n})")
    for r in rows:
        cells = [f"{r[f'recall@{k}']}" for k in KS]
        cells += [
            f"{r[f'parent_recall@{PARENTS}']}",
            f"{r['mrr']:.2f}",
            f"{r['no_results']}",
            f"{r['median_s']:.2f}",
        ]
        print(f"{r['method']:26s}" + "".join(f"{c:>10s}" for c in cells))
    print(
        f"\ntokens in the top-{PARENTS} parents per question (what the extractor would read):"
    )
    for r in rows:
        median, biggest = r["parent_tokens_median"], r["parent_tokens_max"]
        print(f"  {r['method']:26s} median {median:>6,} | max {biggest:>6,}")


async def build(settings: Settings, filing: Path, entity: str, schema: str):
    admin = await asyncpg.connect(settings.database_url, timeout=10)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=1,
        max_size=6,
        server_settings={"search_path": schema},
    )
    assert pool is not None
    await migrate(pool)
    workspace = await repo.default_workspace_id(pool)
    ent = await repo.create_entity(pool, workspace, entity)

    key = settings.openrouter_api_key
    assert key, "OPENROUTER_API_KEY is not set"
    client = make_embedding_client(key, settings.embedding_timeout_s)
    embedder = OpenRouterEmbedder(
        client, model=settings.embedding_model, recorder=db_recorder(pool)
    )
    prefix = f"evalr_{uuid4().hex[:8]}"
    qdrant = make_qdrant_client(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        path=settings.qdrant_path,
    )
    store = QdrantStore(
        qdrant,
        prefix=prefix,
        model=settings.embedding_model,
        payload_indexes=bool(settings.qdrant_url),
    )
    indexer = Indexer(pool, settings, embedder, store)
    service = DocumentService(pool, settings, indexer=indexer)
    return admin, pool, workspace, ent.id, client, store, indexer, service, prefix


async def wait_for(
    pool, document_id: UUID, field: str, wanted: set[str], timeout: float
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        d = await repo.get_document(pool, document_id)
        assert d is not None
        if getattr(d, field) in wanted:
            return d
        await asyncio.sleep(0.5)
    raise SystemExit(f"timed out waiting for {field} in {wanted}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("filing", type=Path)
    ap.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    ap.add_argument("--entity", default="Apple Inc.")
    ap.add_argument(
        "--keep",
        action="store_true",
        help="do not drop the schema/collections (debugging)",
    )
    args = ap.parse_args()

    questions = [
        json.loads(l)
        for l in args.questions.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    base = get_settings()
    if not base.qdrant_url:
        print(
            "note: QDRANT_URL is not set, so embedded Qdrant is used (exact search, no HNSW)"
        )
    upload_dir = HERE / "results" / "uploads"
    settings = base.model_copy(
        update={"upload_dir": str(upload_dir), "index_on_startup": False}
    )
    schema = f"evalr_{uuid4().hex[:10]}"
    admin = pool = client = indexer = None
    try:
        (
            admin,
            pool,
            ws,
            ent_id,
            client,
            _store,
            indexer,
            service,
            _prefix,
        ) = await build(settings, args.filing, args.entity, schema)
        pending = [args.filing.read_bytes()]

        async def read_once(_size: int) -> bytes:
            return pending.pop() if pending else b""

        t = time.perf_counter()
        result = await service.upload(
            workspace_id=ws, entity_id=ent_id, filename=args.filing.name, read=read_once
        )
        doc = await wait_for(
            pool,
            result.document.id,
            "status",
            {"parsed", "needs_review", "needs_ocr", "failed"},
            900,
        )
        print(
            f"parsed in {time.perf_counter() - t:.0f}s: status={doc.status} ({doc.status_detail or 'clean'})"
        )
        n_chunks = await repo.count_chunks(pool, doc.id)
        print(f"chunks: {n_chunks} | blocks: {await repo.count_blocks(pool, doc.id)}")
        t = time.perf_counter()
        doc = await wait_for(pool, doc.id, "index_status", {"indexed", "failed"}, 1800)
        print(
            f"indexed in {time.perf_counter() - t:.0f}s: {doc.index_status} ({doc.index_detail or 'ok'}) with {doc.indexed_model}"
        )
        if doc.index_status != "indexed":
            raise SystemExit("indexing failed: cannot evaluate vector search")

        methods = {
            "full-text (AND, as built)": lambda q: repo.search_chunks_text(
                pool, ws, q, limit=FETCH
            ),
            "full-text (OR of terms)": lambda q: repo.search_chunks_text(
                pool, ws, or_query(q), limit=FETCH
            ),
            "vector": lambda q: indexer.search(ws, q, limit=FETCH),
        }
        outcomes: dict[str, list[Outcome]] = {m: [] for m in methods}
        for q in questions:
            for name, search in methods.items():
                outcomes[name].append(await run_method(name, q, search, pool))

        rows = [summarise(name, outs) for name, outs in outcomes.items()]
        print_table(rows)

        print("\nmisses at top-5 (question | method -> rank of first hit):")
        for i, q in enumerate(questions):
            ranks = {m: outcomes[m][i].first_rank for m in methods}
            if any(r is None or r > 5 for r in ranks.values()):
                short = {
                    m.split(" ")[0]
                    + ("-or" if "OR" in m else "-and" if "AND" in m else ""): r
                    for m, r in ranks.items()
                }
                print(f"  {q['id']:5s} {q['question'][:62]:62s} {short}")

        spent = await pool.fetchval(
            "select coalesce(sum(cost_usd), 0) from traces where kind = 'embed'"
        )
        calls = await pool.fetchval("select count(*) from traces where kind = 'embed'")
        print(f"\nembedding spend this run: ${float(spent):.5f} over {calls} calls")

        out_dir = HERE / "results"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"{datetime.now(tz=UTC):%Y%m%d-%H%M%S}.json"
        out.write_text(
            json.dumps(
                {
                    "filing": args.filing.name,
                    "chunks": n_chunks,
                    "model": settings.embedding_model,
                    "summary": rows,
                    "per_question": {
                        m: [vars(o) for o in outs] for m, outs in outcomes.items()
                    },
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"saved {out.relative_to(HERE.parents[1])}")
    finally:
        if not args.keep:
            if indexer is not None:
                await indexer.close()
            if client is not None:
                await client.aclose()
            if pool is not None:
                await pool.close()
            if admin is not None:
                await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                await admin.close()
            await cleanup_collections(settings)


async def cleanup_collections(settings: Settings) -> None:
    from qdrant_client import AsyncQdrantClient

    if not settings.qdrant_url:
        return
    c = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    try:
        for col in (await c.get_collections()).collections:
            if col.name.startswith("evalr_"):
                await c.delete_collection(col.name)
    finally:
        await c.close()


if __name__ == "__main__":
    asyncio.run(main())
