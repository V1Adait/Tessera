"""The real BM25 encoder (fastembed). These run the actual model, which is a one-off download of a
few small files, cached in a folder; if it cannot be loaded here (no network on a first run) the
tests skip instead of failing."""

import pytest

from backend.app.retrieval.sparse import Bm25Encoder, SparseError, SparseVec


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    # one cache for the whole file; a cold download is a couple of seconds
    return str(tmp_path_factory.mktemp("bm25"))


@pytest.fixture
async def encoder(cache_dir):
    enc = Bm25Encoder(cache_dir=cache_dir)
    try:
        await enc.encode_query("warm up")
    except SparseError:
        pytest.skip("the BM25 model could not be loaded (offline on first run?)")
    return enc


def ids(vector: SparseVec) -> set[int]:
    return set(vector.indices)


async def test_a_chunk_becomes_terms_with_weights(encoder):
    [vector] = await encoder.encode_documents(
        ["The Company repurchased 402 million shares of its common stock."]
    )

    assert not vector.empty
    assert len(vector.indices) == len(set(vector.indices))  # one entry per term
    assert len(vector.indices) == len(vector.values)
    assert all(v > 0 for v in vector.values)


async def test_the_same_word_in_different_forms_is_one_term(encoder):
    [a] = await encoder.encode_documents(["repurchase"])
    [b] = await encoder.encode_documents(["repurchased"])

    assert ids(a) == ids(b)


async def test_stopwords_and_blank_text_give_an_empty_vector(encoder):
    docs = await encoder.encode_documents(["the of and", "", "   "])

    assert all(d.empty for d in docs)
    assert (await encoder.encode_query("what is the")).empty


async def test_a_query_weighs_every_term_equally(encoder):
    query = await encoder.encode_query("net sales revenue")

    assert query.values and set(query.values) == {1.0}


async def test_a_chunk_and_a_question_share_term_ids(encoder):
    [doc] = await encoder.encode_documents(
        ["Net sales increased because of iPhone demand."]
    )
    query = await encoder.encode_query("how did net sales change")

    assert ids(doc) & ids(query)


async def test_numbers_are_split_at_separators_so_figures_still_match_themselves(
    encoder,
):
    [doc] = await encoder.encode_documents(["Total net sales were $416,161 million."])
    query = await encoder.encode_query("$416,161")

    assert ids(query) <= ids(doc)


async def test_order_is_kept_for_a_batch(encoder):
    texts = ["alpha beta gamma", "delta epsilon", "zeta"]
    batch = await encoder.encode_documents(texts)
    singles = [(await encoder.encode_documents([t]))[0] for t in texts]

    assert [ids(v) for v in batch] == [ids(v) for v in singles]


async def test_an_empty_batch_loads_nothing_and_returns_nothing(cache_dir):
    enc = Bm25Encoder(cache_dir=cache_dir)

    assert await enc.encode_documents([]) == []
    assert enc._model is None  # the model was not even loaded


async def test_a_model_that_cannot_load_is_a_plain_error_without_details(tmp_path):
    enc = Bm25Encoder(model="no-such/model-anywhere", cache_dir=str(tmp_path))

    with pytest.raises(SparseError) as caught:
        await enc.encode_query("revolver")

    assert caught.value.code == "unavailable"
    assert "no-such" not in caught.value.message


async def test_concurrent_calls_all_get_answers(encoder):
    import asyncio

    results = await asyncio.gather(
        *[encoder.encode_documents([f"revolver number {i}"]) for i in range(8)]
    )

    assert all(len(r) == 1 and not r[0].empty for r in results)
