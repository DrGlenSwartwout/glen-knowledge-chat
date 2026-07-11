import types

import app as appmod


def _match(id_, score, metadata):
    return types.SimpleNamespace(id=id_, score=score, metadata=metadata)


def test_ff_query_specific_formulations_normalizes_title_to_name(monkeypatch):
    """Pinecone's `specific-formulations` namespace stores the product name
    under `title`, not `name`. The downstream generator (dashboard/ff_matcher)
    reads metadata['name'] and drops candidates with no name, so the adapter
    must backfill name from title when name is absent."""
    monkeypatch.setattr(appmod, "embed", lambda text: [0.0], raising=False)
    monkeypatch.setattr(
        appmod._idx,
        "query",
        lambda **kw: types.SimpleNamespace(
            matches=[_match("x", 0.9, {"title": "Adrenal Restore Formula"})]
        ),
        raising=False,
    )

    results = appmod._ff_query_specific_formulations("adrenal", 5)

    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "Adrenal Restore Formula"
    assert results[0]["metadata"]["title"] == "Adrenal Restore Formula"
    assert results[0]["id"] == "x"
    assert results[0]["score"] == 0.9
