"""atlas_build: seed-input -> split into live (auto) + pending (queue), validated."""
import hashlib
import json

import atlas_build
import atlas_store


def _fake_embed(texts, chunk=200):
    """Deterministic 6-dim vectors so PCA/cosine are stable without the OpenAI API."""
    out = []
    for t in texts:
        h = hashlib.md5(t.encode("utf-8")).digest()
        out.append([b / 255.0 for b in h[:6]])
    return out


def _seed():
    return {"version": "t", "concepts": [
        {"id": "lutein", "label": "Lutein", "aliases": [], "summary": "macular carotenoid",
         "cluster": "eye-health", "parent": "eye-health", "namespaces": ["ingredients"],
         "disposition": "auto", "links": []},
        {"id": "iop-syntropy", "label": "IOP Syntropy", "aliases": [], "summary": "pressure formula",
         "cluster": "eye-health", "parent": "eye-health", "namespaces": ["specific-formulations"],
         "disposition": "auto", "links": [{"type": "product", "source": "rm", "url": "u", "title": "IOP"}]},
        {"id": "drusen", "label": "Drusen", "aliases": [], "summary": "retinal deposits",
         "cluster": "retina-macula", "parent": "retina-macula", "namespaces": ["case-studies"],
         "disposition": "queue", "links": []},
    ]}


def test_build_splits_by_disposition(tmp_path, monkeypatch):
    seed = tmp_path / "atlas-seed-input.json"
    seed.write_text(json.dumps(_seed()))
    live_p, pend_p = tmp_path / "live.json", tmp_path / "pending.json"
    monkeypatch.setattr(atlas_build, "SEED_PATH", seed)
    monkeypatch.setattr(atlas_build, "CONCEPTS_PATH", live_p)
    monkeypatch.setattr(atlas_build, "PENDING_PATH", pend_p)
    monkeypatch.setattr(atlas_build, "batch_embed", _fake_embed)
    monkeypatch.setattr(atlas_build, "load_videos", lambda: [])

    assert atlas_build.build() == 0
    live = json.loads(live_p.read_text())["concepts"]
    pend = json.loads(pend_p.read_text())["concepts"]

    assert {c["id"] for c in live} == {"lutein", "iop-syntropy"}      # auto -> live
    assert {c["id"] for c in pend} == {"drusen"}                       # queue -> pending
    assert all(c["status"] == "live" for c in live)
    assert all(c["status"] == "pending" for c in pend)
    # every emitted record passes the store's validator (coords in [0,1], required fields)
    for c in live + pend:
        ok, err = atlas_store.validate_concept(c)
        assert ok, err
    # product link preserved on the formulation
    iop = next(c for c in live if c["id"] == "iop-syntropy")
    assert any(l["type"] == "product" for l in iop["links"])
