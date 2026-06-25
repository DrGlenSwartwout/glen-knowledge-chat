from scripts.match_products_to_fmp import match_products

def test_match_exact_and_fuzzy_and_review():
    products = {
        "nerve-pulse": {"name": "Nerve Pulse"},
        "msm-syntropy": {"name": "MSM Synergy"},   # alias → FMP "MSM Syntropy"
        "mystery": {"name": "Zzz Unknown Tonic"},
        "already": {"name": "Foo", "fmp_id": "999"},
    }
    fmp_by_name = {  # built by _build_fmp_index in real use; here pre-normalized keys
        "nerve pulse": {"id_pk": "1104"},
        "msm synergy": {"id_pk": "606"},           # _norm maps syntropy->synergy on both sides
    }
    m = match_products(products, fmp_by_name)
    assert m["matched"]["nerve-pulse"] == "1104"
    assert m["matched"]["msm-syntropy"] == "606"
    assert "already" not in m["matched"]            # never overwrite existing fmp_id
    assert any(r["slug"] == "mystery" for r in m["review"])
