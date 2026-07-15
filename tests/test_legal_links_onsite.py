import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_no_external_terms_links_in_static():
    hits = []
    for f in (ROOT / "static").glob("*"):
        if f.suffix in (".html", ".js") and f.is_file():
            if "remedymatch.com/info/terms-and-conditions" in f.read_text(errors="ignore"):
                hits.append(f.name)
    assert not hits, f"external T&C link still present in: {hits}"

def test_affiliate_clause_in_onsite_terms():
    assert "Referral and Affiliate Program" in (ROOT / "static/terms.html").read_text()
