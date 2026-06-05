#!/usr/bin/env python3
"""Re-parse the cached GrooveKart raw HTML to recover the FULL ingredient panel
(the green-plus <p> bullets) that the original scrape dropped, then stage a
re-judge of the products currently flagged gk_stale against the COMPLETE GK data.

Read-only w.r.t. products.json. Writes per-product re-diff inputs to
data/redif-input/<slug>.json + prints coverage. The LLM re-judge + the
products.json gk_stale update happen in later steps."""
import json
import os
import re
import html as H
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.expanduser("~/Downloads/remedymatch-scrape/raw")
PRODUCTS = os.path.join(ROOT, "data", "products.json")
INDIR = os.path.join(ROOT, "data", "redif-input")


def _norm_title(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _slug_from_file(path):
    """raw file '...__236-macular-wellness-crocin.html' -> 'macular-wellness-crocin'."""
    base = os.path.basename(path)[:-5]  # drop .html
    last = base.split("__")[-1]
    return re.sub(r"^\d+-", "", last)


def _h1(html):
    """The product-name h1 (the first h1 with real text; the page's first h1 is <br>)."""
    for m in re.findall(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL):
        t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m)).strip()
        if t:
            return t
    return ""


def _green_bullets(html):
    """Ingredient panel = <p> bullets carrying the green-plus-sign image."""
    out = []
    for p in re.findall(r"<p\b[^>]*>(.*?)</p>", html, re.DOTALL):
        if "green" in p.lower() and "plus" in p.lower():
            t = re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", p))).strip()
            if t and len(t) > 1:
                out.append(t)
    return out


def main():
    # slug + title -> full GK ingredient bullets (from the complete raw HTML)
    by_slug, by_title = {}, {}
    for f in glob.glob(os.path.join(RAW, "*.html")):
        html = open(f, errors="ignore").read()
        bullets = _green_bullets(html)
        if not bullets:
            continue
        rec = {"bullets": bullets}
        by_slug[_slug_from_file(f)] = rec
        t = _h1(html)
        if t:
            by_title[_norm_title(t)] = rec
    print(f"parsed {len(by_slug)} GK pages with a green-plus ingredient panel")

    doc = json.load(open(PRODUCTS))
    products = doc.get("products", {})
    os.makedirs(INDIR, exist_ok=True)

    stale = [(s, p) for s, p in products.items() if p.get("gk_stale")]
    matched = 0
    unmatched = []
    for slug, p in stale:
        gk = by_slug.get(slug) or by_title.get(_norm_title(p.get("pinecone_title") or p.get("name")))
        if not gk:
            unmatched.append(slug)
            continue
        matched += 1
        rec = {
            "slug": slug,
            "name": p.get("name"),
            "authoritative": [i.get("name") for i in (p.get("ingredients") or []) if i.get("name")],
            "gk_bullets": gk["bullets"],
        }
        json.dump(rec, open(os.path.join(INDIR, f"{slug}.json"), "w"), indent=2)

    print(f"stale flagged: {len(stale)}  matched to a re-parsed GK page: {matched}  unmatched: {len(unmatched)}")
    if unmatched:
        print("UNMATCHED (no green-plus panel found by title):", ", ".join(unmatched))


if __name__ == "__main__":
    main()
