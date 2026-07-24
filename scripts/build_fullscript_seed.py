#!/usr/bin/env python3
"""Build data/fullscript_seed.json from Fullscript's PUBLIC open catalog.

Run BY HAND, never by the app. The app makes no runtime calls to Fullscript.

The endpoint backing fullscript.com/catalog needs no authentication, but it is
undocumented and only permits allowlisted operations (arbitrary GraphQL returns
HTTP 400). It expects `variables` as a JSON-ENCODED STRING, not an object.

Usage -- CAPTURE BOTH STREAMS. The dropped-product diagnostics land on
stderr, and a bare `> data/fullscript_seed.json` redirect leaves them
scrolling past in the terminal where they are easy to miss entirely. Redirect
stderr to a log too (or tee it) so you can actually read it afterward:

    python3 scripts/build_fullscript_seed.py > data/fullscript_seed.json \
        2> /tmp/fullscript_seed_dropped.log

or, to watch it live and keep a copy:

    python3 scripts/build_fullscript_seed.py \
        > data/fullscript_seed.json \
        2> >(tee /tmp/fullscript_seed_dropped.log >&2)

Then review by hand, in this order:
  1. `best_ff` mappings are guesses and must be corrected by Glen before the
     seed is committed.
  2. Read the dropped-product log end to end. `_name_matches_term`'s
     substring filter is a blunt backstop with a known class of false
     negative: chemical-form and spelling variance. Worked example that has
     already happened once: "Pure Taurine 500mg" was correctly dropped for
     the search term "magnesium taurate" because "taurate" is not a substring
     of "taurine" -- even though taurine is arguably clinically relevant to
     the same focus area. Expect more of these (ascorbate/ascorbic,
     methylcobalamin/B12, and similar pairs). The generator will not catch
     these for you; add anything clinically relevant back to the seed by
     hand after reading the log.
  3. Read the KEPT list too, not just the dropped one. The filter keeps a
     product if ANY single token of the search term appears in its name, so
     a multi-word term admits products that matched only its generic word
     ("magnesium" out of "magnesium taurate"). That is the same kind of
     off-target hit the filter exists to block, one word weaker. Surviving
     the filter is not clinical endorsement -- every kept product still
     needs your eye before a client ever sees it.

focus_area_items (the E4L scan-code -> focus-area map that drives the scan
matcher) is NOT fetched from Fullscript -- it is copied from data/prl_seed.json,
the sibling PRL channel's seed, which already carries the full E4L focus-area
taxonomy. See _load_focus_area_items().
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://fullscript.com/api/fs-graphql"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

TYPEAHEAD = """
  query TypeaheadSearchV2_Shared_Query($query: String, $filters: SearchFilterObject) {
    viewer {
      typeaheadSearchV2(query: $query, filters: $filters, useSkuIdentifier: true) {
        entityType
        data { id name entityType brandName productSlug }
      }
    }
  }
"""

# focus_area_id -> (focus area name, [search terms], {product name: best Functional
# Formulations equivalent}). Mirrors the focus areas PRL already covers so the two
# channels reach parity. Extend as Glen confirms mappings.
FOCUS_AREAS = {
    9: ("Nervous System", ["magnesium taurate", "l-theanine"], {}),
}


def _tokens(term):
    """Meaningful (>=4 char) whitespace-split tokens of a search term, lowercased."""
    return [t for t in term.lower().split() if len(t) >= 4]


def _name_matches_term(name, term):
    """Keep a hit only if a meaningful token from the search term appears in the
    product name (case-insensitive substring). Fullscript's typeahead is fuzzy
    and surfaces clinically off-target hits (a homeopathic topical gel, a
    thyroid-support capsule) for nervous-system searches; this is a blunt but
    predictable backstop the owner can extend by hand.

    Two known limitations here, both accepted trade-offs rather than bugs to
    fix in this function:

    1. OR-across-tokens, not AND. A multi-word term matches if ANY ONE token
       is present, not all of them -- a product whose name contains only the
       generic word "magnesium" still passes the term "magnesium taurate".
       That is the same class of off-target risk this filter exists to
       prevent, just one word weaker. A survivor of this filter still
       warrants a human glance, especially when the only matching token is a
       generic nutrient name rather than a specific brand/form word.
    2. Chemical-form and spelling variance is invisible to a plain substring
       check and produces a silent false NEGATIVE (a correct-looking drop
       that is still a miss). Worked example that has already happened:
       "Pure Taurine 500mg" is dropped for the term "magnesium taurate"
       because "taurate" is not a substring of "taurine", even though
       taurine is arguably clinically relevant to the same focus area.
       Expect the same for ascorbate/ascorbic, methylcobalamin/B12, and
       similar pairs. Nothing here detects this class -- read the
       dropped-product stderr log and add anything clinically relevant back
       to the seed by hand.

    Callers must not pass a term with no token of 4+ characters -- see
    _check_term_tokens, which guards against that silently filtering out
    every hit for the term with no error at all."""
    toks = _tokens(term)
    if not toks:
        return False
    lname = name.lower()
    return any(t in lname for t in toks)


def _check_term_tokens(term):
    """Guard against a silent zero-result trap: if a search term has no token
    of 4+ characters, _tokens(term) is [] and _name_matches_term returns False
    for EVERY hit, so the term silently contributes zero products to the seed
    with no error at all. This is realistic, not hypothetical -- "B12", "DHA",
    "EPA", and "CoQ10" are all search terms a clinician might reasonably add,
    and the module docstring invites exactly that.

    Call this on every term BEFORE searching (main() does this for all terms
    up front, before any network call, so a bad term is caught without
    burning rate-limit budget on the terms before it).

    Raises SystemExit naming the offending term. The fix is for the curator to
    either lengthen the term (e.g. "vitamin B12" instead of "B12") so it has a
    4+ character token the filter can use, or to add the product to the seed
    by hand instead of relying on this generator for that term."""
    if not _tokens(term):
        raise SystemExit(
            f"search term {term!r} has no token of 4+ characters -- "
            "_name_matches_term's relevance filter cannot evaluate it and "
            "would silently keep zero hits for this term, with no error. "
            "Lengthen the term (e.g. 'vitamin B12' instead of 'B12') or add "
            "the product to the seed by hand instead.")


def _load_focus_area_items():
    """Copy focus_area_items (E4L scan item code -> focus area) from the sibling
    PRL channel's seed, which carries the same E4L focus-area taxonomy. Only
    copies entries for focus areas this generator's FOCUS_AREAS map covers, and
    only after confirming the id-to-name mapping agrees on both sides -- refuses
    (raises) rather than silently copying a mismatched mapping."""
    prl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "data", "prl_seed.json")
    with open(prl_path, "r", encoding="utf-8") as f:
        prl = json.load(f)
    prl_names = {}
    for fp in prl.get("focus_area_products", []):
        prl_names.setdefault(fp["focus_area_id"], set()).add(fp.get("focus_area_name"))
    items = []
    for fa_id, (fa_name, _terms, _ff_map) in sorted(FOCUS_AREAS.items()):
        prl_fa_names = prl_names.get(fa_id)
        if not prl_fa_names:
            raise SystemExit(
                f"focus_area_id {fa_id} ({fa_name!r}) not present in "
                f"{prl_path}; refusing to copy focus_area_items")
        if prl_fa_names != {fa_name}:
            raise SystemExit(
                f"focus_area_id {fa_id} name mismatch: this generator has "
                f"{fa_name!r}, prl_seed.json has {sorted(prl_fa_names)!r}; "
                "refusing to copy a mismatched focus_area_items mapping")
        for fi in prl.get("focus_area_items", []):
            if fi["focus_area_id"] == fa_id:
                items.append({"focus_area_id": fa_id, "item_code": fi["item_code"]})
    return items


def search(term):
    body = json.dumps({
        "query": TYPEAHEAD,
        "variables": json.dumps({
            "query": term,
            "filters": {"list": ["PRODUCTS", "BRANDS", "INGREDIENTS"]},
        }),
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json", "User-Agent": UA,
                 "Origin": "https://fullscript.com",
                 "Referer": "https://fullscript.com/catalog"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as e:
        note = ("this is a rate limit -- wait and retry BY HAND later; do "
                 "not add an automatic retry here, that would violate the "
                 "polite, unhurried-client constraint this script keeps "
                 "(the 1-second sleep between terms)") if e.code == 429 else \
            "the endpoint or network may be having trouble; try again later"
        raise SystemExit(
            f"HTTP {e.code} searching for {term!r}: {e.reason}. {note}."
        ) from e
    if payload.get("errors"):
        raise SystemExit(f"catalog error for {term!r}: {payload['errors']}")
    groups = payload["data"]["viewer"]["typeaheadSearchV2"]
    for g in groups:
        if g.get("entityType") == "Product":
            return g.get("data") or []
    return []


def main():
    # Validate every term up front, before any network call, so a term with
    # no usable token fails immediately instead of burning rate-limit budget
    # on the terms that come before it. See _check_term_tokens.
    for _fa_id, (_fa_name, terms, _ff_map) in sorted(FOCUS_AREAS.items()):
        for term in terms:
            _check_term_tokens(term)

    products, fa_products = {}, []
    for fa_id, (fa_name, terms, ff_map) in sorted(FOCUS_AREAS.items()):
        rank = 0
        for term in terms:
            for hit in search(term):
                name = hit.get("name")
                if not name or name in products:
                    continue
                if not _name_matches_term(name, term):
                    print(f"dropped {name!r} (brand={hit.get('brandName')!r}): "
                          f"no meaningful token from {term!r} found in name",
                          file=sys.stderr)
                    continue
                products[name] = {
                    "name": name,
                    "brand": hit.get("brandName"),
                    "external_id": hit.get("id"),
                    "product_slug": hit.get("productSlug"),
                    "url": None,
                    "focus_tags": [fa_name],
                    "product_type": "supplement",
                    "best_ff": ff_map.get(name),
                    "relation": "substitute" if ff_map.get(name) else None,
                    "ff_alts": [],
                    "source": "seed",
                    "active": 1,
                }
                fa_products.append({"focus_area_id": fa_id, "focus_area_name": fa_name,
                                    "fs_product_name": name, "rank": rank})
                rank += 1
            time.sleep(1.0)  # browsing volume, deliberately unhurried
    json.dump({"products": sorted(products.values(), key=lambda p: p["name"]),
               "focus_area_products": fa_products,
               "focus_area_items": _load_focus_area_items()},
              sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
    print(f"{len(products)} products across {len(FOCUS_AREAS)} focus areas",
          file=sys.stderr)


if __name__ == "__main__":
    main()
