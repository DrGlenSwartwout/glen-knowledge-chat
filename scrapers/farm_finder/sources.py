"""Registry of Farm Finder source adapters.

Each source exposes a scrape callable with the uniform signature
`scrape(limit=None, sleep=<polite default>) -> list[NormalizedFarmRow]`. The
multi-source ingest (ingest.py) crawls every registered source, concatenates
the rows, runs cross-source de-dup (dedupe.py), then maps + upserts.

ORDER IS PRECEDENCE. dedupe_farms() keeps the FIRST occurrence of a duplicate
farm, so sources are listed most-trusted / richest-data first. Food for Humans
stays first (the incumbent, hand-validated, richest practice/product badges).

Adding a source = write scrapers/farm_finder/<name>.py exposing scrape(), then
append one line to _SOURCES below. Scrape callables are thin lazy-import
wrappers so importing this registry never pulls a network client until a source
actually runs (mirrors run_all.py's adapter wrappers).
"""
from typing import Callable, Optional

from scrapers.farm_finder.models import NormalizedFarmRow

FarmScrape = Callable[..., list[NormalizedFarmRow]]


def _foodforhumans(limit=None, sleep=0.5) -> list[NormalizedFarmRow]:
    from scrapers.farm_finder.foodforhumans import scrape
    return scrape(limit=limit, sleep=sleep)


def _usda(limit=None, sleep=1.0) -> list[NormalizedFarmRow]:
    from scrapers.farm_finder.usda import scrape
    return scrape(limit=limit, sleep=sleep)


# Most-trusted first — see module docstring (dedupe keeps first occurrence).
_SOURCES: list[tuple[str, FarmScrape]] = [
    ("foodforhumans", _foodforhumans),
    ("usda", _usda),            # USDA Local Food Directories (keyless bulk export)
    # ("realmilk", _realmilk),  # A Campaign for Real Milk (wpbdp sitemap crawl)
    # ("eatwild", _eatwild),    # EatWild state pages (needs geocode fallback)
]


def get_sources(only: Optional[list[str]] = None) -> list[tuple[str, FarmScrape]]:
    """Return registered (name, scrape) sources, optionally filtered to `only`.

    `only` is a list of source names; unknown names are silently ignored (so an
    --only typo yields an empty run rather than a crash — the caller reports 0)."""
    if only:
        wanted = set(only)
        return [(n, fn) for (n, fn) in _SOURCES if n in wanted]
    return list(_SOURCES)
