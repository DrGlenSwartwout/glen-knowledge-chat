"""Multi-source farm ingest: source registry + cross-source dedupe wiring.

Pure/no-network: fake source callables feed ingest() with apply=False so the
registry ordering, the dedupe collapse, and the summary shape are all proven
without touching the network or the DB."""
from scrapers.farm_finder.models import NormalizedFarmRow
from scrapers.farm_finder import ingest as ingest_mod
from scrapers.farm_finder.sources import get_sources


def _farm(name, website=None, source_org="X", source_url=None):
    return NormalizedFarmRow(
        name=name, website=website, source_org=source_org,
        source_url=source_url or ("u/" + name),
    )


def test_registry_lists_foodforhumans_first():
    # Dedupe keeps the FIRST occurrence, so the incumbent richest source leads.
    names = [n for (n, _fn) in get_sources()]
    assert names[0] == "foodforhumans"


def test_registry_only_filter():
    assert [n for (n, _) in get_sources(only=["foodforhumans"])] == ["foodforhumans"]
    assert get_sources(only=["nonexistent"]) == []


def test_realmilk_excluded_from_weekly_run():
    # realmilk is a ~12h crawl (10s robots delay) -> its own monthly lane, NEVER
    # inline in the weekly run_all pass. Guard against a regression that adds it.
    from scrapers.farm_finder.sources import WEEKLY_SOURCES, SLOW_SOURCES
    assert "realmilk" not in WEEKLY_SOURCES
    assert "realmilk" in SLOW_SOURCES
    assert "usda" in WEEKLY_SOURCES


def test_ingest_dedupes_across_sources_dry_run():
    src_a = ("a", lambda limit=None, sleep=0: [
        _farm("Green Acres", website="https://greenacres.com", source_org="A"),
    ])
    src_b = ("b", lambda limit=None, sleep=0: [
        _farm("Green Acres Farm", website="http://www.greenacres.com", source_org="B"),
        _farm("Blue Barn", website="https://bluebarn.com", source_org="B"),
    ])
    summary = ingest_mod.ingest(
        sources=[src_a, src_b], apply=False, log=lambda *_: None,
    )
    assert summary["scraped"] == 3
    assert summary["deduped"] == 2        # greenacres collapsed across A+B
    assert summary["mapped"] == 2
    assert summary["written"] == 0
    assert summary["applied"] is False
    assert summary["per_source"] == {"a": 1, "b": 2}


def test_ingest_first_source_wins_dedupe():
    # Trust order == source order: the row from the earlier source survives.
    src_a = ("a", lambda limit=None, sleep=0: [
        _farm("Dup", website="https://dup.com", source_org="TRUSTED"),
    ])
    src_b = ("b", lambda limit=None, sleep=0: [
        _farm("Dup", website="https://dup.com", source_org="OTHER"),
    ])
    # Mapping drops source_org, so assert on the deduped rows directly instead.
    from scrapers.farm_finder.dedupe import dedupe_farms
    a_rows = src_a[1]()
    b_rows = src_b[1]()
    deduped = dedupe_farms(a_rows + b_rows, log=lambda *_: None)
    assert len(deduped) == 1
    assert deduped[0].source_org == "TRUSTED"
