"""Phase 3: farm dedupe helper + weekly-cron wiring."""
from scrapers.farm_finder.models import NormalizedFarmRow
from scrapers.farm_finder.dedupe import dedupe_farms, dedupe_key, _domain


def _farm(name, website=None, lat=None, lng=None, source_url=None, source_org="X"):
    return NormalizedFarmRow(name=name, website=website, lat=lat, lng=lng,
                             source_url=source_url or ("u/" + name), source_org=source_org)


def test_domain_normalizes_scheme_and_www():
    assert _domain("https://www.Foo.com/x") == "foo.com"
    assert _domain("http://foo.com") == "foo.com"
    assert _domain(None) is None


def test_dedupe_collapses_same_website_across_sources():
    rows = [
        _farm("Green Acres", website="https://greenacres.com/", source_org="A"),
        _farm("Green Acres Farm", website="http://www.greenacres.com", source_org="B"),
    ]
    out = dedupe_farms(rows, log=lambda *_: None)
    assert len(out) == 1
    assert out[0].source_org == "A"  # first (most-trusted source) kept


def test_dedupe_collapses_same_name_and_coords_when_no_website():
    rows = [
        _farm("Sunny Meadow", lat=40.1234, lng=-83.5678, source_org="A"),
        _farm("Sunny  Meadow", lat=40.1240, lng=-83.5681, source_org="B"),  # ~same spot
    ]
    out = dedupe_farms(rows, log=lambda *_: None)
    assert len(out) == 1


def test_dedupe_keeps_distinct_farms():
    rows = [
        _farm("Farm A", website="https://a.com"),
        _farm("Farm B", website="https://b.com"),
        _farm("No Site", lat=1.0, lng=2.0),
    ]
    assert len(dedupe_farms(rows, log=lambda *_: None)) == 3


def test_no_website_no_coords_stays_distinct():
    a, b = _farm("Mystery", source_url="u/1"), _farm("Mystery", source_url="u/2")
    assert dedupe_key(a) != dedupe_key(b)


def test_farm_adapter_wired_into_weekly_cron():
    # The weekly run_all cron must include the farm adapter so farms re-crawl
    # alongside practitioners. (Importing run_all does not hit the DB.)
    from scrapers.practitioner_finder.run_all import ADAPTERS
    names = [n for (n, _fn) in ADAPTERS]
    assert "foodforhumans_farms" in names
