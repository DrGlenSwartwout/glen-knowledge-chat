from dashboard import ebook_catalog as cat


def test_pilot_entry_present_and_shaped():
    e = cat.get("healing-glaucoma-starter")
    assert e is not None
    assert e["title"] and e["site"] == "healingglaucoma.com"
    assert e["pdf"] == "starter.pdf" and e["audio"] == "starter.mp3"
    assert e["dir"] == "healing-glaucoma-starter"


def test_unknown_slug_is_none():
    assert cat.get("does-not-exist") is None


def test_all_contains_pilot():
    assert any(e["slug"] == "healing-glaucoma-starter" for e in cat.all())


def test_pilot_has_condition_field():
    e = cat.get("healing-glaucoma-starter")
    assert e["condition"] == "glaucoma"


REQUIRED_KEYS = {"slug", "title", "site", "dir", "pdf", "audio", "condition"}

ALL_SLUGS = {
    "healing-glaucoma-starter": {
        "site": "healingglaucoma.com", "condition": "glaucoma",
    },
    "macular-regeneration-starter": {
        "site": "macularegeneration.com", "condition": "macular",
    },
    "cataract-solutions-starter": {
        "site": "cataractlab.com", "condition": "cataract",
    },
    "dry-eye-relief-starter": {
        "site": "dryeyelab.com", "condition": "dry-eye",
    },
    "refreshing-vision-starter": {
        "site": "refreshingvision.com", "condition": "vision-improvement",
    },
}


def test_all_five_slugs_resolve_with_required_keys_and_expected_shape():
    for slug, expect in ALL_SLUGS.items():
        e = cat.get(slug)
        assert e is not None, f"missing catalog entry: {slug}"
        assert set(e.keys()) >= REQUIRED_KEYS
        assert e["slug"] == slug
        assert e["dir"] == slug
        assert e["pdf"] == "starter.pdf"
        assert e["audio"] == "starter.mp3"
        assert e["site"] == expect["site"]
        assert e["condition"] == expect["condition"]


def test_all_returns_all_five_slugs():
    slugs = {e["slug"] for e in cat.all()}
    assert slugs == set(ALL_SLUGS.keys())
