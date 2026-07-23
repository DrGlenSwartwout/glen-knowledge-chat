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
