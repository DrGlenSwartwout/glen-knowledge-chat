import sqlite3, pytest, app as app_mod
from dashboard import product_ratings as pr, fullscript as fs


@pytest.fixture
def cx_db(monkeypatch, tmp_path):
    db = str(tmp_path / "b.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    pr.init_tables(cx); fs.init_tables(cx)
    # one confirmed-green product keyed fullscript::slug-green
    pr.record_screen(cx, "fullscript::slug-green", brand="B", product_name="G",
                     other_ingredients_raw="cellulose", other_ingredients_parsed=["cellulose"],
                     screen={"color": "green", "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"})
    # green must pass through ai_draft before confirm
    pr.set_tier2(cx, "fullscript::slug-green", None, "{}")
    pr.confirm(cx, "fullscript::slug-green")
    cx.commit(); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    return db


def _enrich(groups):
    # drive just the enrichment helper the task adds (see impl): given groups of
    # products with product_slug, attach purity when flag on.
    return app_mod._enrich_fullscript_purity(groups)


def test_flag_off_adds_no_purity(monkeypatch, cx_db):
    monkeypatch.setattr(app_mod, "_purity_badges_enabled", lambda: False)
    groups = [{"products": [{"product_slug": "slug-green"}]}]
    _enrich(groups)
    assert "purity" not in groups[0]["products"][0]


def test_flag_on_adds_confirmed_color(monkeypatch, cx_db):
    monkeypatch.setattr(app_mod, "_purity_badges_enabled", lambda: True)
    groups = [{"products": [{"product_slug": "slug-green"},
                            {"product_slug": "slug-none"}]}]
    _enrich(groups)
    assert groups[0]["products"][0]["purity"] == {"color": "green"}   # confirmed
    assert "purity" not in groups[0]["products"][1]                    # no confirmed row -> no badge
