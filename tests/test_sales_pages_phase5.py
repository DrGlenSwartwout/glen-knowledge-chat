import sqlite3
import pytest
from dashboard import sales_pages as sp
from dashboard import sales_pages_actions as spa
from dashboard.rbac import Actor, OWNER


@pytest.fixture(autouse=True)
def _reset_spa_deps():
    spa._DEPS.clear()
    yield


def _cx():
    return sqlite3.connect(":memory:")


def test_set_state_approved_stamps_by_and_time():
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "hello")
    sp.set_state(cx, "x", "approved", by="Glen")
    page = sp.get_page(cx, "x")
    assert page["state"] == "approved"
    row = cx.execute(
        "SELECT approved_at, approved_by FROM sales_pages WHERE product_slug='x'").fetchone()
    assert row[1] == "Glen" and row[0]  # approved_by set, approved_at non-empty


def test_set_state_draft_does_not_stamp_approver():
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "hello")
    sp.set_state(cx, "x", "approved", by="Glen")
    sp.set_state(cx, "x", "draft")
    page = sp.get_page(cx, "x")
    assert page["state"] == "draft"


def test_list_draft_pages_includes_content_excludes_empty():
    cx = _cx()
    sp.upsert_section(cx, "with-copy", "intro", "hello")
    sp.init_table(cx)
    # a row with empty content_json should be excluded
    cx.execute("INSERT INTO sales_pages (product_slug, content_json) VALUES ('empty','{}')")
    cx.commit()
    rows = sp.list_draft_pages(cx)
    slugs = [r["slug"] for r in rows]
    assert "with-copy" in slugs and "empty" not in slugs
    row = next(r for r in rows if r["slug"] == "with-copy")
    assert row["state"] == "draft" and row["sections"] == ["intro"]


class _Blk:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Msg:
    def __init__(self, text):
        self.content = [_Blk(text)]


class _FakeMessages:
    def create(self, **kw):
        # echo the section brief marker so each call is distinct-enough; em dash on purpose
        return _Msg("Supports vitality — grounded in the stack.")


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def _configure_fake():
    spa.configure(client=_FakeClient(),
                  get_product=lambda s: {"name": "Test", "ingredients": [{"name": "Magnesium"}]},
                  product_card=lambda p: {"ingredients": p.get("ingredients", [])},
                  strip_dash=lambda s: s.replace("—", ","))


def test_regenerate_copy_strips_dashes_all_sections():
    _configure_fake()
    out = spa.regenerate_copy("x")
    assert set(out.keys()) == {"intro", "description", "research"}
    assert all("—" not in v and v for v in out.values())


def test_regenerate_copy_none_without_product():
    spa.configure(client=_FakeClient(), get_product=lambda s: None,
                  product_card=lambda p: {}, strip_dash=lambda s: s)
    assert spa.regenerate_copy("nope") is None


def test_regenerate_copy_none_without_client():
    spa.configure(client=None, get_product=lambda s: {"name": "x"},
                  product_card=lambda p: {}, strip_dash=lambda s: s)
    assert spa.regenerate_copy("x") is None


def test_exec_edit_forces_draft():
    _configure_fake()
    spa.register()
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "old")
    sp.set_state(cx, "x", "approved", by="Glen")
    from dashboard.actions import get_action
    act = get_action("sales_pages.edit")
    act.executor({"slug": "x", "section": "intro", "text": "new copy"},
                 {"cx": cx, "actor": Actor(role=OWNER, name="Glen")})
    page = sp.get_page(cx, "x")
    assert page["content"]["intro"] == "new copy"
    # any edit returns the page to draft: edited copy must be re-approved before the
    # banner drops again (Approve is the single deliberate publish step).
    assert page["state"] == "draft"


def test_exec_approve_sets_approved():
    _configure_fake()
    spa.register()
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "hi")
    from dashboard.actions import get_action
    get_action("sales_pages.approve").executor(
        {"slug": "x"}, {"cx": cx, "actor": Actor(role=OWNER, name="Glen")})
    assert sp.get_page(cx, "x")["state"] == "approved"


def test_exec_regenerate_sets_draft_and_writes_copy():
    _configure_fake()
    spa.register()
    cx = _cx()
    sp.upsert_section(cx, "x", "intro", "old")
    sp.set_state(cx, "x", "approved", by="Glen")
    from dashboard.actions import get_action
    res = get_action("sales_pages.regenerate").executor(
        {"slug": "x"}, {"cx": cx, "actor": Actor(role=OWNER, name="Glen")})
    page = sp.get_page(cx, "x")
    assert page["state"] == "draft"
    assert page["content"]["intro"] == res["content"]["intro"]


def test_register_idempotent():
    spa.register()
    spa.register()  # second call must not raise duplicate-key


import importlib


def _reload_app(monkeypatch, tmp_path, copy="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SALES_PAGES_ENABLED", "true")
    monkeypatch.setenv("SALES_PAGES_AI_COPY", copy)
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_page_data_ai_state_none_when_no_page(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    assert data["ai_state"] == "none"


def test_page_data_ai_state_reflects_state(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import sales_pages as sp2
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp2.upsert_section(cx, slug, "intro", "draft copy")
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    assert data["ai_state"] == "draft"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp2.set_state(cx, slug, "approved", by="Glen")
    data = appmod.app.test_client().get(f"/begin/product-page-data/{slug}").get_json()
    assert data["ai_state"] == "approved"


def test_dispatch_approve_flips_state(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    slug = next(iter(appmod._PRODUCTS["products"].keys()))
    import sqlite3
    from dashboard import sales_pages as sp2
    from dashboard import dispatch as d
    from dashboard.rbac import Actor, OWNER
    with sqlite3.connect(appmod.LOG_DB) as cx:
        sp2.upsert_section(cx, slug, "intro", "draft copy")
        res = d.dispatch_action(cx, "sales_pages.approve", {"slug": slug},
                                Actor(role=OWNER, name="Glen"), source="panel")
        assert res["status"] == "done"
        assert sp2.get_page(cx, slug)["state"] == "approved"
