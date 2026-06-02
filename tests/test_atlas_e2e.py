import json
import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    import atlas_store, app as app_module
    monkeypatch.setattr(atlas_store, "CONCEPTS_PATH", tmp_path / "atlas-concepts.json")
    (tmp_path / "atlas-concepts.json").write_text(json.dumps({"version": "t", "concepts": [
        {"id": "light-therapy", "label": "Light Therapy", "aliases": ["syntonics"],
         "summary": "light frequencies for vision", "namespaces": [], "cluster": "vision",
         "parent": "vision", "coords": {"x": 0.3, "y": 0.4}, "neighbors": ["detox"],
         "links": [{"type": "video", "source": "youtube", "url": "https://y.t/x", "title": "Light"}],
         "status": "live"},
        {"id": "detox", "label": "Detox", "aliases": [], "summary": "elimination",
         "namespaces": [], "cluster": "foundations", "parent": "foundations",
         "coords": {"x": 0.7, "y": 0.6}, "neighbors": ["light-therapy"], "links": [],
         "status": "live"}]}))
    import threading, werkzeug.serving
    app_module.app.config["TESTING"] = True
    srv = werkzeug.serving.make_server("127.0.0.1", 0, app_module.app)
    port = srv.socket.getsockname()[1]
    t = threading.Thread(target=srv.serve_forever); t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown(); t.join()


def test_atlas_views_selection_and_ask(live_server):
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page()
        pg.goto(live_server + "/atlas")
        pg.wait_for_selector(".rm-node")
        # select on map
        pg.click('.rm-node[data-id="light-therapy"]')
        assert "Light Therapy" in pg.inner_text(".rm-drawer")
        # switch to A–Z, selection preserved
        pg.click('.rm-atlas__tab[data-view="az"]')
        assert pg.query_selector(".rm-list--sel") is not None
        # ask -> highlight + answer
        pg.fill(".rm-chat__input", "what light helps vision?")
        pg.press(".rm-chat__input", "Enter")
        pg.wait_for_selector(".rm-term")
        assert "Atlas:" in pg.inner_text(".rm-chat__answer")
        b.close()
