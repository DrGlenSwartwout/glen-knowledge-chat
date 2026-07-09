# tests/test_cert_portal_routes.py
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CERT_PORTAL_ENABLED", "true")
    import app as appmod
    # Hermetic sqlite: point LOG_DB at a tmp file so tests never touch the dev db,
    # then build the canonical auth/user tables there via the app's own init.
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    # Never send real email/contacts during tests.
    monkeypatch.setattr(appmod, "send_magic_link_email", lambda *a, **k: ("test", None))
    monkeypatch.setattr(appmod, "_send_inquiry_email", lambda *a, **k: True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _mint_cert_token(appmod, email):
    """Insert a cert_portal auth token directly and return the raw token."""
    import secrets
    from datetime import timedelta
    tok = secrets.token_urlsafe(16)
    now = appmod._now_utc()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute(
            "INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
            "VALUES (?,?,?,?,?)",
            (appmod._hash_token(tok), email, "cert_portal", now.isoformat(),
             (now + timedelta(minutes=appmod.AUTH_TOKEN_TTL_MIN)).isoformat()))
        cx.commit()
    return tok


def test_login_always_200(client):
    c, _ = client
    r = c.post("/cert/login", json={"email": "doc@x.com"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_auth_sets_cookie_and_redirects(client):
    c, appmod = client
    tok = _mint_cert_token(appmod, "doc@x.com")
    # GET only confirms (mail scanners prefetch it); the POST signs in.
    assert c.get(f"/cert/auth/{tok}").status_code == 200
    r = c.post(f"/cert/auth/{tok}")
    assert r.status_code == 302
    assert "rm_cert_email" in r.headers.get("Set-Cookie", "")


def test_auth_rejects_bad_token(client):
    c, _ = client
    r = c.get("/cert/auth/not-a-real-token")
    assert r.status_code == 400


def test_portal_page_served_when_enabled(client):
    c, _ = client
    r = c.get("/cert")
    assert r.status_code == 200


def test_portal_404_when_flag_off(client, monkeypatch):
    c, appmod = client
    # The flag is read live (not at import), so force it off via the helper.
    monkeypatch.setattr(appmod, "_cert_portal_enabled", lambda: False)
    r = c.get("/cert")
    assert r.status_code == 404


def _auth_client(client):
    """Return a test client that already holds the rm_cert_email cookie."""
    c, appmod = client
    tok = _mint_cert_token(appmod, "doc@x.com")
    c.post(f"/cert/auth/{tok}")
    return c, appmod


def test_submit_requires_cookie(client):
    c, _ = client
    r = c.post("/api/cert/submit", json={"title": "x"})
    assert r.status_code == 401


def test_mine_requires_cookie(client):
    c, _ = client
    r = c.get("/api/cert/mine")
    assert r.status_code == 401


def test_submit_requires_permission(client, monkeypatch):
    c, appmod = _auth_client(client)
    monkeypatch_pp(appmod, monkeypatch)
    r = c.post("/api/cert/submit", json={
        "title": "My case", "description": "d", "url": "https://e.com/p",
        "formats": ["article"], "modules": [1], "permission": False})
    assert r.status_code == 400
    assert "permission" in r.get_json()["error"].lower()


def test_submit_requires_module_and_link(client, monkeypatch):
    c, appmod = _auth_client(client)
    monkeypatch_pp(appmod, monkeypatch)
    # no module
    r = c.post("/api/cert/submit", json={
        "title": "t", "url": "https://e.com/p", "formats": ["article"],
        "modules": [], "permission": True})
    assert r.status_code == 400
    # no url and no file
    r = c.post("/api/cert/submit", json={
        "title": "t", "url": "", "formats": ["article"],
        "modules": [1], "permission": True})
    assert r.status_code == 400


def test_submit_creates_row_and_mine_lists_it(client, monkeypatch):
    c, appmod = _auth_client(client)
    monkeypatch_pp(appmod, monkeypatch)
    r = c.post("/api/cert/submit", json={
        "title": "My case", "description": "what happened",
        "url": "https://e.com/p", "formats": ["article", "demo_video"],
        "modules": [1, 2], "topic_angle": "transformation", "permission": True})
    assert r.status_code == 200
    sid = r.get_json()["submission"]["id"]
    assert sid
    r2 = c.get("/api/cert/mine")
    body = r2.get_json()
    assert any(s["id"] == sid for s in body["submissions"])
    # progress rollup present; nothing approved yet so 0 covered
    assert body["progress"]["approved_count"] == 0


# Helper: monkeypatch the practitioner lookups used by submit/approve/publish.
# Takes pytest's monkeypatch fixture so the stub is torn down after each test.
def monkeypatch_pp(appmod, monkeypatch):
    from dashboard import practitioner_portal as pp
    monkeypatch.setattr(pp, "id_for_email", lambda email: "p-test")


def _console_key(appmod):
    return appmod.CONSOLE_SECRET or ""


def test_console_cert_page_served(client):
    c, _ = client
    r = c.get("/console/cert")
    assert r.status_code == 200


def test_review_list_console_gated(client):
    c, appmod = client
    r = c.get("/api/cert/review/list")  # no key
    if appmod.CONSOLE_SECRET:
        assert r.status_code == 401
    else:
        assert r.status_code == 200


def test_approve_syncs_modules_completed(client, monkeypatch):
    c, appmod = _auth_client(client)
    monkeypatch_pp(appmod, monkeypatch)
    # create a submission covering modules 1,2
    c.post("/api/cert/submit", json={
        "title": "t", "url": "https://e.com/p", "formats": ["article"],
        "modules": [1, 2], "permission": True})
    # capture upsert sync call
    calls = {}
    from dashboard import practitioner_portal as pp
    monkeypatch.setattr(pp, "upsert_cert_student",
                        lambda email, **kw: calls.update(kw) or ("pid", kw.get("modules_completed", 0)))
    # find the submission id
    sid = c.get("/api/cert/mine").get_json()["submissions"][0]["id"]
    key = _console_key(appmod)
    r = c.post("/api/cert/review/approve?key=" + key,
               json={"id": sid, "credited_modules": [1, 2]})
    assert r.status_code == 200
    assert calls.get("modules_completed") == 2  # 2 distinct modules covered


def test_approve_ignores_non_numeric_credited(client, monkeypatch):
    c, appmod = _auth_client(client)
    monkeypatch_pp(appmod, monkeypatch)
    c.post("/api/cert/submit", json={
        "title": "t", "url": "https://e.com/p", "formats": ["article"],
        "modules": [1, 2], "permission": True})
    from dashboard import practitioner_portal as pp
    monkeypatch.setattr(pp, "upsert_cert_student", lambda email, **kw: ("pid", 0))
    sid = c.get("/api/cert/mine").get_json()["submissions"][0]["id"]
    key = _console_key(appmod)
    # a junk value must not 500 the handler — it is skipped
    r = c.post("/api/cert/review/approve?key=" + key,
               json={"id": sid, "credited_modules": [1, "two", 2]})
    assert r.status_code == 200
    assert r.get_json()["modules_covered"] == 2


def test_publish_requires_approved_and_permission(client, monkeypatch):
    c, appmod = _auth_client(client)
    monkeypatch_pp(appmod, monkeypatch)
    c.post("/api/cert/submit", json={
        "title": "t", "url": "https://e.com/p", "formats": ["article"],
        "modules": [1], "permission": True})
    sid = c.get("/api/cert/mine").get_json()["submissions"][0]["id"]
    key = _console_key(appmod)
    # publish before approve → 400
    r = c.post("/api/cert/review/publish?key=" + key, json={"id": sid})
    assert r.status_code == 400
    # approve, then stub embed + pinecone, then publish
    c.post("/api/cert/review/approve?key=" + key,
           json={"id": sid, "credited_modules": [1]})
    monkeypatch.setattr(appmod, "embed", lambda text: [0.0] * 1536)
    captured = {}
    monkeypatch.setattr(appmod._idx, "upsert",
                        lambda **kw: captured.update(kw))
    from dashboard import practitioner_portal as pp
    monkeypatch.setattr(pp, "name_for_email", lambda email: "Dr Test")
    r = c.post("/api/cert/review/publish?key=" + key, json={"id": sid})
    assert r.status_code == 200
    assert captured.get("namespace") == "case-studies"
    assert captured["vectors"][0]["id"] == "cert-" + sid
