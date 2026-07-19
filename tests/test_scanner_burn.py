"""Emailed single-use links must survive a mail scanner's GET.

Corporate mail filters, link checkers and browser prefetch all issue GET on
every URL in an email, before the human ever clicks. When a GET consumed the
token, the recipient landed on "link already used". These tests pin the fix:
GET renders a confirm page and mutates nothing; the POST behind the button does
the work.
"""
import importlib
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


# auth_tokens.expires_at is NOT stored in one format. Each minter picks its own,
# and each validator parses the way its minter wrote. Mint the way prod does, or
# the token reads as expired for reasons that have nothing to do with this fix.
#   aware  -> _now_utc().isoformat()            e.g. "...+00:00"
#   naive  -> datetime.utcnow().isoformat()+"Z" e.g. "...Z"
_TZ_AWARE_PURPOSES = {"reorder", "biofield", "magic_link"}


def _mint(appmod, purpose, email="scan@example.com", minutes=60, extra=None):
    """Insert a live token of `purpose` straight into auth_tokens."""
    import secrets
    tok = secrets.token_urlsafe(32)
    if purpose in _TZ_AWARE_PURPOSES:
        now = datetime.now(timezone.utc)
        fmt = lambda d: d.isoformat()
    else:
        now = datetime.utcnow()
        fmt = lambda d: d.isoformat() + "Z"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute(
            "CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT PRIMARY KEY, "
            "email TEXT, purpose TEXT NOT NULL, extra TEXT, created_at TEXT NOT NULL, "
            "expires_at TEXT NOT NULL, consumed_at TEXT)")
        cx.execute(
            "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
            "VALUES (?,?,?,?,?,?)",
            (appmod._hash_token(tok), email, purpose, extra,
             fmt(now), fmt(now + timedelta(minutes=minutes))))
        cx.commit()
    return tok


def _consumed(appmod, tok):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        r = cx.execute("SELECT consumed_at FROM auth_tokens WHERE token_hash=?",
                       (appmod._hash_token(tok),)).fetchone()
    return r is not None and r[0] is not None


# The emailed sign-in links that used to burn on GET, as (purpose, url template).
SIGNIN_LINKS = [
    ("reorder", "/reorder/auth/{tok}"),
    ("biofield", "/biofield/auth/{tok}"),
    ("membership_magic_link", "/coaching/auth/{tok}"),
]


@pytest.mark.parametrize("purpose,url", SIGNIN_LINKS)
def test_scanner_get_does_not_burn_the_token(tmp_path, monkeypatch, purpose, url):
    appmod = _app(tmp_path, monkeypatch)
    tok = _mint(appmod, purpose)
    c = appmod.app.test_client()

    r = c.get(url.format(tok=tok))          # the scanner
    assert r.status_code == 200, "GET should render the confirm page, not redirect"
    assert not _consumed(appmod, tok), "GET burned the token"


@pytest.mark.parametrize("purpose,url", SIGNIN_LINKS)
def test_human_can_still_sign_in_after_a_scanner_fetched_the_link(
        tmp_path, monkeypatch, purpose, url):
    """The whole point: scanner first, human second, human still gets in."""
    appmod = _app(tmp_path, monkeypatch)
    tok = _mint(appmod, purpose)
    c = appmod.app.test_client()

    c.get(url.format(tok=tok))                       # scanner prefetch
    r = c.post(url.format(tok=tok))                  # human clicks Continue
    assert r.status_code == 302, f"POST should sign in and redirect, got {r.status_code}"
    assert _consumed(appmod, tok), "POST must consume the token"


@pytest.mark.parametrize("purpose,url", SIGNIN_LINKS)
def test_token_is_still_single_use_across_two_posts(tmp_path, monkeypatch, purpose, url):
    appmod = _app(tmp_path, monkeypatch)
    tok = _mint(appmod, purpose)
    c = appmod.app.test_client()

    assert c.post(url.format(tok=tok)).status_code == 302
    second = c.post(url.format(tok=tok))
    assert second.status_code != 302, "a consumed token must not sign anyone in again"


def test_confirm_page_carries_no_autosubmitting_script(tmp_path, monkeypatch):
    """An auto-submitting form would re-create the exact bug this prevents."""
    appmod = _app(tmp_path, monkeypatch)
    tok = _mint(appmod, "reorder")
    body = appmod.app.test_client().get(f"/reorder/auth/{tok}").get_data(as_text=True)
    assert "<form" in body and 'method="post"' in body.lower().replace("'", '"')
    assert "<script" not in body.lower()
    assert "submit()" not in body.replace(" ", "")


def test_magic_link_verify_get_does_not_burn(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    tok = _mint(appmod, "magic_link")
    c = appmod.app.test_client()

    r = c.get(f"/auth/magic-link/verify?token={tok}")
    assert r.status_code == 200
    assert not _consumed(appmod, tok)
    # the token rides in a hidden field so the POST can find it again
    assert tok in r.get_data(as_text=True)

    assert c.post("/auth/magic-link/verify", data={"token": tok}).status_code in (302, 200)
    assert _consumed(appmod, tok)


def test_practitioner_optout_get_does_not_opt_anyone_out(tmp_path, monkeypatch):
    """Opting out is a mutation. A scanner GET must not perform it."""
    appmod = _app(tmp_path, monkeypatch)
    tok = _mint(appmod, "practitioner_optout", email="doc@example.com",
                minutes=60 * 24, extra='{"practitioner_id": "p1"}')
    c = appmod.app.test_client()

    r = c.get(f"/practitioner-optout/{tok}")
    assert r.status_code == 200
    assert not _consumed(appmod, tok)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rows = cx.execute(
            "SELECT COUNT(*) FROM practitioner_inquiry_opt_outs WHERE email=?",
            ("doc@example.com",)).fetchone()[0]
    assert rows == 0, "GET opted the practitioner out"
