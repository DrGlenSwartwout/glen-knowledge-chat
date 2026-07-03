# tests/test_portal_identity.py
"""The identity seam: turn a portal token (today) or a client session cookie
(scaffolded, inactive) into an Identity, resolved against the unified people row.
This is the single choke point real login drops into next slice."""
import sqlite3

import pytest


def _seed_portal(cx, email="brooke@example.com", name="Brooke Webb", content=None):
    from dashboard import client_portal as cp
    cp.init_client_portal_table(cx)
    token, _ = cp.upsert_portal(cx, email, name, content or {"greeting": "hi"})
    return token


def test_identity_from_token_resolves_person_and_roles(tmp_path):
    from dashboard import portal_identity as pi
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    pi._ensure_people_table(cx)
    cx.execute(
        "INSERT INTO people (email, name, roles, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("brooke@example.com", "Brooke Webb", '["client", "student"]', "t", "t"),
    )
    cx.commit()
    tok = _seed_portal(cx)

    ident = pi.identity_from_token(cx, tok)

    assert ident is not None
    assert ident.email == "brooke@example.com"
    assert ident.person_id > 0
    assert ident.roles == ["client", "student"]
    assert ident.auth_method == "token"


def test_identity_from_token_lazily_creates_person_as_client(tmp_path):
    from dashboard import portal_identity as pi
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    pi._ensure_people_table(cx)
    tok = _seed_portal(cx, email="new@example.com", name="New Client")

    ident = pi.identity_from_token(cx, tok)

    assert ident is not None
    assert ident.email == "new@example.com"
    assert ident.person_id > 0
    assert ident.roles == ["client"]  # default role for a portal-link holder
    # persisted, so a second resolve finds the same person row
    again = pi.identity_from_token(cx, tok)
    assert again.person_id == ident.person_id


def test_identity_from_unknown_token_is_none(tmp_path):
    from dashboard import portal_identity as pi
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    pi._ensure_people_table(cx)
    from dashboard import client_portal as cp
    cp.init_client_portal_table(cx)

    assert pi.identity_from_token(cx, "not-a-real-token") is None


# ── Scaffolded session branch (the real-login drop-in point; dark today) ──────

def test_client_session_roundtrip(tmp_path):
    from dashboard import portal_identity as pi
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    pi._ensure_people_table(cx)
    cx.execute(
        "INSERT INTO people (email, name, roles, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("logged-in@example.com", "LI", '["client", "practitioner"]', "t", "t"),
    )
    cx.commit()
    pid = cx.execute("SELECT id FROM people WHERE email=?", ("logged-in@example.com",)).fetchone()[0]

    sess = pi.create_client_session(cx, pid, "logged-in@example.com")
    ident = pi.identity_from_session(cx, sess)

    assert ident is not None
    assert ident.person_id == pid
    assert ident.email == "logged-in@example.com"
    assert ident.roles == ["client", "practitioner"]
    assert ident.auth_method == "session"


def test_client_magic_link_is_one_time(tmp_path):
    from dashboard import portal_identity as pi
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    pi._ensure_people_table(cx)
    cx.execute(
        "INSERT INTO people (email, name, roles, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("ml@example.com", "ML", '["client"]', "t", "t"))
    cx.commit()
    pid = cx.execute("SELECT id FROM people WHERE email=?", ("ml@example.com",)).fetchone()[0]

    link = pi.create_client_magic_link(cx, pid, "ml@example.com")
    assert pi.consume_client_magic_link(cx, link) == pid
    # one-time: a second consume fails
    assert pi.consume_client_magic_link(cx, link) is None


def test_resolve_identity_uses_token_branch(tmp_path):
    from dashboard import portal_identity as pi
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    pi._ensure_people_table(cx)
    tok = _seed_portal(cx, email="tok@example.com", name="Tok")

    ident = pi.resolve_identity(cx, token=tok)

    assert ident is not None and ident.auth_method == "token"
    assert ident.email == "tok@example.com"


def test_resolve_identity_ignores_session_when_login_disabled(tmp_path):
    from dashboard import portal_identity as pi
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    pi._ensure_people_table(cx)
    cx.execute(
        "INSERT INTO people (email, name, roles, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("s@example.com", "S", '["client"]', "t", "t"),
    )
    cx.commit()
    pid = cx.execute("SELECT id FROM people WHERE email=?", ("s@example.com",)).fetchone()[0]
    sess = pi.create_client_session(cx, pid, "s@example.com")

    # client login is dark by default → the session cookie is ignored
    assert pi.resolve_identity(cx, session_token=sess) is None
    # and honored only when the flag is flipped (next slice)
    ident = pi.resolve_identity(cx, session_token=sess, client_login_enabled=True)
    assert ident is not None and ident.auth_method == "session"


def test_resolve_identity_prefers_explicit_token_over_session(tmp_path):
    """An explicit /portal/<token> link wins over a logged-in session, so
    previewing another client's link shows THAT client (matches incognito)
    instead of a mix of the token's content and your own account."""
    from dashboard import portal_identity as pi
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    pi._ensure_people_table(cx)
    # a logged-in session for one person...
    cx.execute(
        "INSERT INTO people (email, name, roles, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("me@example.com", "Me", '["client"]', "t", "t"),
    )
    cx.commit()
    my_pid = cx.execute("SELECT id FROM people WHERE email=?", ("me@example.com",)).fetchone()[0]
    sess = pi.create_client_session(cx, my_pid, "me@example.com")
    # ...opening a DIFFERENT client's token link, login flag on
    other_tok = _seed_portal(cx, email="other@example.com", name="Other")

    ident = pi.resolve_identity(cx, token=other_tok, session_token=sess,
                                client_login_enabled=True)

    assert ident is not None and ident.auth_method == "token"
    assert ident.email == "other@example.com"


def test_resolve_identity_me_sentinel_falls_through_to_session(tmp_path):
    """The tokenless home passes token='me' (never a real token) so it still
    resolves to the logged-in session, not a bogus token lookup."""
    from dashboard import portal_identity as pi
    from dashboard import client_portal as cp
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    pi._ensure_people_table(cx)
    cp.init_client_portal_table(cx)  # route handlers always init this first
    cx.execute(
        "INSERT INTO people (email, name, roles, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("home@example.com", "Home", '["client"]', "t", "t"),
    )
    cx.commit()
    pid = cx.execute("SELECT id FROM people WHERE email=?", ("home@example.com",)).fetchone()[0]
    sess = pi.create_client_session(cx, pid, "home@example.com")

    ident = pi.resolve_identity(cx, token="me", session_token=sess,
                                client_login_enabled=True)

    assert ident is not None and ident.auth_method == "session"
    assert ident.email == "home@example.com"
