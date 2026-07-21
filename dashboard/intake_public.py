"""Public (funnel) intake: scoped edit-tokens and the safe public submit path.

The funnel's truly.vip/join short link points at /begin/intake, an UNAUTHENTICATED
page a cold visitor fills immediately. Security model:

  * A visitor NEVER receives the master portal token. On opt-in they get a SCOPED
    intake-session token (random, stored, expiring) that authorizes editing ONLY
    their own intake row — never portal access. The master portal setup link is
    emailed to the address separately (the hybrid: low-friction fill now + a
    verified portal identity delivered by email).
  * The intake row is always keyed by the OPT-IN email bound to the token, never
    by an email typed into the form body — so a session can only ever write its
    own row (no impersonation, no cross-client overwrite).

Pure logic only: no Flask, no network. The LLM I/O for the chat mode lives in the
route; the prompt/parse helpers live in intake_chat.py.
"""
import secrets

TOKEN_TTL_HOURS = 72  # scoped intake-session lifetime


def init_intake_sessions_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS intake_sessions ("
        " token TEXT PRIMARY KEY,"
        " email TEXT NOT NULL,"
        " name TEXT,"
        " created_at TEXT NOT NULL,"
        " expires_at REAL NOT NULL)")          # epoch seconds — tz-proof compare


def create_session(cx, email, name, now):
    """Mint a scoped intake-session token bound to `email`. `now` is a datetime.
    Returns the opaque token string."""
    email = (email or "").strip().lower()
    token = secrets.token_urlsafe(32)
    expires = now.timestamp() + TOKEN_TTL_HOURS * 3600
    cx.execute(
        "INSERT INTO intake_sessions (token, email, name, created_at, expires_at)"
        " VALUES (?,?,?,?,?)",
        (token, email, (name or "").strip(), now.isoformat(), expires))
    cx.commit()
    return token


def resolve_session(cx, token, now):
    """Return the bound email for a valid, unexpired token, else None."""
    if not token:
        return None
    row = cx.execute(
        "SELECT email, expires_at FROM intake_sessions WHERE token=?", (token,)).fetchone()
    if not row:
        return None
    email, expires = row[0], row[1]
    if now.timestamp() > float(expires):
        return None
    return email


def merge_answers(current, updates):
    """Merge LLM-extracted `updates` into `current`, keeping ONLY known field ids
    and coercing scale values to ints. Unknown keys and non-numeric scales are
    dropped. Never trusts a submitted `email` here — identity is the token's."""
    from dashboard import intake as _intake
    known = {f["id"]: f for f in _intake._fields()}
    out = dict(current or {})
    for k, v in (updates or {}).items():
        if k not in known or k == "email":
            continue
        f = known[k]
        if f["type"] == "scale":
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
        out[k] = v
    return out


def public_submit(cx, email, answers, now_iso):
    """Submit a funnel intake, keyed by the token-bound `email` (not any email in
    `answers`). GUARD: never overwrite a genuine already-submitted intake — mirror
    import_response's guard so a re-submit or an abusive replay can't clobber real
    data. Returns 'submitted' | 'already'.
    """
    from dashboard import intake as _intake
    email = (email or "").strip().lower()
    existing = _intake.get_response(cx, email)
    if existing and existing["status"] == "submitted":
        a = existing["answers"] or {}
        if not a.get("_external") and not a.get("_imported"):
            return "already"
    payload = {**(answers or {}), "email": email, "_source": "public-funnel"}
    _intake.submit(cx, email, payload, now_iso)
    return "submitted"


def save_public_draft(cx, email, answers, now_iso):
    """Autosave a funnel draft, keyed by the token-bound email. No-op if the row is
    already a genuine submission (can't scribble over a finished intake)."""
    from dashboard import intake as _intake
    email = (email or "").strip().lower()
    if _intake.is_submitted(cx, email):
        return False
    payload = {**(answers or {}), "email": email, "_source": "public-funnel"}
    _intake.save_draft(cx, email, payload, now_iso)
    return True
