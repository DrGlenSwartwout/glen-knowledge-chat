"""Tests for dashboard.practitioner_portal — cart, magic-link/session tokens,
two-door registration validation, and the registration insert (Phase 3c/3d)."""

from datetime import datetime, timedelta

import pytest

PID = "00000000-0000-0000-0000-000000000009"


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "chat_log.db")


# ── validate_registration (pure) ──────────────────────────────────────────────

def test_validate_licensed_ok():
    from dashboard.practitioner_portal import validate_registration
    clean, err = validate_registration({
        "email": "DR@Clinic.com", "name": "Dr Jane", "portal_role": "licensed",
        "license_number": "OD-123", "license_state": "CA"})
    assert err is None
    assert clean["email"] == "dr@clinic.com"   # lowercased
    assert clean["portal_role"] == "licensed"


def test_validate_coach_ok():
    from dashboard.practitioner_portal import validate_registration
    clean, err = validate_registration({
        "email": "c@x.com", "name": "Coach", "portal_role": "coach",
        "resale_license_number": "RS-9"})
    assert err is None and clean["portal_role"] == "coach"


def test_validate_rejects_bad_email_role_and_missing_license():
    from dashboard.practitioner_portal import validate_registration
    assert validate_registration({"email": "nope", "name": "x", "portal_role": "licensed"})[0] is None
    assert validate_registration({"email": "a@b.com", "name": "x", "portal_role": "other"})[0] is None
    assert validate_registration({"email": "a@b.com", "name": "x",
                                  "portal_role": "licensed"})[1]  # no license_number
    assert validate_registration({"email": "a@b.com", "name": "x",
                                  "portal_role": "coach"})[1]      # no resale number


# ── cart (SQLite) ─────────────────────────────────────────────────────────────

def test_cart_set_get_update_remove_clear(db):
    from dashboard.practitioner_portal import cart_set, cart_items, cart_clear
    cart_set(PID, "a", 5, db_path=db)
    cart_set(PID, "b", 2, db_path=db)
    assert cart_items(PID, db_path=db) == [{"slug": "a", "qty": 5}, {"slug": "b", "qty": 2}]
    cart_set(PID, "a", 9, db_path=db)               # update
    assert cart_items(PID, db_path=db)[0]["qty"] == 9
    cart_set(PID, "a", 0, db_path=db)               # remove
    assert [i["slug"] for i in cart_items(PID, db_path=db)] == ["b"]
    cart_clear(PID, db_path=db)
    assert cart_items(PID, db_path=db) == []


# ── tokens ────────────────────────────────────────────────────────────────────

def test_magic_link_is_single_use(db):
    from dashboard.practitioner_portal import create_magic_link_token, consume_magic_link
    t0 = datetime(2026, 6, 1, 12, 0, 0)
    tok = create_magic_link_token(PID, "dr@x.com", now=t0, db_path=db)
    assert consume_magic_link(tok, now=t0, db_path=db) == PID
    assert consume_magic_link(tok, now=t0, db_path=db) is None   # already used


def test_magic_link_expires(db):
    from dashboard.practitioner_portal import create_magic_link_token, consume_magic_link
    t0 = datetime(2026, 6, 1, 12, 0, 0)
    tok = create_magic_link_token(PID, "dr@x.com", now=t0, db_path=db)
    assert consume_magic_link(tok, now=t0 + timedelta(minutes=16), db_path=db) is None


def test_session_token_validates_until_expiry(db):
    from dashboard.practitioner_portal import create_session_token, practitioner_id_from_session
    t0 = datetime(2026, 6, 1, 12, 0, 0)
    tok = create_session_token(PID, now=t0, db_path=db)
    assert practitioner_id_from_session(tok, now=t0, db_path=db) == PID
    assert practitioner_id_from_session(tok, now=t0 + timedelta(days=31), db_path=db) is None
    assert practitioner_id_from_session("garbage", now=t0, db_path=db) is None


# ── registration insert (fake Supabase cursor) ────────────────────────────────

class _FakeCur:
    def __init__(self):
        self.inserts = []
        self._r = None
    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        if s.startswith("SELECT id FROM practitioners WHERE lower(email)"):
            self._r = None                      # new email -> insert path
        elif "INSERT INTO practitioners" in s and "RETURNING id" in s:
            self.inserts.append(list(params))
            self._r = {"id": "P-NEW"}
        else:
            self._r = None
    def fetchone(self):
        return self._r


class _FakeCtx:
    def __init__(self, cur): self.cur = cur
    def __enter__(self): return self.cur
    def __exit__(self, *a): return False


@pytest.fixture
def fake_supabase(monkeypatch):
    cur = _FakeCur()
    import db_supabase
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(cur))
    return cur


def test_register_licensed_unlocks_immediately(fake_supabase):
    from dashboard.practitioner_portal import register_practitioner, validate_registration
    clean, _ = validate_registration({"email": "od@x.com", "name": "OD",
                                      "portal_role": "licensed", "license_number": "1"})
    pid, unlocked = register_practitioner(clean, now=datetime(2026, 6, 1))
    assert pid == "P-NEW"
    assert unlocked is True
    assert fake_supabase.inserts[0][-1] is not None        # wholesale_unlocked_at set


def test_order_history_records_newest_first_and_is_idempotent(db):
    from dashboard.practitioner_portal import record_order, order_history
    record_order(PID, invoice_id="INV1", doc_number="1001", total_cents=100000,
                 credit_cents=50000, db_path=db, now=datetime(2026, 6, 1))
    record_order(PID, invoice_id="INV2", doc_number="1002", total_cents=80000,
                 db_path=db, now=datetime(2026, 6, 2))
    record_order(PID, invoice_id="INV1", doc_number="1001", total_cents=100000,
                 db_path=db, now=datetime(2026, 6, 3))   # retry — must not duplicate
    h = order_history(PID, db_path=db)
    assert [o["invoice_id"] for o in h] == ["INV2", "INV1"]   # newest first
    assert len(h) == 2
    assert h[1]["credit_cents"] == 50000


def test_name_to_slug_exact_and_fuzzy():
    from dashboard.practitioner_portal import name_to_slug
    cat = {"gi-repair": {"name": "GI Repair"}, "microbiome": {"name": "Microbiome"}}
    assert name_to_slug("gi repair", cat) == "gi-repair"            # exact, case-insensitive
    assert name_to_slug("microbiome support", cat) == "microbiome"  # fuzzy (pn within nl)
    assert name_to_slug("nope", cat) is None
    assert name_to_slug("", cat) is None


def test_assist_cross_sell_resolves_in_catalog_only():
    from dashboard.practitioner_portal import assist_cross_sell
    cat = {"gastrozyme": {"name": "GastroZyme"}, "gi-repair": {"name": "GI Repair"},
           "microbiome": {"name": "Microbiome"}}
    pairings = {"pairings": {"gastrozyme": ["GI Repair", "Microbiome", "Nonexistent Tool"]}}
    out = assist_cross_sell("gastrozyme", catalog=cat, pairings=pairings)
    assert out == [{"name": "GI Repair", "slug": "gi-repair"},
                   {"name": "Microbiome", "slug": "microbiome"}]   # out-of-catalog dropped
    assert assist_cross_sell("unknown-slug", catalog=cat, pairings=pairings) == []


def test_resolve_named_products_keeps_in_catalog_dedup():
    from dashboard.practitioner_portal import resolve_named_products
    cat = {"lens-zyme": {"name": "Lens-Zyme"},
           "crystalline-clarity": {"name": "Crystalline Clarity"}}
    items = [{"name": "Lens-Zyme", "why": "a"},
             {"name": "Crystalline Clarity", "why": "b"},
             {"name": "Made Up Tool", "why": "c"},      # not in catalog -> dropped
             {"name": "Lens-Zyme", "why": "dup"}]        # dup slug -> dropped
    out = resolve_named_products(items, cat)
    assert out == [{"name": "Lens-Zyme", "why": "a", "slug": "lens-zyme"},
                   {"name": "Crystalline Clarity", "why": "b", "slug": "crystalline-clarity"}]
    assert resolve_named_products([], cat) == []


def test_record_dispensary_order_history(db):
    from dashboard.practitioner_portal import record_dispensary_order, dispensary_order_history
    record_dispensary_order(PID, invoice_id="CINV1", customer_email="c@x.com", bottles=3,
                            credit_earned_cents=6000, db_path=db, now=datetime(2026, 6, 1))
    record_dispensary_order(PID, invoice_id="CINV2", customer_email="d@x.com", bottles=1,
                            credit_earned_cents=2000, db_path=db, now=datetime(2026, 6, 2))
    record_dispensary_order(PID, invoice_id="CINV1", customer_email="c@x.com", bottles=3,
                            credit_earned_cents=6000, db_path=db, now=datetime(2026, 6, 3))  # dup
    h = dispensary_order_history(PID, db_path=db)
    assert [o["invoice_id"] for o in h] == ["CINV2", "CINV1"]   # newest first
    assert len(h) == 2
    assert h[1]["bottles"] == 3 and h[1]["credit_earned_cents"] == 6000


class _DispCur:
    def __init__(self, rows): self.rows = rows; self._r = None
    def execute(self, sql, params=()):
        s = " ".join(sql.split()); p = list(params)
        if s.startswith("SELECT dispensary_code FROM practitioners WHERE id"):
            r = self.rows.get(p[0]); self._r = {"dispensary_code": (r or {}).get("dispensary_code")}
        elif s.startswith("UPDATE practitioners SET dispensary_code"):
            r = self.rows.setdefault(p[1], {"dispensary_code": None})
            if r.get("dispensary_code") is None: r["dispensary_code"] = p[0]
            self._r = None
        elif s.startswith("SELECT id FROM practitioners WHERE dispensary_code"):
            self._r = next(({"id": pid} for pid, r in self.rows.items()
                            if r.get("dispensary_code") == p[0]), None)
        else:
            self._r = None
    def fetchone(self): return self._r


def test_dispensary_code_get_or_create_and_lookup(monkeypatch):
    rows = {PID: {"dispensary_code": None}}
    import db_supabase
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(_DispCur(rows)))
    from dashboard.practitioner_portal import (
        get_or_create_dispensary_code, practitioner_id_by_dispensary_code)
    code = get_or_create_dispensary_code(PID, _gen=lambda: "abc123")
    assert code == "abc123"
    assert get_or_create_dispensary_code(PID, _gen=lambda: "XXX") == "abc123"  # stable
    assert practitioner_id_by_dispensary_code("abc123") == PID
    assert practitioner_id_by_dispensary_code("nope") is None


def test_resolve_named_products_excludes_info_only():
    from dashboard.practitioner_portal import resolve_named_products
    cat = {"lens-zyme": {"name": "Lens-Zyme"}, "emf": {"name": "EMF", "info_only": True}}
    out = resolve_named_products([{"name": "Lens-Zyme", "why": "a"},
                                  {"name": "EMF", "why": "b"}], cat)
    assert out == [{"name": "Lens-Zyme", "why": "a", "slug": "lens-zyme"}]   # EMF dropped


def test_is_orderable():
    from dashboard.practitioner_portal import is_orderable
    cat = {"lens-zyme": {"name": "Lens-Zyme"}, "emf": {"name": "EMF", "info_only": True}}
    assert is_orderable("lens-zyme", cat) is True
    assert is_orderable("emf", cat) is False
    assert is_orderable("nope", cat) is False


def test_register_coach_stays_locked(fake_supabase):
    from dashboard.practitioner_portal import register_practitioner, validate_registration
    clean, _ = validate_registration({"email": "c@x.com", "name": "C",
                                      "portal_role": "coach", "resale_license_number": "9"})
    pid, unlocked = register_practitioner(clean, now=datetime(2026, 6, 1))
    assert unlocked is False
    assert fake_supabase.inserts[0][-1] is None            # locked until first module


def test_modules_completed_for_email(monkeypatch):
    import db_supabase
    from dashboard import practitioner_portal as pp

    class _Cur:
        def __init__(self, row): self._row = row
        def execute(self, *a, **k): self._a = a
        def fetchone(self): return self._row

    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(_Cur({"modules_completed": 9})))
    assert pp.modules_completed_for_email("doc@x.com") == 9

    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(_Cur({"modules_completed": None})))
    assert pp.modules_completed_for_email("doc@x.com") == 0

    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(_Cur(None)))
    assert pp.modules_completed_for_email("nobody@x.com") is None

    assert pp.modules_completed_for_email("") is None
