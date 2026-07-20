import re

import pytest
from dashboard import practitioner_profile as pp


def test_sanitize_bio_strips_html_and_collapses_ws():
    assert pp.sanitize_bio("  <b>Dr.</b>  Glen   heals ") == "Dr. Glen heals"


def test_sanitize_bio_keeps_contact_detail():
    """Deliberate divergence from the client sanitizer — a practitioner may put
    their own phone/email/URL in their own bio."""
    s = pp.sanitize_bio("Reach me at dr@x.com or 555-123-4567, https://drglen.com")
    assert "dr@x.com" in s and "555-123-4567" in s and "https://drglen.com" in s


def test_sanitize_bio_rejects_over_600():
    with pytest.raises(ValueError):
        pp.sanitize_bio("x" * 601)


def test_sanitize_bio_600_exactly_ok():
    assert len(pp.sanitize_bio("x" * 600)) == 600


def test_clean_services_strips_caps_and_drops_empties():
    out = pp.clean_services(["<i>Acupuncture</i>", "  ", "Nutrition", ""])
    assert out == ["Acupuncture", "Nutrition"]


def test_clean_services_caps_count_at_12():
    assert len(pp.clean_services([f"svc{i}" for i in range(20)])) == 12


def test_clean_services_caps_item_len_at_60():
    """Finding 2: MAX_SERVICE_LEN=60 truncation had no test."""
    out = pp.clean_services(["x" * 100])
    assert out == ["x" * 60]
    assert len(out[0]) == 60


def test_clean_services_60_char_item_preserved_whole():
    """Boundary: exactly 60 chars must survive untruncated."""
    item = "x" * 60
    out = pp.clean_services([item])
    assert out == [item]
    assert len(out[0]) == 60


# --- Finding 1: bare comparison operators in prose must not be eaten as tags ---

def test_sanitize_bio_survives_comparison_operators_intact():
    text = "Reduced A1C from 9.2 to <6.0 and BP <120 >80 today"
    assert pp.sanitize_bio(text) == text


def test_sanitize_bio_survives_spaced_comparison_intact():
    text = "kept IOP < 15 mmHg"
    assert pp.sanitize_bio(text) == text


def test_sanitize_bio_still_strips_bold_tags():
    assert pp.sanitize_bio("<b>Dr.</b> Glen") == "Dr. Glen"


def test_sanitize_bio_still_strips_script_tag_markup():
    out = pp.sanitize_bio("<script>alert(1)</script>bio")
    assert "<script>" not in out
    assert "</script>" not in out


def test_sanitize_bio_still_strips_img_tag_with_attrs():
    assert pp.sanitize_bio("<img src=x onerror=y>hi") == "hi"


def test_sanitize_bio_nested_angle_brackets_no_interpretable_tag_survives():
    out = pp.sanitize_bio("<<b>>text")
    # the inner "<b>" is a real tag and is stripped; the leftover "<" and ">"
    # are not adjacent to a letter, so they can't be interpreted as a tag.
    assert "<b>" not in out
    assert not re.search(r"<\s*/?\s*[a-zA-Z]", out)


def test_format_location_variants():
    assert pp.format_location("Hilo", "HI") == "Hilo, HI"
    assert pp.format_location("Hilo", "") == "Hilo"
    assert pp.format_location("", "HI") == ""
    assert pp.format_location(None, None) == ""


def test_profile_public_fields_frozen():
    assert pp.PROFILE_PUBLIC_FIELDS == frozenset(
        {"bio", "photo_url", "logo_url", "services", "location", "accepting_clients"})


# --- Task 3: profile_for_slug — provenance-gated read ---

import sqlite3


class _FakeCur:
    """Serves one configurable practitioners row for the SELECT; records nothing else."""
    def __init__(self, row):
        self._row = row
    def execute(self, sql, params=()):
        self._last = " ".join(sql.split())
    def fetchone(self):
        return self._row
    def close(self):
        pass


class _FakeCtx:
    def __init__(self, cur): self.cur = cur
    def __enter__(self): return self.cur
    def __exit__(self, *a): return False


def _cx_with_slug(slug="prof-jane-doe", email="jane@example.com"):
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    cx.executescript(
        "CREATE TABLE affiliate_signups (slug TEXT, email TEXT, status TEXT);")
    cx.execute("INSERT INTO affiliate_signups VALUES (?,?, 'approved')", (slug, email))
    cx.commit()
    return cx


def _patch_supabase(monkeypatch, row):
    import db_supabase
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(_FakeCur(row)))


def test_profile_for_slug_unknown_slug_returns_empty(monkeypatch):
    _patch_supabase(monkeypatch, None)
    assert pp.profile_for_slug(_cx_with_slug(), "no-such-slug") == {}


def test_profile_for_slug_scraped_row_returns_empty(monkeypatch):
    """PROVENANCE MUTATION TEST: a row WITH a bio but null timestamp must publish
    nothing. Proves the gate filters, not that the happy path happens to be empty."""
    scraped = {"bio": "scraped text", "photo_url": "p", "logo_url": "",
               "specialties": ["x"], "city": "Hilo", "state": "HI",
               "accepting_new_patients": True, "profile_self_authored_at": None}
    _patch_supabase(monkeypatch, scraped)
    assert pp.profile_for_slug(_cx_with_slug(), "prof-jane-doe") == {}


def test_profile_for_slug_self_authored_row_publishes(monkeypatch):
    authored = {"bio": "I heal", "photo_url": "https://x/p.jpg", "logo_url": "",
                "specialties": ["Acupuncture", "Nutrition"], "city": "Hilo",
                "state": "HI", "accepting_new_patients": True,
                "profile_self_authored_at": "2026-07-20T00:00:00Z"}
    _patch_supabase(monkeypatch, authored)
    v = pp.profile_for_slug(_cx_with_slug(), "prof-jane-doe")
    assert v["bio"] == "I heal"
    assert v["services"] == ["Acupuncture", "Nutrition"]
    assert v["location"] == "Hilo, HI"
    assert v["accepting_clients"] is True


def test_profile_for_slug_never_returns_street_address(monkeypatch):
    """address1/postal must never appear even if present on the row."""
    authored = {"bio": "b", "photo_url": "", "logo_url": "", "specialties": [],
                "city": "Hilo", "state": "HI", "accepting_new_patients": True,
                "profile_self_authored_at": "2026-07-20T00:00:00Z",
                "address1": "123 Secret St", "postal": "96720"}
    _patch_supabase(monkeypatch, authored)
    v = pp.profile_for_slug(_cx_with_slug(), "prof-jane-doe")
    assert "123 Secret St" not in str(v.values())
    assert "96720" not in str(v.values())
    assert set(v) <= pp.PROFILE_PUBLIC_FIELDS


def test_profile_for_slug_supabase_down_returns_empty(monkeypatch):
    """A DB fault must degrade to {}, never raise — the storefront stays up."""
    import db_supabase
    def _boom():
        raise RuntimeError("supabase down")
    monkeypatch.setattr(db_supabase, "supabase_cursor", _boom)
    assert pp.profile_for_slug(_cx_with_slug(), "prof-jane-doe") == {}


class _MultiRowCur:
    """Models multiple `practitioners` rows sharing an email (the table has no
    unique constraint on email). fetchone() returns the row an
    `ORDER BY profile_self_authored_at DESC NULLS LAST LIMIT 1` would pick --
    but ONLY when the executed SQL actually contains that clause. Without it,
    fetchone() returns rows in raw insertion order (the scraped row is seeded
    first), so a future removal of the ORDER BY makes this test fail rather
    than silently pass."""
    def __init__(self, rows):
        self._rows = rows
        self.last_sql = ""

    def execute(self, sql, params=()):
        self.last_sql = " ".join(sql.split())

    def fetchone(self):
        if "order by profile_self_authored_at desc" in self.last_sql.lower():
            ordered = sorted(
                self._rows,
                key=lambda r: r.get("profile_self_authored_at") is None)
            return ordered[0]
        return self._rows[0]

    def close(self):
        pass


def test_profile_for_slug_prefers_authored_row_when_email_duplicated(monkeypatch):
    """IMPORTANT fix: practitioners.email has NO unique constraint. If two
    scraped rows share an email, the read must return the self-authored row
    (the one the write path stamped) over a scraped duplicate -- otherwise a
    practitioner's saved profile can silently be shadowed by another row."""
    scraped = {"bio": "scraped text", "photo_url": "scraped.jpg", "logo_url": "",
               "specialties": ["scraped-svc"], "city": "Nowhere", "state": "XX",
               "accepting_new_patients": True, "profile_self_authored_at": None}
    authored = {"bio": "I heal", "photo_url": "https://x/p.jpg", "logo_url": "",
                "specialties": ["Acupuncture"], "city": "Hilo", "state": "HI",
                "accepting_new_patients": True,
                "profile_self_authored_at": "2026-07-20T00:00:00Z"}
    cur = _MultiRowCur([scraped, authored])  # scraped seeded FIRST
    import db_supabase
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(cur))

    v = pp.profile_for_slug(_cx_with_slug(), "prof-jane-doe")

    assert v["bio"] == "I heal"
    assert v["services"] == ["Acupuncture"]
    assert v["location"] == "Hilo, HI"
    # Prove the ordering was actually requested, not just that the fake
    # happened to return the right row -- a future removal of the ORDER BY
    # clause must fail this test.
    assert "profile_self_authored_at desc" in cur.last_sql.lower()


# --- Task 5: save_profile — the write path ---

class _RecordingCur:
    """Records the UPDATE sql + params."""
    def __init__(self): self.calls = []
    def execute(self, sql, params=()):
        self.calls.append((" ".join(sql.split()), list(params)))
    def fetchone(self): return None
    def close(self): pass


def _patch_recording(monkeypatch):
    cur = _RecordingCur()
    import db_supabase
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(cur))
    return cur


def test_save_profile_stamps_provenance_and_sanitizes(monkeypatch):
    cur = _patch_recording(monkeypatch)
    out = pp.save_profile("pid-1", {
        "bio": "<b>I heal</b> reach dr@x.com",
        "photo_url": " https://x/p.jpg ",
        "services": ["<i>Acupuncture</i>", ""],
        "city": "Hilo", "state": "HI", "accepting_clients": False})
    assert out["bio"] == "I heal reach dr@x.com"       # HTML stripped, email kept
    assert out["services"] == ["Acupuncture"]
    assert out["photo_url"] == "https://x/p.jpg"
    assert out["accepting_clients"] is False
    sql, params = cur.calls[-1]
    assert "UPDATE practitioners SET" in sql
    assert "profile_self_authored_at=now()" in sql.replace(" ", "").lower() \
        or "profile_self_authored_at = now()" in sql
    assert "pid-1" in params


def test_save_profile_rejects_long_bio(monkeypatch):
    _patch_recording(monkeypatch)
    with pytest.raises(ValueError):
        pp.save_profile("pid-1", {"bio": "x" * 601})
