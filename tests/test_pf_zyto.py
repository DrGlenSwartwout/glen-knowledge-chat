"""Unit tests for the vetted ZYTO / EVOX CSV migration (no DB / no network).

Covers the pure routing logic: which CSV rows go to the patient map vs the GHL
Elite/Pro prospect list, row normalization, and the GHL dry-run formatter."""
import csv
import io

from scrapers.practitioner_finder import migrate_zyto as m


def test_truthy_and_clean():
    assert m._truthy("true") and m._truthy("Yes") and m._truthy("1") and m._truthy("X")
    assert not m._truthy("") and not m._truthy("no") and not m._truthy(None)
    assert m._clean("  hi ") == "hi"
    assert m._clean("   ") is None and m._clean(None) is None


def test_wants_map_override_and_fallback():
    # falls back to offers_evox when no list_on_map column / value
    assert m._wants_map({"offers_evox": "true"}) is True
    assert m._wants_map({"offers_evox": "false"}) is False
    assert m._wants_map({"offers_evox": ""}) is False
    # explicit vetting override wins both ways
    assert m._wants_map({"offers_evox": "false", "list_on_map": "true"}) is True
    assert m._wants_map({"offers_evox": "true", "list_on_map": "false"}) is False
    # blank override falls through to offers_evox
    assert m._wants_map({"offers_evox": "true", "list_on_map": ""}) is True


def test_to_row_tags_and_required_fields():
    row = m._to_row({
        "name": "Dr. Jane Doe", "practice_name": "Wellness Inc",
        "city": "Austin", "state": "TX", "postal": "78701",
        "website": "https://w.com", "source_url": "https://w.com/evox",
        "phone": "555-1212", "email": "j@w.com",
    })
    d = row.to_dict()
    assert d["tier"] == "org_member"
    assert d["source_org"] == "ZYTO"
    assert d["specialties"] == ["holistic_health", "biocommunication"]
    assert d["source_url"] == "https://w.com/evox"
    assert d["country"] == "US"
    # telehealth is set post-upsert, never via the shared row model
    assert "telehealth" not in d


def test_to_row_source_url_falls_back_to_website():
    row = m._to_row({"name": "Dr. X", "website": "https://x.com", "source_url": ""})
    assert row.source_url == "https://x.com"


def test_to_row_skips_without_key_or_name():
    assert m._to_row({"name": "", "source_url": ""}) is None
    assert m._to_row({"name": "Dr. X", "source_url": "", "website": ""}) is None
    # a practice_name alone is enough for a display name
    assert m._to_row({"practice_name": "Clinic", "source_url": "https://c.com"}) is not None


def test_ghl_eligibility_elite_pro_with_email_only():
    assert m._ghl_eligible({"device_tier": "Elite", "email": "a@b.com"}) is True
    assert m._ghl_eligible({"device_tier": "pro", "email": "a@b.com"}) is True
    assert m._ghl_eligible({"device_tier": "Compass", "email": "a@b.com"}) is False
    assert m._ghl_eligible({"device_tier": "unknown", "email": "a@b.com"}) is False
    assert m._ghl_eligible({"device_tier": "Elite", "email": ""}) is False


def test_load_ghl_dry_run_counts_only_eligible():
    recs = [
        {"name": "Dr. A B", "device_tier": "Elite", "email": "a@b.com",
         "source_url": "https://x.com", "phone": "555"},
        {"name": "Dr. C", "device_tier": "Compass", "email": "c@d.com"},   # not Elite/Pro
        {"name": "Dr. E", "device_tier": "Pro", "email": ""},               # no email
    ]
    res = m.load_ghl(recs, dry_run=True)
    assert res == {"eligible": 1, "synced": 1, "errors": 0}


def test_read_csv_roundtrip(tmp_path):
    path = tmp_path / "zyto.csv"
    fieldnames = ["name", "city", "state", "email", "device_tier",
                  "offers_evox", "source_url"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({"name": "Dr. A", "city": "Reno", "state": "NV",
                    "email": "a@reno.com", "device_tier": "Elite",
                    "offers_evox": "true", "source_url": "https://a.com/evox"})
    rows = m.read_csv(str(path))
    assert len(rows) == 1 and rows[0]["name"] == "Dr. A"
    assert m._wants_map(rows[0]) is True
    assert m._ghl_eligible(rows[0]) is True
