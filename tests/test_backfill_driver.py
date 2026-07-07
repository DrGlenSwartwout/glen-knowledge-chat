import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "backfill_driver",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "backfill_portal_findings.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


_F = [{"code": "ED3", "name": "Cell Driver", "description": "d", "rank": 1}]


def test_skips_email_without_portal():
    patches, skips = mod.plan_backfill(
        portal_emails={"has@p.com"},
        intake_emails=["missing@p.com"],
        report_dates_of=lambda e: [],
        findings_of=lambda e, d: _F)
    assert patches == []
    assert skips == [{"email": "missing@p.com", "reason": "no existing portal (would create/dup)"}]


def test_matched_with_report_dates_patches_each_date():
    patches, skips = mod.plan_backfill(
        portal_emails={"a@p.com"},
        intake_emails=["a@p.com"],
        report_dates_of=lambda e: ["2026-06-25", "2026-07-01"],
        findings_of=lambda e, d: _F if d == "2026-06-25" else [])
    # only the date that yields findings is patched; the empty one is dropped
    assert patches == [{"email": "a@p.com", "scan_date": "2026-06-25", "findings": _F}]
    assert skips == []


def test_matched_no_report_dates_patches_portal_record():
    patches, skips = mod.plan_backfill(
        portal_emails={"a@p.com"},
        intake_emails=["a@p.com"],
        report_dates_of=lambda e: [],
        findings_of=lambda e, d: _F)
    assert patches == [{"email": "a@p.com", "scan_date": None, "findings": _F}]


def test_matched_but_no_findings_is_skipped():
    patches, skips = mod.plan_backfill(
        portal_emails={"a@p.com"},
        intake_emails=["a@p.com"],
        report_dates_of=lambda e: [],
        findings_of=lambda e, d: [])
    assert patches == []
    assert skips == [{"email": "a@p.com", "reason": "no findings computed"}]
