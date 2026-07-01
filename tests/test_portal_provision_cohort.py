from scripts.portal_provision_cohort import build_payloads


def test_build_payloads_normalizes_and_flags_send():
    rows = [{"name": "Maria Sutryn", "email": "Maria_Sutryn@Outlook.com ", "confidence": "high"}]
    out = build_payloads(rows, send=True)
    assert out == [{"email": "maria_sutryn@outlook.com", "name": "Maria Sutryn", "send": True}]


def test_build_payloads_skips_blank_or_low_confidence():
    rows = [
        {"name": "No Email", "email": "", "confidence": "high"},
        {"name": "Fuzzy Match", "email": "x@y.com", "confidence": "unresolved"},
    ]
    assert build_payloads(rows, send=True) == []
