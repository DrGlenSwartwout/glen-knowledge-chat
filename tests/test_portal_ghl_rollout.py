from scripts.portal_ghl_rollout import next_wave


def test_next_wave_orders_skips_sent_low_conf_blank_and_caps():
    rows = [
        {"email": "a@x.com", "name": "A", "confidence": "high"},
        {"email": "b@x.com", "name": "B", "confidence": "high"},          # already sent
        {"email": "", "name": "C", "confidence": "high"},                 # blank email
        {"email": "d@x.com", "name": "D", "confidence": "unresolved"},    # low confidence
        {"email": "e@x.com", "name": "E", "confidence": "high"},
        {"email": "f@x.com", "name": "F", "confidence": "high"},          # beyond the cap
    ]
    w = next_wave(rows, sent_emails={"B@x.com"}, wave_size=2)  # sent-set matched case-insensitively
    assert w == [{"email": "a@x.com", "name": "A"}, {"email": "e@x.com", "name": "E"}]


def test_next_wave_is_empty_when_all_sent():
    rows = [{"email": "a@x.com", "name": "A", "confidence": "high"}]
    assert next_wave(rows, sent_emails={"a@x.com"}, wave_size=50) == []
