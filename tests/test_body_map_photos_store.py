# tests/test_body_map_photos_store.py
import sqlite3
from dashboard import body_map_photos as bmp


def _cx():
    cx = sqlite3.connect(":memory:")
    bmp.init_table(cx)
    return cx


def test_bytea_blob_round_trips_all_byte_values():
    cx = _cx()
    blob = bytes(range(256)) * 30
    assert bmp.put(cx, "c@x.com", "face", "", blob, "image/jpeg", "portal-self") is True
    got = bmp.get(cx, "c@x.com", "face", "")
    assert got["blob"] == blob and got["content_type"] == "image/jpeg"
    assert got["transform"] is None and got["source"] == "portal-self"


def test_slot_key_is_email_system_side():
    cx = _cx()
    bmp.put(cx, "C@x.com", "iris", "left", b"L", "image/png", "portal-self")
    bmp.put(cx, "c@x.com", "iris", "right", b"R", "image/png", "portal-self")
    assert bmp.get(cx, "c@x.com", "iris", "left")["blob"] == b"L"
    assert bmp.get(cx, "c@x.com", "iris", "right")["blob"] == b"R"
    rows = bmp.list_for_email(cx, "c@x.com")
    assert {(r["system"], r["side"]) for r in rows} == {("iris", "left"), ("iris", "right")}


def test_none_side_normalizes_to_empty():
    cx = _cx()
    bmp.put(cx, "c@x.com", "face", None, b"F", "image/jpeg", "console")
    assert bmp.get(cx, "c@x.com", "face", "")["blob"] == b"F"
    assert bmp.get(cx, "c@x.com", "face", None)["blob"] == b"F"


def test_reput_replaces_photo_and_clears_transform():
    cx = _cx()
    bmp.put(cx, "c@x.com", "face", "", b"one", "image/jpeg", "portal-self")
    bmp.set_transform(cx, "c@x.com", "face", "", {"mx": 1, "my": 0, "tx": 2, "ty": 3})
    bmp.put(cx, "c@x.com", "face", "", b"two", "image/jpeg", "portal-self")  # new photo
    got = bmp.get(cx, "c@x.com", "face", "")
    assert got["blob"] == b"two" and got["transform"] is None       # transform cleared
    assert len(bmp.list_for_email(cx, "c@x.com")) == 1              # still one row


def test_set_transform_round_trips_and_clears():
    cx = _cx()
    bmp.put(cx, "c@x.com", "face", "", b"f", "image/jpeg", "portal-self")
    assert bmp.set_transform(cx, "c@x.com", "face", "",
                             {"mx": 1.5, "my": -0.5, "tx": 300, "ty": 12.25}) is True
    assert bmp.get_transform(cx, "c@x.com", "face", "") == {"mx": 1.5, "my": -0.5,
                                                            "tx": 300.0, "ty": 12.25}
    assert bmp.set_transform(cx, "c@x.com", "face", "", None) is True   # clear
    assert bmp.get_transform(cx, "c@x.com", "face", "") is None


def test_set_transform_rejects_malformed():
    cx = _cx()
    bmp.put(cx, "c@x.com", "face", "", b"f", "image/jpeg", "portal-self")
    for bad in ({"mx": 1, "my": 0, "tx": 2}, {"mx": "x", "my": 0, "tx": 2, "ty": 3},
                {"mx": float("nan"), "my": 0, "tx": 2, "ty": 3}, {"mx": True, "my": 0, "tx": 2, "ty": 3},
                {"mx": float("inf"), "my": 0, "tx": 2, "ty": 3}, "notadict", []):
        assert bmp.set_transform(cx, "c@x.com", "face", "", bad) is False
    assert bmp.get_transform(cx, "c@x.com", "face", "") is None   # nothing persisted


def test_get_missing_returns_none():
    assert bmp.get(_cx(), "nobody@x.com", "face", "") is None


def test_list_excludes_blobs_and_reports_has_transform():
    cx = _cx()
    bmp.put(cx, "c@x.com", "face", "", b"f", "image/jpeg", "portal-self")
    bmp.put(cx, "c@x.com", "hand", "", b"h", "image/jpeg", "portal-self")
    bmp.set_transform(cx, "c@x.com", "hand", "", {"mx": 1, "my": 0, "tx": 0, "ty": 0})
    rows = {r["system"]: r for r in bmp.list_for_email(cx, "c@x.com")}
    assert "blob" not in rows["face"] and "image_blob" not in rows["face"]
    assert rows["face"]["has_transform"] is False
    assert rows["hand"]["has_transform"] is True
