import base64, sqlite3, sys
from pathlib import Path
import pytest

repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
from scripts import sync_client_photos as s
from dashboard import client_photos as cp

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 8


def _db_with_client(id_pk="21459", email="jc@x.com"):
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE fmp_clients (id_pk TEXT, email TEXT)")
    cx.execute("INSERT INTO fmp_clients (id_pk, email) VALUES (?,?)", (id_pk, email))
    cx.commit()
    return cx


def test_sniff_content_type():
    assert s.sniff_content_type(PNG) == "image/png"
    assert s.sniff_content_type(JPEG) == "image/jpeg"
    assert s.sniff_content_type(b"RIFF\x00\x00\x00\x00WEBPblah") == "image/webp"
    assert s.sniff_content_type(b"not an image") is None


def test_resolve_email_id_and_email_and_unknown():
    cx = _db_with_client("21459", "jc@x.com")
    assert s.resolve_email(cx, "21459") == "jc@x.com"
    assert s.resolve_email(cx, "Someone@X.com") == "someone@x.com"   # email stem passes through
    assert s.resolve_email(cx, "99999") is None                       # unknown id


def test_sync_folder_writes_resolves_and_pushes(tmp_path):
    cx = _db_with_client("21459", "jc@x.com")
    (tmp_path / "21459.jpg").write_bytes(PNG)          # png bytes, .jpg name -> sniffed png
    (tmp_path / "bob@x.com.png").write_bytes(JPEG)     # email-named
    (tmp_path / "99999.jpg").write_bytes(PNG)          # no matching client
    (tmp_path / "notes.txt").write_bytes(b"hi")        # ignored (not an image ext)
    (tmp_path / "junk.png").write_bytes(b"not-image")  # image ext, non-image bytes
    pushed = []
    res = s.sync_folder(cx, str(tmp_path), push_fn=lambda e, b, c: pushed.append(e) or True)
    by_file = {r["file"]: r for r in res}
    assert by_file["21459.jpg"]["action"].startswith("synced") and by_file["21459.jpg"]["email"] == "jc@x.com"
    assert cp.get(cx, "jc@x.com")["content_type"] == "image/png"      # sniffed, not ext
    assert by_file["bob@x.com.png"]["action"].startswith("synced")
    assert by_file["99999.jpg"]["action"] == "skip:no-email"
    assert by_file["junk.png"]["action"] == "skip:not-an-image"
    assert "notes.txt" not in by_file                                 # non-image ext skipped
    assert set(pushed) == {"jc@x.com", "bob@x.com"}


def test_sync_folder_respects_precedence(tmp_path):
    cx = _db_with_client("21459", "jc@x.com")
    cp.put(cx, "jc@x.com", b"client-own", "image/png", source="portal-self")
    (tmp_path / "21459.jpg").write_bytes(JPEG)
    res = s.sync_folder(cx, str(tmp_path), push_fn=lambda e, b, c: True)
    assert res[0]["action"] == "skip:precedence"
    assert cp.get(cx, "jc@x.com")["blob"] == b"client-own"           # untouched


def test_sync_folder_dry_does_not_write(tmp_path):
    cx = _db_with_client("21459", "jc@x.com")
    (tmp_path / "21459.jpg").write_bytes(PNG)
    res = s.sync_folder(cx, str(tmp_path), push_fn=None, write=False)
    assert res[0]["action"] == "would-sync"
    assert cp.get(cx, "jc@x.com") is None                            # nothing written
