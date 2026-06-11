"""Tests for chat attachment handling (images + PDF documents).

Attachment bytes are forwarded to Claude for one OCR/extraction pass and then
discarded; only the extracted text + a count are persisted. These tests cover
the pure normalization helper, the extraction-call shaping, and the route
consent gate.
"""

import base64

import pytest


def _img_data_url(media="image/png", nbytes=10):
    raw = b"\x00" * nbytes
    return f"data:{media};base64," + base64.b64encode(raw).decode()


def _pdf_data_url(nbytes=10):
    raw = b"%PDF-1.4\n" + b"\x00" * nbytes
    return "data:application/pdf;base64," + base64.b64encode(raw).decode()


def test_normalize_images_unchanged_shape():
    import app as app_module
    blocks, errors = app_module._normalize_attachments(
        [{"data_url": _img_data_url()}], []
    )
    assert errors == []
    assert blocks == [{
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png",
                   "data": blocks[0]["source"]["data"]},
    }]


def test_normalize_pdf_document_block():
    import app as app_module
    blocks, errors = app_module._normalize_attachments(
        [], [{"data_url": _pdf_data_url()}]
    )
    assert errors == []
    assert blocks[0]["type"] == "document"
    assert blocks[0]["source"]["type"] == "base64"
    assert blocks[0]["source"]["media_type"] == "application/pdf"


def test_normalize_rejects_non_pdf_document():
    import app as app_module
    blocks, errors = app_module._normalize_attachments(
        [], [{"data_url": _img_data_url(media="image/png")}]
    )
    assert blocks == []
    assert any("application/pdf" in e for e in errors)


def test_normalize_rejects_disallowed_image_media():
    import app as app_module
    blocks, errors = app_module._normalize_attachments(
        [{"data_url": _img_data_url(media="image/tiff")}], []
    )
    assert blocks == []
    assert any("not allowed" in e for e in errors)


def test_normalize_caps_image_count_at_three():
    import app as app_module
    imgs = [{"data_url": _img_data_url()} for _ in range(5)]
    blocks, _ = app_module._normalize_attachments(imgs, [])
    assert len([b for b in blocks if b["type"] == "image"]) == 3


def test_normalize_caps_doc_count_at_two():
    import app as app_module
    docs = [{"data_url": _pdf_data_url()} for _ in range(4)]
    blocks, _ = app_module._normalize_attachments([], docs)
    assert len([b for b in blocks if b["type"] == "document"]) == 2


def test_normalize_rejects_oversize_pdf():
    import app as app_module
    # 11 MB raw → over the 10 MB doc cap
    big = "data:application/pdf;base64," + ("A" * (11 * 1024 * 1024 * 4 // 3 + 8))
    blocks, errors = app_module._normalize_attachments([], [big])
    assert blocks == []
    assert any("10 MB" in e for e in errors)


def test_normalize_combined_cap():
    import app as app_module
    # Each item is under its own per-file cap, but together they exceed the
    # ~25 MB combined base64 budget. 3 images (~6.9M b64 each) + 1 PDF (~13M
    # b64) fit (~33.7M < ~35M); the 2nd PDF trips the combined cap.
    img = "data:image/png;base64," + ("A" * 6_900_000)
    pdf = "data:application/pdf;base64," + ("A" * 13_000_000)
    blocks, errors = app_module._normalize_attachments([img, img, img], [pdf, pdf])
    assert len(blocks) == 4  # 3 images + 1 document
    assert any("combined" in e for e in errors)


def test_normalize_image_payload_back_compat():
    import app as app_module
    blocks, errors = app_module._normalize_image_payload([{"data_url": _img_data_url()}])
    assert errors == []
    assert blocks[0]["type"] == "image"


def test_extract_forwards_image_and_document_blocks(monkeypatch):
    import app as app_module

    captured = {}

    class _FakeContent:
        text = "Attachment 1: Vitamin D 5000 IU"

    class _FakeResp:
        content = [_FakeContent()]

    def _fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        captured["model"] = kwargs["model"]
        return _FakeResp()

    monkeypatch.setattr(app_module._cl.messages, "create", _fake_create)

    blocks = [
        {"type": "image", "source": {"type": "base64",
         "media_type": "image/png", "data": "AAAA"}},
        {"type": "document", "source": {"type": "base64",
         "media_type": "application/pdf", "data": "BBBB"}},
    ]
    out = app_module.extract_attachment_content(blocks, "what is in these?")

    assert out == "Attachment 1: Vitamin D 5000 IU"
    sent = captured["messages"][0]["content"]
    # both attachment blocks forwarded, plus the trailing instruction text
    assert sent[0]["type"] == "image"
    assert sent[1]["type"] == "document"
    assert sent[-1]["type"] == "text"


def test_extract_empty_blocks_returns_empty():
    import app as app_module
    assert app_module.extract_attachment_content([], "q") == ""


@pytest.fixture
def client():
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_chat_documents_without_consent_is_400(client):
    r = client.post("/chat", json={
        "query": "read this",
        "documents": ["data:application/pdf;base64,AAAA"],
        "images_consented": False,
    })
    assert r.status_code == 400
    assert b"consent" in r.data.lower()
