# Chat Document & Image OCR Upload — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let visitors attach images and PDF documents to every live chat surface and have their content OCR-extracted so the assistant can answer about lab results, supplement labels, E4L scan PDFs, and photos.

**Architecture:** Generalize the existing image plumbing in `app.py` into one attachment helper (`_normalize_attachments`) that also emits Claude `document` blocks for PDFs, and one extraction pass (`extract_attachment_content`) over images + PDF pages. Wire it into all three live chat routes. On the frontend, lift the upload UI out of the retired `index.html` into a shared, mountable module (`static/chat-attachments.js`) included on `embed.html`, `concierge.html`, and `begin-match.html`. Attachment bytes are sent to Claude once for extraction then discarded; only extracted text + a count are persisted.

**Tech Stack:** Python/Flask, Anthropic SDK (`claude-haiku-4-5-20251001`, native PDF `document` blocks, no beta header), vanilla JS (shared-module pattern like `op-nav.js`/`mic-input.js`/`tts-output.js`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-chat-document-image-ocr-design.md`

**Work location:** worktree `/tmp/wt-deploy-chat-b4521cf9` on branch `sess/b4521cf9`. Run all commands from there: `cd /tmp/wt-deploy-chat-b4521cf9`.

---

## File Structure

- `app.py`
  - **Modify** `_normalize_image_payload` (lines ~661-713) → add `_normalize_attachments(images, documents)`; keep the old name as a thin wrapper.
  - **Modify** `extract_image_content` (lines ~716-749) → add `extract_attachment_content(blocks, query)`; keep the old name as a wrapper.
  - **Modify** `/chat` route attachment block (lines ~1545-1576, 1590, 1652-1659) to also accept `documents`.
  - **Modify** `/begin/match/chat` route (lines ~1904-1979) to add the consent gate + extraction + context injection.
  - **Modify** `/begin/concierge/chat` route (lines ~2712-2748) to add the consent gate + extraction + context injection (retrieval moves inside `generate()`).
- `static/chat-attachments.js` — **Create.** Shared, mountable upload UI + payload getter.
- `static/embed.html`, `static/concierge.html`, `static/begin-match.html` — **Modify.** Include the module, mount it, merge its payload into the chat POST body.
- `tests/test_chat_attachments.py` — **Create.** Unit tests for the helper + extraction + consent gate.
- `static/index.html` — **left unchanged** (retired).

---

## Task 1: `_normalize_attachments` helper (images + PDFs)

**Files:**
- Modify: `app.py` (replace `_normalize_image_payload` at ~661-713)
- Test: `tests/test_chat_attachments.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat_attachments.py`:

```python
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
    # Two 9 MB PDFs (each under the 10 MB per-file cap) exceed the ~25 MB
    # combined base64 budget on the second item once both are counted? No —
    # 9+9 = 18 MB < 25 MB, both pass. Use larger to trip the combined cap.
    nine_and_a_half = "data:application/pdf;base64," + ("A" * (int(9.5 * 1024 * 1024) * 4 // 3))
    blocks, errors = app_module._normalize_attachments(
        [], [nine_and_a_half, nine_and_a_half]
    )
    # first fits, second trips the combined cap
    assert len(blocks) == 1
    assert any("combined" in e for e in errors)


def test_normalize_image_payload_back_compat():
    import app as app_module
    blocks, errors = app_module._normalize_image_payload([{"data_url": _img_data_url()}])
    assert errors == []
    assert blocks[0]["type"] == "image"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -m pytest tests/test_chat_attachments.py -v`
Expected: FAIL — `_normalize_attachments` does not exist yet (AttributeError).

- [ ] **Step 3: Implement `_normalize_attachments`**

In `app.py`, replace the entire `_normalize_image_payload` function (lines ~661-713) with:

```python
def _normalize_attachments(images, documents):
    """Build Anthropic content blocks from user-attached images and PDFs.

    Each entry (in either list) may be:
      - "data:<media>;base64,<b64>"           (string form)
      - {"data_url": "data:<media>;base64,…"}  (dict form)
      - {"data": "<b64>", "media_type": "…"}   (explicit form)

    Images become image blocks (unchanged behavior). Documents become PDF
    `document` blocks. Bytes are forwarded to Claude for one extraction pass
    and never persisted. Returns (blocks, errors). Caps:
      - images: 3 max, ~5 MB raw (~6.7 MB base64) each, png/jpeg/webp/gif
      - documents: 2 max, ~10 MB raw (~13.3 MB base64) each, application/pdf
      - combined: total base64 length across all attachments ≤ ~25 MB raw,
        keeping the whole request under Claude's 32 MB limit
    """
    MAX_IMAGES = 3
    MAX_IMAGE_B64 = 5 * 1024 * 1024 * 4 // 3
    MAX_DOCS = 2
    MAX_DOC_B64 = 10 * 1024 * 1024 * 4 // 3
    MAX_TOTAL_B64 = 25 * 1024 * 1024 * 4 // 3
    IMG_ALLOWED = ("image/png", "image/jpeg", "image/webp", "image/gif")

    blocks, errors = [], []
    state = {"total": 0}

    def _decode(entry):
        if isinstance(entry, str):
            if entry.startswith("data:") and ";base64," in entry:
                head, b64 = entry.split(";base64,", 1)
                return head[5:], b64
            raise ValueError("unsupported string format")
        if isinstance(entry, dict) and entry.get("data_url"):
            d = entry["data_url"]
            if d.startswith("data:") and ";base64," in d:
                head, b64 = d.split(";base64,", 1)
                return head[5:], b64
            raise ValueError("bad data_url")
        if isinstance(entry, dict) and entry.get("data"):
            return entry.get("media_type", "image/png"), entry["data"]
        raise ValueError("unrecognized payload shape")

    for i, entry in enumerate((images or [])[:MAX_IMAGES]):
        try:
            media, b64 = _decode(entry)
        except ValueError as e:
            errors.append(f"image[{i}]: {e}"); continue
        if media not in IMG_ALLOWED:
            errors.append(f"image[{i}]: media_type {media!r} not allowed"); continue
        if len(b64) > MAX_IMAGE_B64:
            errors.append(f"image[{i}]: exceeds 5 MB size limit"); continue
        if state["total"] + len(b64) > MAX_TOTAL_B64:
            errors.append(f"image[{i}]: combined attachment size limit exceeded"); continue
        state["total"] += len(b64)
        blocks.append({"type": "image",
                       "source": {"type": "base64", "media_type": media, "data": b64}})

    for i, entry in enumerate((documents or [])[:MAX_DOCS]):
        try:
            media, b64 = _decode(entry)
        except ValueError as e:
            errors.append(f"document[{i}]: {e}"); continue
        if media != "application/pdf":
            errors.append(f"document[{i}]: only application/pdf is accepted"); continue
        if len(b64) > MAX_DOC_B64:
            errors.append(f"document[{i}]: exceeds 10 MB size limit"); continue
        if state["total"] + len(b64) > MAX_TOTAL_B64:
            errors.append(f"document[{i}]: combined attachment size limit exceeded"); continue
        state["total"] += len(b64)
        blocks.append({"type": "document",
                       "source": {"type": "base64",
                                  "media_type": "application/pdf", "data": b64}})

    return blocks, errors


def _normalize_image_payload(images):
    """Back-compat wrapper — images only. Prefer _normalize_attachments."""
    return _normalize_attachments(images, [])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -m pytest tests/test_chat_attachments.py -v`
Expected: PASS (all 9 tests).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-b4521cf9
git add app.py tests/test_chat_attachments.py
git commit -m "feat(chat): _normalize_attachments — images + PDF document blocks"
```

---

## Task 2: `extract_attachment_content` (one OCR pass over images + PDFs)

**Files:**
- Modify: `app.py` (replace `extract_image_content` at ~716-749)
- Test: `tests/test_chat_attachments.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chat_attachments.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -m pytest tests/test_chat_attachments.py::test_extract_forwards_image_and_document_blocks -v`
Expected: FAIL — `extract_attachment_content` not defined.

- [ ] **Step 3: Implement `extract_attachment_content`**

In `app.py`, replace the entire `extract_image_content` function (lines ~716-749) with:

```python
def extract_attachment_content(blocks, query):
    """Single non-streaming Claude pass to extract structured text from attached
    images and/or PDF documents. Returns the extraction string. Attachment bytes
    are NOT persisted — they exist only in this function's call to Anthropic.
    """
    if not blocks:
        return ""
    instr = (
        "Extract everything visible in these attachments as plain text. Each "
        "attachment may be an image or a multi-page PDF document. Focus on:\n"
        "• Any text, labels, headings, captions\n"
        "• Numbers, measurements, dosages, lab values, ranges\n"
        "• Supplement ingredient lists, milligram amounts, serving sizes\n"
        "• Lab/test result values with units and reference ranges if present\n"
        "• E4L scan results: item codes (EI/ES/ED/ET/MB), category labels, scores\n"
        "• Any visible chart axes, legend entries, or graph markers\n"
        "• Visible symptoms in clinical photos (describe objectively)\n"
        "• Handwritten notes (transcribe carefully)\n\n"
        f"USER'S QUESTION: {query}\n\n"
        "Return a clean, structured extraction. Label each attachment "
        "(Attachment 1, Attachment 2, …) and each PDF page if multiple. Do not "
        "analyze, diagnose, or recommend — just extract. Be exhaustive but concise."
    )
    try:
        resp = _cl.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": [
                *blocks,
                {"type": "text", "text": instr},
            ]}],
        )
        return (resp.content[0].text or "").strip() if resp.content else ""
    except Exception as e:
        return f"[attachment-extraction-error: {e}]"


def extract_image_content(image_blocks, query):
    """Back-compat wrapper. Prefer extract_attachment_content."""
    return extract_attachment_content(image_blocks, query)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -m pytest tests/test_chat_attachments.py -v`
Expected: PASS (all 11 tests).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-b4521cf9
git add app.py tests/test_chat_attachments.py
git commit -m "feat(chat): extract_attachment_content — OCR pass over images + PDF pages"
```

---

## Task 3: Wire `/chat` route to accept documents

**Files:**
- Modify: `app.py` (`/chat` route, lines ~1545-1576, ~1590, ~1652-1659)
- Test: `tests/test_chat_attachments.py`

- [ ] **Step 1: Write the failing test (consent gate covers documents)**

Append to `tests/test_chat_attachments.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -m pytest tests/test_chat_attachments.py::test_chat_documents_without_consent_is_400 -v`
Expected: FAIL — `/chat` currently ignores `documents`, so no 400 is returned (it proceeds toward embedding).

- [ ] **Step 3: Implement document acceptance in `/chat`**

In `app.py`, replace the image block (lines ~1545-1558) with:

```python
    # Attachments — opt-in gated images + PDFs, extraction-only storage.
    # Bytes are passed to Claude for extraction then discarded; only the
    # extracted text is persisted to query_log.
    images_consented = bool(data.get("images_consented"))
    raw_images = data.get("images") or []
    raw_documents = data.get("documents") or []
    attachment_blocks = []
    attachment_errors = []
    if raw_images or raw_documents:
        if not images_consented:
            return jsonify({
                "error": "Attachment consent required. Check the consent box "
                         "before attaching documents or images."
            }), 400
        attachment_blocks, attachment_errors = _normalize_attachments(
            raw_images, raw_documents)
```

In the same route, update the `generate()` body. Replace lines ~1567-1570:

```python
        extracted_text = ""
        if attachment_blocks:
            yield sse({"status": f"Reading {len(attachment_blocks)} attachment(s)…"})
            extracted_text = extract_attachment_content(attachment_blocks, query)
```

Replace the embedding-label line (~1576):

```python
            embedding_input = f"{query}\n\nATTACHMENT CONTENT:\n{extracted_text}"
```

Replace the `image_count` in the no-matches early return (~1590):

```python
                       "image_count": len(attachment_blocks)})
```

Replace the `image_context` block (~1652-1659) with:

```python
        image_context = ""
        if extracted_text:
            image_context = (
                f"ATTACHMENT CONTENT EXTRACTED FROM USER UPLOAD(S):\n"
                f"{extracted_text}\n\n"
                f"Reference the attachment content as part of the user's question "
                f"context. Quote specific values or labels from it when relevant.\n\n"
            )
```

Then find the later use of `len(image_blocks)` for logging/`image_count` in this route (search the route for `image_blocks` and `image_count`) and change `image_blocks` → `attachment_blocks`. Verify none remain:

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && awk 'NR>=1426 && NR<=1812' app.py | grep -n "image_blocks" || echo "clean"`
Expected: `clean`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -m pytest tests/test_chat_attachments.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-b4521cf9
git add app.py tests/test_chat_attachments.py
git commit -m "feat(chat): /chat route accepts PDF documents alongside images"
```

---

## Task 4: Wire `/begin/match/chat` route

**Files:**
- Modify: `app.py` (`begin_match_chat`, lines ~1904-1979)
- Test: `tests/test_chat_attachments.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chat_attachments.py`:

```python
def test_match_documents_without_consent_is_400(client):
    r = client.post("/begin/match/chat", json={
        "query": "match me from this scan",
        "documents": ["data:application/pdf;base64,AAAA"],
        "images_consented": False,
    })
    assert r.status_code == 400
    assert b"consent" in r.data.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -m pytest tests/test_chat_attachments.py::test_match_documents_without_consent_is_400 -v`
Expected: FAIL — the match route ignores `documents` and returns 200/stream.

- [ ] **Step 3: Implement attachment handling in `begin_match_chat`**

In `app.py`, in `begin_match_chat`, after the `auth_user` block and before `if not query:` (i.e. after line ~1920), insert:

```python
    images_consented = bool(data.get("images_consented"))
    raw_images = data.get("images") or []
    raw_documents = data.get("documents") or []
    attachment_blocks = []
    if raw_images or raw_documents:
        if not images_consented:
            return jsonify({
                "error": "Attachment consent required. Check the consent box "
                         "before attaching documents or images."
            }), 400
        attachment_blocks, _ = _normalize_attachments(raw_images, raw_documents)
```

Then replace the start of `generate()` (the `try: q_vec = embed(query)` block, lines ~1924-1929) with:

```python
    def generate():
        extracted_text = ""
        if attachment_blocks:
            yield sse({"status": f"Reading {len(attachment_blocks)} attachment(s)…"})
            extracted_text = extract_attachment_content(attachment_blocks, query)
        emb_input = (f"{query}\n\nATTACHMENT CONTENT:\n{extracted_text}"
                     if extracted_text else query)
        try:
            q_vec = embed(emb_input)
        except Exception as e:
            yield sse({"error": f"Embedding failed: {e}"}); return
        matches = _match_query_namespaces(q_vec)
        context_str, sources_list = build_context(matches) if matches else ("", [])
```

Then inject the extracted text into the user message. Replace the `messages.append({...})` block (lines ~1974-1979) with:

```python
        attach_block = (f"ATTACHMENT CONTENT (from the person's uploaded files; "
                        f"quote specific values when relevant):\n{extracted_text}\n\n"
                        if extracted_text else "")
        messages.append({"role": "user", "content":
            f"USER MESSAGE: {query}\n\n{whom_line}\n{household_note}\n{personal_block}"
            f"{tools_block}{attach_block}"
            f"RETRIEVED SNIPPETS:\n{context_str}\n\n"
            "Continue the Socratic match. If you can now name the ONE best remedy, name it and "
            "invite them to open its page; otherwise ask the single best next question."})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -m pytest tests/test_chat_attachments.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-b4521cf9
git add app.py tests/test_chat_attachments.py
git commit -m "feat(match): /begin/match/chat accepts document + image attachments"
```

---

## Task 5: Wire `/begin/concierge/chat` route

**Files:**
- Modify: `app.py` (`begin_concierge_chat`, lines ~2712-2748)
- Test: `tests/test_chat_attachments.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chat_attachments.py`:

```python
def test_concierge_documents_without_consent_is_400(client):
    r = client.post("/begin/concierge/chat", json={
        "query": "what pairs with this label",
        "documents": ["data:application/pdf;base64,AAAA"],
        "images_consented": False,
    })
    assert r.status_code == 400
    assert b"consent" in r.data.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -m pytest tests/test_chat_attachments.py::test_concierge_documents_without_consent_is_400 -v`
Expected: FAIL — concierge ignores `documents`.

- [ ] **Step 3: Implement attachment handling in `begin_concierge_chat`**

In `app.py`, in `begin_concierge_chat`, the current retrieval runs **before** `generate()` (lines ~2726-2735). It must move inside `generate()` so it can incorporate the extracted text. Replace lines ~2723-2748 (from `if not query:` through the first `messages.append({...})`) with:

```python
    if not query:
        return jsonify({"error": "Empty query"}), 400

    images_consented = bool(data.get("images_consented"))
    raw_images = data.get("images") or []
    raw_documents = data.get("documents") or []
    attachment_blocks = []
    if raw_images or raw_documents:
        if not images_consented:
            return jsonify({
                "error": "Attachment consent required. Check the consent box "
                         "before attaching documents or images."
            }), 400
        attachment_blocks, _ = _normalize_attachments(raw_images, raw_documents)

    # Pairing priors for what they bought.
    priors = (_PAIRINGS.get("pairings", {}) or {}).get(bought_slug, []) if bought_slug else []
    priors_block = (f"SUGGESTED COMPLEMENTS for {bought['name'] if bought else 'their purchase'} "
                    f"(offer these first, one at a time): {', '.join(priors)}\n\n") if priors else ""

    def generate():
        extracted_text = ""
        if attachment_blocks:
            yield sse({"status": f"Reading {len(attachment_blocks)} attachment(s)…"})
            extracted_text = extract_attachment_content(attachment_blocks, query)

        # A little RAG for rationale/benefits (now inside generate so the
        # extracted attachment text can sharpen retrieval).
        context_str = ""
        try:
            emb_q = query + " " + (bought["name"] if bought else "")
            if extracted_text:
                emb_q += "\n\nATTACHMENT CONTENT:\n" + extracted_text
            matches = _match_query_namespaces(embed(emb_q))
            context_str, _ = build_context(matches) if matches else ("", [])
        except Exception as e:
            print(f"[concierge] retrieval: {e}", flush=True)

        attach_block = (f"ATTACHMENT CONTENT (from the person's uploaded files; "
                        f"quote specific values when relevant):\n{extracted_text}\n\n"
                        if extracted_text else "")
        messages = []
        for turn in history[-8:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content":
            f"THEY JUST BOUGHT: {bought['name'] if bought else 'a remedy'}.\n"
            f"{priors_block}{attach_block}"
            f"RETRIEVED SNIPPETS (for rationale/benefits):\n{context_str}\n\n"
            f"MEMBER MESSAGE: {query}\n\n"
            "Continue as the concierge: affirm, ask the single best next question, or suggest ONE "
            "complement with a short why. Keep it warm and brief."})
```

Note: this removes the old pre-`generate()` retrieval block (the `context_str = ""` / `try: matches = …` at ~2730-2735) — it is now inside `generate()`. Verify the old block is gone:

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && awk 'NR>=2712 && NR<=2760' app.py | grep -c "matches = _match_query_namespaces"`
Expected: `1` (only the one now inside `generate()`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -m pytest tests/test_chat_attachments.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-b4521cf9
git add app.py tests/test_chat_attachments.py
git commit -m "feat(concierge): /begin/concierge/chat accepts document + image attachments"
```

---

## Task 6: Shared frontend module `static/chat-attachments.js`

**Files:**
- Create: `static/chat-attachments.js`

This module is self-contained: it renders its own UI into a host element, holds pending files in memory only, enforces client-side caps mirroring the backend, and exposes a payload getter. No test framework runs JS here; correctness is verified by the manual smoke test in Task 10.

- [ ] **Step 1: Create the module**

Create `static/chat-attachments.js`:

```javascript
/* chat-attachments.js — shared document + image upload for the chat surfaces.
 *
 * Usage:
 *   <div id="chat-attach"></div>
 *   <script src="/static/chat-attachments.js"></script>
 *   <script>
 *     const attach = ChatAttach.mount({ host: '#chat-attach', dropZone: '#input-bar' });
 *     // when sending: const { images, documents, consented } = attach.getPayload();
 *     // after a successful send: attach.clear();
 *   </script>
 *
 * Files are held in memory only (never localStorage). Consent is the single
 * boolean persisted under 'amg_images_consented'. The backend re-validates
 * everything; these caps are UX guardrails.
 */
(function (global) {
  'use strict';

  var MAX_IMAGES = 3;
  var MAX_DOCS = 2;
  var MAX_IMAGE_BYTES = 5 * 1024 * 1024;
  var MAX_DOC_BYTES = 10 * 1024 * 1024;
  var IMG_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'];
  var CONSENT_KEY = 'amg_images_consented';

  var STYLE = [
    '.ca-wrap{font:inherit;margin:6px 0;}',
    '.ca-consent{display:flex;align-items:flex-start;gap:7px;font-size:12px;opacity:.85;line-height:1.35;}',
    '.ca-consent input{margin-top:2px;}',
    '.ca-row{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-top:6px;}',
    '.ca-btn{cursor:pointer;border:1px solid currentColor;background:transparent;color:inherit;',
    'border-radius:6px;padding:4px 10px;font-size:13px;opacity:.9;}',
    '.ca-btn:hover{opacity:1;}',
    '.ca-help{font-size:12px;opacity:.6;}',
    '.ca-err{font-size:12px;color:#c0392b;}',
    '.ca-chip{display:inline-flex;align-items:center;gap:6px;border:1px solid rgba(128,128,128,.4);',
    'border-radius:6px;padding:2px 6px;font-size:12px;max-width:180px;}',
    '.ca-chip img{width:26px;height:26px;object-fit:cover;border-radius:3px;}',
    '.ca-chip .ca-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}',
    '.ca-chip .ca-x{cursor:pointer;border:none;background:transparent;color:inherit;font-size:15px;line-height:1;}'
  ].join('');

  function injectStyleOnce() {
    if (document.getElementById('ca-style')) return;
    var s = document.createElement('style');
    s.id = 'ca-style';
    s.textContent = STYLE;
    document.head.appendChild(s);
  }

  function readAsDataURL(file) {
    return new Promise(function (resolve, reject) {
      var r = new FileReader();
      r.onload = function () { resolve(r.result); };
      r.onerror = function () { reject(r.error); };
      r.readAsDataURL(file);
    });
  }

  function mount(opts) {
    opts = opts || {};
    var host = typeof opts.host === 'string'
      ? document.querySelector(opts.host) : opts.host;
    if (!host) throw new Error('ChatAttach.mount: host not found');
    injectStyleOnce();

    var pending = []; // { data_url, name, kind: 'image'|'document' }

    host.classList.add('ca-wrap');
    host.innerHTML =
      '<label class="ca-consent">' +
        '<input type="checkbox" class="ca-consent-cb">' +
        '<span>Allow attaching documents and images (lab results, supplement ' +
        'labels, scan PDFs, photos). Content is extracted as text to answer ' +
        'your question; the original file is not saved.</span>' +
      '</label>' +
      '<div class="ca-row">' +
        '<button type="button" class="ca-btn ca-add">+ Add file</button>' +
        '<span class="ca-help">images or PDF · or drag-drop</span>' +
        '<span class="ca-err"></span>' +
        '<input type="file" class="ca-input" multiple style="display:none" ' +
          'accept="image/png,image/jpeg,image/webp,image/gif,application/pdf">' +
      '</div>';

    var cb = host.querySelector('.ca-consent-cb');
    var input = host.querySelector('.ca-input');
    var addBtn = host.querySelector('.ca-add');
    var errEl = host.querySelector('.ca-err');
    var row = host.querySelector('.ca-row');

    try { cb.checked = localStorage.getItem(CONSENT_KEY) === 'true'; } catch (e) {}
    cb.addEventListener('change', function () {
      try { localStorage.setItem(CONSENT_KEY, cb.checked ? 'true' : 'false'); } catch (e) {}
    });

    function consented() { return !!cb.checked; }

    function setError(msg) {
      errEl.textContent = msg || '';
      if (msg) setTimeout(function () {
        if (errEl.textContent === msg) errEl.textContent = '';
      }, 4000);
    }

    function refresh() {
      Array.prototype.slice.call(host.querySelectorAll('.ca-chip'))
        .forEach(function (el) { el.remove(); });
      pending.forEach(function (f, idx) {
        var chip = document.createElement('span');
        chip.className = 'ca-chip';
        var thumb = f.kind === 'image'
          ? '<img src="' + f.data_url + '" alt="">'
          : '<span aria-hidden="true">📄</span>';
        chip.innerHTML = thumb +
          '<span class="ca-name">' + (f.name || 'file') + '</span>' +
          '<button type="button" class="ca-x" title="Remove">×</button>';
        chip.querySelector('.ca-x').addEventListener('click', function () {
          pending.splice(idx, 1); refresh();
        });
        row.insertBefore(chip, addBtn);
      });
    }

    function countKind(kind) {
      return pending.filter(function (f) { return f.kind === kind; }).length;
    }

    async function addFiles(fileList) {
      if (!fileList || !fileList.length) return;
      if (!consented()) { setError('Check the consent box first.'); return; }
      var files = Array.prototype.slice.call(fileList);
      for (var i = 0; i < files.length; i++) {
        var file = files[i];
        var isImage = file.type && IMG_TYPES.indexOf(file.type) !== -1;
        var isPdf = file.type === 'application/pdf';
        if (!isImage && !isPdf) { setError('Only images or PDF files are accepted.'); continue; }
        if (isImage && countKind('image') >= MAX_IMAGES) { setError('Max ' + MAX_IMAGES + ' images.'); continue; }
        if (isPdf && countKind('document') >= MAX_DOCS) { setError('Max ' + MAX_DOCS + ' PDFs.'); continue; }
        if (isImage && file.size > MAX_IMAGE_BYTES) { setError('"' + file.name + '" exceeds the 5 MB image limit.'); continue; }
        if (isPdf && file.size > MAX_DOC_BYTES) { setError('"' + file.name + '" exceeds the 10 MB PDF limit.'); continue; }
        try {
          var data_url = await readAsDataURL(file);
          pending.push({ data_url: data_url, name: file.name, kind: isImage ? 'image' : 'document' });
        } catch (e) { setError('Could not read "' + file.name + '".'); }
      }
      refresh();
    }

    addBtn.addEventListener('click', function () { input.click(); });
    input.addEventListener('change', function () { addFiles(input.files); input.value = ''; });

    // Paste (images only — browsers expose pasted images, not PDFs).
    document.addEventListener('paste', function (e) {
      if (!e.clipboardData || !e.clipboardData.items) return;
      var imgs = Array.prototype.slice.call(e.clipboardData.items)
        .filter(function (it) { return it.kind === 'file' && it.type.indexOf('image/') === 0; });
      if (!imgs.length) return;
      if (!consented()) { setError('Check the consent box first.'); e.preventDefault(); return; }
      e.preventDefault();
      addFiles(imgs.map(function (it) { return it.getAsFile(); }).filter(Boolean));
    });

    // Drag-drop onto an optional drop zone.
    var zone = opts.dropZone
      ? (typeof opts.dropZone === 'string' ? document.querySelector(opts.dropZone) : opts.dropZone)
      : null;
    if (zone) {
      ['dragenter', 'dragover'].forEach(function (ev) {
        zone.addEventListener(ev, function (e) { e.preventDefault(); zone.style.outline = '2px dashed currentColor'; });
      });
      ['dragleave', 'drop'].forEach(function (ev) {
        zone.addEventListener(ev, function (e) { e.preventDefault(); zone.style.outline = ''; });
      });
      zone.addEventListener('drop', function (e) {
        if (e.dataTransfer && e.dataTransfer.files) addFiles(e.dataTransfer.files);
      });
    }

    return {
      getPayload: function () {
        return {
          images: pending.filter(function (f) { return f.kind === 'image'; })
            .map(function (f) { return { data_url: f.data_url }; }),
          documents: pending.filter(function (f) { return f.kind === 'document'; })
            .map(function (f) { return { data_url: f.data_url, name: f.name }; }),
          consented: consented()
        };
      },
      clear: function () { pending = []; refresh(); }
    };
  }

  global.ChatAttach = { mount: mount };
})(window);
```

- [ ] **Step 2: Syntax-check the module**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && node --check static/chat-attachments.js && echo OK`
Expected: `OK`. (If `node` is unavailable, skip — Task 10's browser smoke test covers it.)

- [ ] **Step 3: Commit**

```bash
cd /tmp/wt-deploy-chat-b4521cf9
git add static/chat-attachments.js
git commit -m "feat(ui): shared chat-attachments.js upload module (images + PDF)"
```

---

## Task 7: Mount the module on `embed.html`

**Files:**
- Modify: `static/embed.html`

- [ ] **Step 1: Find the chat input area and POST body**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && grep -nE "id=\"input-bar\"|fetch\(|/chat|JSON.stringify|body:" static/embed.html | head -30`
Expected: shows the input bar element id, the `fetch('/chat'...)` call, and the `JSON.stringify({...})` request body. Note the exact ids/variables for the next steps.

- [ ] **Step 2: Add the host element + script include**

Insert a host `<div id="chat-attach"></div>` immediately above the chat input bar element found in Step 1. Before the closing `</body>`, add:

```html
<script src="/static/chat-attachments.js"></script>
<script>
  window.__chatAttach = ChatAttach.mount({ host: '#chat-attach', dropZone: '#input-bar' });
</script>
```

(If the input bar uses a different id than `input-bar`, pass that id to `dropZone`. If no obvious drop target exists, omit `dropZone`.)

- [ ] **Step 3: Merge the payload into the chat POST body**

In the function that POSTs to `/chat`, just before the `JSON.stringify({...})` call, add:

```javascript
  var _att = (window.__chatAttach && window.__chatAttach.getPayload()) || {};
```

Add these keys inside the stringified body object:

```javascript
      images: _att.images || [],
      documents: _att.documents || [],
      images_consented: !!_att.consented,
```

After a successful send (where the page clears the input box), add:

```javascript
  if (window.__chatAttach) window.__chatAttach.clear();
```

- [ ] **Step 4: Commit**

```bash
cd /tmp/wt-deploy-chat-b4521cf9
git add static/embed.html
git commit -m "feat(ui): mount document+image upload on embed.html chat"
```

---

## Task 8: Mount the module on `concierge.html`

**Files:**
- Modify: `static/concierge.html`

- [ ] **Step 1: Find the chat input + POST body**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && grep -nE "id=\"input|fetch\(|/begin/concierge/chat|JSON.stringify|body:" static/concierge.html | head -30`
Expected: shows the input element id, the `fetch('/begin/concierge/chat'...)` call, and the request body object. Note the exact names.

- [ ] **Step 2: Add the host element + script include**

Insert `<div id="chat-attach"></div>` directly above the concierge chat input element. Before `</body>` add:

```html
<script src="/static/chat-attachments.js"></script>
<script>
  window.__chatAttach = ChatAttach.mount({ host: '#chat-attach' });
</script>
```

(Add `dropZone: '#<input-bar-id>'` if the page has an input-bar container.)

- [ ] **Step 3: Merge payload into the POST body**

Before the `JSON.stringify({...})` for `/begin/concierge/chat`, add:

```javascript
  var _att = (window.__chatAttach && window.__chatAttach.getPayload()) || {};
```

Add inside the body object:

```javascript
      images: _att.images || [],
      documents: _att.documents || [],
      images_consented: !!_att.consented,
```

After a successful send, add:

```javascript
  if (window.__chatAttach) window.__chatAttach.clear();
```

- [ ] **Step 4: Commit**

```bash
cd /tmp/wt-deploy-chat-b4521cf9
git add static/concierge.html
git commit -m "feat(ui): mount document+image upload on concierge.html chat"
```

---

## Task 9: Mount the module on `begin-match.html`

**Files:**
- Modify: `static/begin-match.html`

- [ ] **Step 1: Find the chat input + POST body**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && grep -nE "id=\"input|fetch\(|/begin/match/chat|JSON.stringify|body:" static/begin-match.html | head -30`
Expected: shows the input element id, the `fetch('/begin/match/chat'...)` call, and the request body object.

- [ ] **Step 2: Add the host element + script include**

Insert `<div id="chat-attach"></div>` directly above the match chat input element. Before `</body>` add:

```html
<script src="/static/chat-attachments.js"></script>
<script>
  window.__chatAttach = ChatAttach.mount({ host: '#chat-attach' });
</script>
```

(Add `dropZone` if an input-bar container exists.)

- [ ] **Step 3: Merge payload into the POST body**

Before the `JSON.stringify({...})` for `/begin/match/chat`, add:

```javascript
  var _att = (window.__chatAttach && window.__chatAttach.getPayload()) || {};
```

Add inside the body object:

```javascript
      images: _att.images || [],
      documents: _att.documents || [],
      images_consented: !!_att.consented,
```

After a successful send, add:

```javascript
  if (window.__chatAttach) window.__chatAttach.clear();
```

- [ ] **Step 4: Commit**

```bash
cd /tmp/wt-deploy-chat-b4521cf9
git add static/begin-match.html
git commit -m "feat(ui): mount document+image upload on begin-match.html chat"
```

---

## Task 10: Full suite + manual smoke test

**Files:** none (verification only)

- [ ] **Step 1: Run the whole backend test suite**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -m pytest -q`
Expected: all tests pass (including the new `tests/test_chat_attachments.py`). If unrelated pre-existing failures appear, note them and confirm they exist on `main` before this branch — do not fix out-of-scope failures here.

- [ ] **Step 2: Boot the app locally and smoke-test each surface**

Use the healthy venv (per the urllib3/SSL note in memory). Run the app:

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m flask --app app run -p 5050`
Then in a browser:
- Open `http://localhost:5050/embed` (or the embed host page), `/begin/concierge`, `/begin/match`.
- On each: confirm the consent checkbox + "+ Add file" appear, attach a small PDF and a JPEG, confirm chips render with a remove (×), then send a question like "what's in this file?".
- Confirm the answer references content from the file and that no error is shown.
- Open the served `/static/chat-attachments.js` in DevTools and confirm no console errors on load.

Expected: each surface accepts a PDF + image, the assistant answers using the extracted content, and the chips clear after send.

- [ ] **Step 3: Verify nothing persists the bytes**

Run: `cd /tmp/wt-deploy-chat-b4521cf9 && python -c "import sqlite3,os; db=os.path.expanduser('~/deploy-chat/chat_log.db'); cx=sqlite3.connect(db); print(cx.execute('select image_count, length(extracted_image_data) from query_log order by id desc limit 3').fetchall())"`
Expected: rows show a non-zero `image_count` (attachment count) and a text length for the extraction, confirming only text — never raw bytes — was stored.

- [ ] **Step 4: Final commit (if any smoke-test tweaks were needed)**

```bash
cd /tmp/wt-deploy-chat-b4521cf9
git add -A
git commit -m "test: smoke-test fixes for chat attachments" || echo "nothing to commit"
```

---

## Done criteria

- All three live chat surfaces show a consent + upload control and accept images **and** PDFs.
- Backend extracts text from both in one Haiku pass, feeds it into retrieval + the answer, and persists only extracted text + a count.
- `tests/test_chat_attachments.py` passes and the full suite is green.
- Branch `sess/b4521cf9` pushed; PR opened for Glen to review (do not merge).
