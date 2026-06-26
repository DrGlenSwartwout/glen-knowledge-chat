# Portal PDF One-Click Download — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Serve portal-asset PDFs with `Content-Disposition: attachment` so the portal "Download report" link saves in one click (cross-origin), while audio stays inline.

**Architecture:** One conditional in the existing `/portal-asset/<filename>` serve route in `app.py`.

**Tech Stack:** Python 3.11, Flask 3.1 (`send_from_directory` supports `as_attachment` + `download_name`).

## Global Constraints

- `.pdf` → `as_attachment=True` + `download_name="Biofield-Analysis.pdf"`. `.mp3` → inline, unchanged (must NOT be attachment, or the `<audio>` player breaks).
- Only the serve route changes. No portal-page/connector/upload-route/URL change.
- `app.py` can't import offline → verified live (curl headers) post-deploy.

---

### Task 1: Serve PDFs as attachment

**Files:**
- Modify: `app.py` — `portal_asset_serve` (`/portal-asset/<filename>`, ~line 16092).

- [ ] **Step 1: Edit the route**

Current:
```python
@app.route("/portal-asset/<filename>")
def portal_asset_serve(filename):
    m = re.match(_PORTAL_ASSET_RE, filename)
    if not m:
        return jsonify({"error": "invalid filename"}), 400
    return send_from_directory(str(_PORTAL_ASSETS_DIR), filename,
                               mimetype=_PORTAL_ASSET_MIME[m.group(1)])
```
Change the `return` to make PDFs download:
```python
@app.route("/portal-asset/<filename>")
def portal_asset_serve(filename):
    m = re.match(_PORTAL_ASSET_RE, filename)
    if not m:
        return jsonify({"error": "invalid filename"}), 400
    is_pdf = m.group(1) == "pdf"
    return send_from_directory(
        str(_PORTAL_ASSETS_DIR), filename,
        mimetype=_PORTAL_ASSET_MIME[m.group(1)],
        as_attachment=is_pdf,
        download_name=("Biofield-Analysis.pdf" if is_pdf else None))
```
(Flask treats `download_name=None` as "use the path's filename" — harmless for the mp3 inline case, where `as_attachment=False` keeps it inline regardless.)

- [ ] **Step 2: Parse-check**

Run: `~/.venvs/deploy-chat311/bin/python -c "import ast; ast.parse(open('app.py').read()); print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(portal-media): serve report PDF as attachment (one-click download)"
```

- [ ] **Step 4: Live verification (post-deploy — record commands in report)**

```bash
# PDF -> Content-Disposition: attachment + application/pdf
curl -sI "https://glen-knowledge-chat.onrender.com/portal-asset/<an-existing>.pdf" | grep -iE "content-disposition|content-type"
# MP3 -> NO attachment, still audio/mpeg
curl -sI "https://glen-knowledge-chat.onrender.com/portal-asset/<an-existing>.mp3" | grep -iE "content-disposition|content-type"
```
Then on Karin's portal: the PDF link downloads in one click; the audio still plays inline.

---

## Self-Review

**1. Spec coverage:** PDF→attachment + download_name, mp3→inline unchanged → Task 1. ✅
**2. Placeholder scan:** No TBD; complete code; live-verify commands concrete (the `<an-existing>` is a real asset filename to fill at go-live, not a code placeholder). ✅
**3. Type consistency:** single route; `m.group(1)` already used for the mimetype; `is_pdf` derived from it. ✅
