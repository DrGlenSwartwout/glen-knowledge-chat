"""Local cataloging helpers (run on Glen's Mac, not in the deployed app).

transcribe() runs Whisper over a recording; suggest_catalog() asks an LLM for a
title, interest tags, and out-take moments from the transcript. Both accept an
injectable client for tests and use openai.OpenAI() otherwise (pattern:
dashboard/fmp_biofield.py)."""

import json
import os

_SUGGEST_SYSTEM = (
    "You catalog a recorded health coaching or course session for a members "
    "community. Given the transcript, return STRICT JSON with keys: "
    "title (short, warm, no em dashes, no ALL CAPS), "
    "interest_tags (3-7 lowercase topic tags), "
    "outtakes (2-4 objects, each {start, end, title, reason}) picking short "
    "self-contained highlight moments that tease the full session. "
    "start/end are seconds. Only JSON."
)


def _client(client):
    if client is not None:
        return client
    import openai
    return openai.OpenAI()


def transcribe(audio_path, *, client=None):
    """Whisper transcription → {"text", "segments":[{"start","end","text"}]}."""
    c = _client(client)
    with open(audio_path, "rb") as fh:
        r = c.audio.transcriptions.create(model="whisper-1", file=fh,
                                          response_format="verbose_json")
    segs = []
    for s in (getattr(r, "segments", None) or []):
        # segments may be dicts or objects depending on client version
        get = (lambda k: s[k]) if isinstance(s, dict) else (lambda k: getattr(s, k))
        segs.append({"start": float(get("start")), "end": float(get("end")),
                     "text": get("text")})
    return {"text": getattr(r, "text", "") or "", "segments": segs}


def suggest_catalog(transcript_text, *, client=None):
    """LLM → {"title","interest_tags","outtakes"}; degrades to empty on any error."""
    empty = {"title": "", "interest_tags": [], "outtakes": []}
    try:
        c = _client(client)
        r = c.chat.completions.create(
            model=os.environ.get("COMMUNITY_CATALOG_MODEL", "gpt-4o"),
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": _SUGGEST_SYSTEM},
                      {"role": "user", "content": transcript_text[:120000]}])
        data = json.loads(r.choices[0].message.content)
        interest_tags = data.get("interest_tags", [])
        outtakes = data.get("outtakes", [])
        return {"title": data.get("title", "") or "",
                "interest_tags": interest_tags if isinstance(interest_tags, list) else [],
                "outtakes": outtakes if isinstance(outtakes, list) else []}
    except Exception as e:
        print(f"[community-catalog] suggest failed: {e!r}", flush=True)
        return empty


def cut_outtakes(src_path, outtakes, *, workdir, trimmer=None):
    """Cut each approved out-take range into workdir/outtake-<i>.mp4. Returns a list
    of {title, interest_tags, path}. trimmer injectable (defaults to video_trim)."""
    import os as _os
    if trimmer is None:
        from dashboard.video_trim import trim_video as trimmer
    results = []
    for i, ot in enumerate(outtakes):
        dst = _os.path.join(workdir, f"outtake-{i}.mp4")
        trimmer(src_path, dst, float(ot["start"]), float(ot["end"]))
        results.append({"title": ot.get("title", ""),
                        "interest_tags": list(ot.get("interest_tags", []) or []),
                        "path": dst})
    return results


def publish_session(*, base_url, console_key, full, outtake_files,
                    uploader=None, poster=None):
    """Upload each out-take clip file, then POST the catalog entry to the prod
    publish endpoint. Returns the response JSON. uploader/poster injectable."""
    import os as _os
    if uploader is None:
        from dashboard.biofield_portal_publish import upload_asset as uploader
    if poster is None:
        import requests
        poster = requests.post
    outtakes = []
    for f in outtake_files:
        with open(f["path"], "rb") as fh:
            data = fh.read()
        filename = _os.path.basename(f["path"])
        url = uploader(data, filename, base_url=base_url, console_key=console_key)
        outtakes.append({"title": f["title"], "interest_tags": f["interest_tags"],
                         "video_ref": url})
    body = {"type": full["type"], "title": full["title"],
            "description": full.get("description", ""), "video_ref": full["video_ref"],
            "interest_tags": full.get("interest_tags", []),
            "transcript": full.get("transcript", ""), "outtakes": outtakes}
    r = poster(f"{base_url.rstrip('/')}/api/console/community/publish",
               json=body, headers={"X-Console-Key": console_key}, timeout=60)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"publish failed {r.status_code}")
    return r.json()
