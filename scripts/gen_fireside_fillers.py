"""Render the fireside filler + interjection phrases to mp3 in Glen's clone.

Reads static/fireside/fireside-manifest.json, renders each `text` via ElevenLabs
(same voice/model as /chat/tts), and writes the `file` path. Run once:

    doppler run -p remedy-match -c prd -- python3 scripts/gen_fireside_fillers.py

Offline / no key: pass --placeholder to emit short silent mp3s via ffmpeg so the
UI and render-verify still work; swap in real clips later by re-running with a key.
"""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "static" / "fireside" / "fireside-manifest.json"
EL_BASE = "https://api.elevenlabs.io/v1"


def _abs(rel: str) -> Path:
    return ROOT / rel.lstrip("/")


def _silent(out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", "0.7", "-q:a", "9", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _render(text: str, out: Path):
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "")
    payload = json.dumps({
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.80, "style": 0.20},
    }).encode()
    req = urllib.request.Request(
        f"{EL_BASE}/text-to-speech/{voice}", data=payload,
        headers={"xi-api-key": api_key, "Content-Type": "application/json",
                 "Accept": "audio/mpeg"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        audio = resp.read()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio)


def main():
    placeholder = "--placeholder" in sys.argv or not os.environ.get("ELEVENLABS_API_KEY")
    manifest = json.loads(MANIFEST.read_text())
    clips = list(manifest["fillers"]) + list(manifest["interjections"])
    for c in clips:
        out = _abs(c["file"])
        if placeholder:
            _silent(out)
            print(f"[placeholder] {out.name}")
        else:
            _render(c["text"], out)
            print(f"[rendered]    {out.name}  <- {c['text']!r}")


if __name__ == "__main__":
    main()
