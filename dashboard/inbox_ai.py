"""Inbox AI — summarize incoming messages + draft replies in Glen's voice.

Three public functions called from the inbox routes:

    summarize(body) → {"summary": [bullets...], "actions": [numbered actions...]}
    draft_reply(body, sender, voice_samples) → str (suggested reply)
    regenerate_reply(body, prior_draft, prompt, voice_samples) → str

Uses Anthropic Haiku 4.5 for speed. Prompt-cached system block keeps
per-call cost low.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

# Reuse the same model the rest of the app uses
_MODEL = "claude-haiku-4-5-20251001"

# Glen's voice anchors — kept terse so the model has room to use voice samples
_GLEN_VOICE_PRINCIPLES = """\
Glen Swartwout is a 70-year-old natural-medicine practitioner (Healing Oasis,
Hilo Hawai'i). He writes warm, brief, doctor-tone replies. Hallmarks:

- Opens with "Aloha [Name]," (no comma after Hi)
- Uses "Mahalo" for thanks; "in health, Glen" or just "Glen" to close
- Plain English, no jargon dump
- Short paragraphs (2-4 lines max)
- Speaks of "Practice Better" or "PB" for the practice portal
- Refers to his assistant as "Rae" (Susan Luscombe)
- Acknowledges the person's specifics before pivoting to action
- Mentions "Biofield Analysis" / "voice scan" when relevant
- Treats every person as a fellow steward of their own healing — collaborative, not prescriptive
"""


def _client():
    """Lazy import so unit tests that mock the module don't pull anthropic at import."""
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def summarize(body: str) -> dict:
    """Return {summary: [bullets], actions: [numbered actions]} from the message body."""
    if not body or not body.strip():
        return {"summary": [], "actions": []}
    prompt = (
        "Summarize the following email message into:\n"
        "1) A bullet list of key points (max 4 bullets, one short sentence each)\n"
        "2) A numbered list of suggested actions Glen could take (max 4 actions)\n\n"
        "Return STRICT JSON: {\"summary\": [\"...\", ...], \"actions\": [\"...\", ...]}\n"
        "No markdown fences, no prose outside the JSON.\n\n"
        f"MESSAGE:\n{body[:6000]}"
    )
    cli = _client()
    resp = cli.messages.create(
        model=_MODEL, max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    # Tolerate accidental fences
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json\n"):
            text = text[5:]
    try:
        out = json.loads(text)
        return {"summary": out.get("summary", []), "actions": out.get("actions", [])}
    except json.JSONDecodeError:
        return {"summary": [text[:400]], "actions": []}


def _voice_samples_block(voice_samples: Optional[List[dict]]) -> str:
    if not voice_samples:
        return ""
    blocks = []
    for s in voice_samples[:5]:
        body = (s.get("body") or "").strip()
        if body:
            blocks.append(f"---\n{body[:1200]}")
    if not blocks:
        return ""
    return (
        "\n\nFor voice reference, here are Glen's recent outgoing emails. "
        "Match this voice and cadence (do NOT copy phrasing verbatim):\n\n"
        + "\n".join(blocks)
    )


def draft_reply(
    body: str,
    sender: str = "",
    voice_samples: Optional[List[dict]] = None,
) -> str:
    """Generate an initial suggested reply in Glen's voice."""
    if not body or not body.strip():
        return ""
    sender_name_hint = ""
    if sender:
        # "Name <addr>" → "Name"
        import re
        m = re.match(r'^"?([^"<]+?)"?\s*<.+>$', sender.strip())
        sender_name_hint = (m.group(1).strip() if m else sender.strip()).split()[0]

    user_prompt = (
        "Draft Glen's reply to this incoming email.\n"
        "Output ONLY the reply body — no preamble, no markdown, no quoted history.\n"
        "Keep it under 6 short paragraphs. Open with 'Aloha [first name],' if the sender is a person.\n\n"
        f"INCOMING FROM {sender}:\n{body[:6000]}"
    )
    if sender_name_hint:
        user_prompt += f"\n\n(Sender's first name appears to be: {sender_name_hint})"

    cli = _client()
    resp = cli.messages.create(
        model=_MODEL, max_tokens=900,
        system=_GLEN_VOICE_PRINCIPLES + _voice_samples_block(voice_samples),
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text.strip()


def regenerate_reply(
    body: str,
    prior_draft: str,
    prompt: str,
    sender: str = "",
    voice_samples: Optional[List[dict]] = None,
) -> str:
    """Re-draft the reply incorporating Glen's prompt/instructions.

    `prompt` is Glen's free-text instruction (e.g., "make it warmer", "write
    from scratch with focus on the appointment Tuesday", "ignore prior draft").
    """
    if not body or not body.strip():
        return ""
    user_prompt = (
        "Re-draft Glen's reply to the incoming email below, following his instructions.\n"
        "Output ONLY the reply body. No preamble, no markdown.\n\n"
        f"INCOMING FROM {sender}:\n{body[:5000]}\n\n"
        f"PRIOR DRAFT (may be incorporated, modified, or ignored per Glen's instructions):\n{prior_draft[:3000]}\n\n"
        f"GLEN'S INSTRUCTIONS:\n{prompt[:1500]}"
    )
    cli = _client()
    resp = cli.messages.create(
        model=_MODEL, max_tokens=900,
        system=_GLEN_VOICE_PRINCIPLES + _voice_samples_block(voice_samples),
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text.strip()
