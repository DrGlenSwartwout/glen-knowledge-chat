#!/usr/bin/env python3
"""One-time migration: create the Savant + Perdomo households from the
contacts we identified during the 2026-05-26 GHL dedup work.

Runs by curling the live web service /api/households endpoint. The web
service handles all DB + GHL sync. This script is just the orchestrator.

Required env vars:
  WEB_URL         — default https://glen-knowledge-chat.onrender.com
  CONSOLE_SECRET  — admin key for /api/households

Both households are no-ops if they already exist (the endpoint returns 200
with an "already_member" flag for repeat member-adds; for already-existing
slugs it returns 400 — we treat as informational and continue).
"""
import os
import sys
import json
import urllib.parse
import urllib.request
import urllib.error

WEB_URL = os.environ.get("WEB_URL", "https://glen-knowledge-chat.onrender.com").rstrip("/")
SECRET = os.environ.get("CONSOLE_SECRET", "")
if not SECRET:
    print("ERROR: CONSOLE_SECRET not set", flush=True)
    sys.exit(1)


def api(method, path, body=None):
    url = f"{WEB_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, method=method, data=data,
                                  headers={"Content-Type": "application/json",
                                           "X-Console-Key": SECRET})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def people_by_email(email):
    """Returns list of person dicts matching the email (lowercase exact)."""
    status, data = api("GET",
                       f"/api/people?q={urllib.parse.quote(email)}&limit=10")
    if status != 200:
        return []
    return [p for p in (data.get("people") or [])
            if (p.get("email") or "").lower() == email.lower()]


def main():
    print(f"Migration against {WEB_URL}")

    # Savant: Lotika (mother, head) + Omika (daughter) — both at lotikasavant@hotmail.com
    print("\n[Savant household]")
    savants = people_by_email("lotikasavant@hotmail.com")
    if len(savants) >= 2:
        lotika = next((p for p in savants if (p.get("first_name") or "").lower() == "lotika"), savants[0])
        omika  = next((p for p in savants if (p.get("first_name") or "").lower() == "omika"), savants[1])
        status, data = api("POST", "/api/households",
                           {"name": "Savant", "head_person_id": lotika["id"],
                            "member_person_ids": [lotika["id"], omika["id"]],
                            "created_by": "glen-migration"})
        print(f"  Savant → HTTP {status}: {data}")
    else:
        print(f"  ⚠ only {len(savants)} Savant people found — skipping")

    # Perdomo household — Kauilani is the beta-cohort head; Kanehekai +
    # Kimberly (likely Kauilani's prior name) are the other household members.
    print("\n[Perdomo household]")
    perdomos = people_by_email("restorealoha@gmail.com")
    if len(perdomos) >= 2:
        kauilani = next((p for p in perdomos
                          if (p.get("first_name") or "").lower() == "kauilani"), None)
        if not kauilani:
            print("  ⚠ no Kauilani found — skipping Perdomo migration")
        else:
            ids = [p["id"] for p in perdomos]
            status, data = api("POST", "/api/households",
                               {"name": "Perdomo", "head_person_id": kauilani["id"],
                                "member_person_ids": ids,
                                "created_by": "glen-migration"})
            print(f"  Perdomo → HTTP {status}: {data}")
    else:
        print(f"  ⚠ only {len(perdomos)} Perdomo people found — skipping")

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
