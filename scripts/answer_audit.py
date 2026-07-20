#!/usr/bin/env python3
"""Ask the LIVE bot real buying questions and check its facts against the catalog.

WHY THIS EXISTS

The unit suite checks structure — that a link table has rows, that a prompt has
a rule. It cannot check TRUTH. Two customer-facing money bugs shipped on
2026-07-20 and neither was findable by any test:

  * the NIR Brain Frequency Helmet ($4,997 + $32 shipping) was quoted live as
    "$754 (includes $132 shipping)" — the $132 had leaked from another product
  * "Harmony Soft Laser" was linked to /begin/product/clarity, a $69.97
    formulation rather than the $997 laser

Both were single clauses inside fluent, confident answers. Both were found by a
human reading a complete response. This automates that reading.

WHAT IT CHECKS

For each question, against the answer the live bot actually returns:

  1. LINK TARGETS EXIST      — every /begin/product/<slug> resolves in the catalog
  2. LINK TEXT MATCHES       — the anchor text names the product it links to
                               (catches clarity-linked-as-Harmony)
  3. PRICES ARE REAL         — every dollar figure equals a price or shipping
                               figure of a product named in that answer
  4. NO RETIRED DESTINATIONS — no practicebetter.io, no GrooveKart storefront

Findings are reported, never auto-fixed: a false positive here is cheap, a
wrong "all clear" is not.

USAGE
    python3 scripts/answer_audit.py                     # production
    python3 scripts/answer_audit.py --base http://localhost:5000
    python3 scripts/answer_audit.py --only price        # one question set

Exits non-zero if anything is flagged, so it can be scheduled.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://illtowell.com"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

# Questions chosen to exercise the paths that have actually broken.
QUESTIONS = [
    ("price", "do you sell a brain frequency helmet and how much"),
    ("price", "how much is the Harmony Soft Laser"),
    ("price", "what does the Therapeutic Nightlight cost"),
    ("price", "how much is Terrain Restore"),
    ("buy",   "where can I buy AllerFree"),
    ("buy",   "I want to order a water ionizer"),
    ("buy",   "what water bottle do you recommend for hydrogen water"),
    ("route", "how do I log in to see my courses and buy products"),
    ("route", "where do I access the free ASH MasterClass course"),
    ("dep",   "can I still get Dental Regen Powder"),
    ("dep",   "do you sell molecular hydrogen tablets"),
]

MONEY = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]{2})?)")
MDLINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
BARE = re.compile(r"(?<!\()https?://[^\s)\]\"<>]+")
STOP = {"the", "and", "of", "for", "with", "a", "an", "in", "on", "plus", "by"}


def toks(s):
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower())
            if t not in STOP and len(t) > 1}


def ask(base, q, timeout=180):
    body = json.dumps({"query": q, "level": "self-healing", "mode": "brief"}).encode()
    req = urllib.request.Request(f"{base}/chat", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": UA})
    parts = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                try:
                    parts.append(json.loads(line[6:]).get("token", "") or "")
                except Exception:
                    pass
    return "".join(parts)


def load_catalog():
    p = Path(__file__).resolve().parent.parent / "data" / "products.json"
    return json.load(open(p))["products"]


def money_cents(s):
    return int(round(float(s.replace(",", "")) * 100))


def audit(answer, products):
    """Return a list of finding strings for one answer."""
    out = []
    links = MDLINK.findall(answer)

    # 4. retired destinations
    for pat, what in ((r"practicebetter\.io", "Practice Better URL"),
                      (r"remedymatch\.com", "GrooveKart storefront URL")):
        if re.search(pat, answer, re.I):
            out.append(f"ROUTING: emits a {what}")

    linked_slugs = []
    for text, url in links:
        if "/begin/product/" not in url:
            continue
        slug = url.split("/begin/product/")[-1].strip("/").split("?")[0]
        linked_slugs.append(slug)
        p = products.get(slug)
        # 1. target exists
        if p is None:
            out.append(f"LINK: '{text}' -> /{slug} which is NOT in the catalog")
            continue
        if p.get("inactive"):
            out.append(f"LINK: '{text}' -> /{slug} which is INACTIVE (page 404s)")
        # 2. anchor text names the product it links to
        if not (toks(text) & (toks(p.get("name")) | toks(slug))):
            out.append(f"LINK: '{text}' -> /{slug} ('{p.get('name')}') — "
                       f"anchor text shares no word with the target product")

    # 3. every dollar figure traces to a product named in this answer
    if MONEY.search(answer):
        allowed = set()
        for slug in linked_slugs:
            p = products.get(slug) or {}
            for k in ("price_cents", "regular_cents", "flat_shipping_cents",
                      "service_value_cents", "service_regular_cents"):
                if p.get(k):
                    allowed.add(int(p[k]))
        # any product NAMED in the answer also justifies its own price
        for slug, p in products.items():
            nm = p.get("name") or ""
            if len(nm) >= 8 and nm.lower() in answer.lower():
                for k in ("price_cents", "regular_cents", "flat_shipping_cents"):
                    if p.get(k):
                        allowed.add(int(p[k]))
        # common non-product figures we should not flag
        allowed |= {1300, 2300, 3200, 10000}          # USPS S/M/L + own-parcel
        for raw in MONEY.findall(answer):
            c = money_cents(raw)
            if c == 0 or c in allowed:
                continue
            out.append(f"PRICE: states ${raw} — not the price or shipping of any "
                       f"product named or linked in this answer")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--only", help="run only questions in this group")
    ap.add_argument("--show", action="store_true", help="print each full answer")
    args = ap.parse_args()

    products = load_catalog()
    qs = [q for q in QUESTIONS if not args.only or q[0] == args.only]
    print(f"auditing {len(qs)} questions against {args.base}\n")

    total, asked = 0, 0
    for group, q in qs:
        try:
            answer = ask(args.base, q)
        except Exception as e:
            print(f"[{group}] {q}\n  ASK FAILED: {e!r}\n")
            total += 1
            continue
        asked += 1
        findings = audit(answer, products)
        flag = "FLAGGED" if findings else "ok"
        print(f"[{group}] {q}\n  {flag}")
        for f in findings:
            print(f"    - {f}")
        if args.show or findings:
            print("  ---\n  " + answer.strip().replace("\n", "\n  ")[:1200] + "\n")
        total += len(findings)
        time.sleep(1)

    print(f"\n{asked}/{len(qs)} answered · {total} finding(s)")
    if total:
        print("\nFindings are reported, not fixed. Each one is a claim the bot made "
              "that the catalog does not support — check before changing anything.")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
