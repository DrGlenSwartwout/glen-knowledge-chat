"""Tests for the one-time GrooveKart order-history backfill
(dashboard.gk_email_history), which ingests GrooveKart order-confirmation
emails from Gmail into `purchase_history` (source='groovekart'). Mirrors
tests/test_fmp_history.py's fixture pattern for rebuild_from_gk_emails."""
import sqlite3

from dashboard import gk_email_history as gh
from dashboard import purchase_history as ph


# Real sample body (verbatim, from the task spec) — plaintext Gmail body for
# a GrooveKart "New order" confirmation, one product line.
REAL_SAMPLE_BODY = """
[https://remedymatch.com/]

Congratulations!

A new order was placed on Remedy Match by the following customer:
Pamela Kilmer (pkilmer108@gmail.com)

ORDER: YOUHZZCDS Placed on 06-28-2026

PAYMENT: Credit Card

REFERENCE

PRODUCT

UNIT PRICE

QUANTITY

TOTAL PRICE

<tr style="background-color:#DDE2E6;">
					<td style="padding:0.6em 0.4em;"></td>
					<td style="padding:0.6em 0.4em;">
						<strong><a href="https://remedymatch.com/remedies/syntropy/265-ocuheal-eye-drops">OcuHeal Eye Drops - </a></strong>
					</td>
					<td style="padding:0.6em 0.4em; text-align:right;">$69.97</td>
					<td style="padding:0.6em 0.4em; text-align:center;">2</td>
					<td style="padding:0.6em 0.4em; text-align:right;">$139.94</td>
				</tr>

PRODUCTS

$139.94
"""


def test_parse_order_email_real_sample():
    result = gh.parse_order_email(REAL_SAMPLE_BODY)
    assert result["email"] == "pkilmer108@gmail.com"
    assert result["purchased_at"] == "2026-06-28"
    assert result["order_ref"] == "YOUHZZCDS"
    assert result["slugs"] == ["ocuheal-eye-drops"]


def test_parse_order_email_multiple_products():
    body = """
A new order was placed on Remedy Match by the following customer:
Jane Doe (jane@example.com)

ORDER: ABC12345 Placed on 01-15-2026

<a href="https://remedymatch.com/remedies/syntropy/100-neuro-magnesium">Neuro-Magnesium - </a>
<a href="https://remedymatch.com/remedies/terrain/200-terrain-restore">Terrain Restore - </a>
"""
    result = gh.parse_order_email(body)
    assert result["email"] == "jane@example.com"
    assert result["purchased_at"] == "2026-01-15"
    assert result["order_ref"] == "ABC12345"
    assert result["slugs"] == ["neuro-magnesium", "terrain-restore"]


def test_parse_order_email_missing_pieces_are_pragmatic():
    result = gh.parse_order_email("nothing useful here")
    assert result["email"] == ""
    assert result["purchased_at"] is None
    assert result["order_ref"] is None
    assert result["slugs"] == []


def _cx():
    cx = sqlite3.connect(":memory:")
    ph.init_purchase_history_table(cx)
    return cx


def _fake_fetch():
    return [
        {
            "subject": "New order : #1010 - YOUHZZCDS",
            "body": (
                "A new order was placed on Remedy Match by the following customer:\n"
                "Pamela Kilmer (pkilmer108@gmail.com)\n\n"
                "ORDER: YOUHZZCDS Placed on 06-28-2026\n\n"
                '<a href="https://remedymatch.com/remedies/syntropy/265-ocuheal-eye-drops">OcuHeal Eye Drops - </a>\n'
                '<a href="https://remedymatch.com/remedies/misc/999-unmapped-thing">Unmapped Thing - </a>\n'
            ),
        },
        {
            "subject": "New order : #1011 - NOEMAILXX",
            "body": (
                "A new order was placed on Remedy Match by the following customer:\n\n"
                "ORDER: NOEMAILXX Placed on 06-29-2026\n\n"
                '<a href="https://remedymatch.com/remedies/syntropy/265-ocuheal-eye-drops">OcuHeal Eye Drops - </a>\n'
            ),
        },
    ]


_CATALOG_SLUGS = {"ocuheal-eye-drops", "terrain-restore"}


def test_rebuild_from_gk_emails_writes_known_skips_unmapped_and_noemail():
    cx = _cx()
    result = gh.rebuild_from_gk_emails(cx, fetch_fn=_fake_fetch, catalog_slugs=_CATALOG_SLUGS)

    assert result == {"orders": 2, "rows": 1, "skipped_unmapped": 1, "skipped_noemail": 1}

    rows = cx.execute(
        "SELECT email, slug, purchased_at, source, source_ref FROM purchase_history"
    ).fetchall()
    assert rows == [("pkilmer108@gmail.com", "ocuheal-eye-drops", "2026-06-28", "groovekart", "YOUHZZCDS")]


def test_rebuild_from_gk_emails_is_idempotent():
    cx = _cx()
    gh.rebuild_from_gk_emails(cx, fetch_fn=_fake_fetch, catalog_slugs=_CATALOG_SLUGS)
    result2 = gh.rebuild_from_gk_emails(cx, fetch_fn=_fake_fetch, catalog_slugs=_CATALOG_SLUGS)
    assert result2["rows"] == 1
    count = cx.execute(
        "SELECT COUNT(*) FROM purchase_history WHERE source='groovekart'"
    ).fetchone()[0]
    assert count == 1
