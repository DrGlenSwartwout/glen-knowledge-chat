"""Derive slug-keyed `purchase_history` rows from the FMP invoice-history
projection (`fmp_invoice_items` / `fmp_invoices` / `fmp_clients`, built by
`dashboard.fmp_orders`).

Maps each line item's `id_fk_product` to a product slug via the FINALIZED
`resolved` map in `data/fmp_slug_map.json` (built by the Task 2 mapping
script). Ids in `exclude` (and any id not in `resolved`) are skipped. Rows
with no client email on file are skipped. Writes into `purchase_history`
(source='fmp') via `dashboard.purchase_history.replace_source`, which fully
replaces the 'fmp' slice each run (idempotent re-derivation on every ingest).

Pure: caller passes cx (and the already-loaded slug_map dict).
"""
from dashboard import purchase_history as _ph


def rebuild_from_fmp(cx, slug_map):
    """Rebuild the 'fmp' slice of `purchase_history` from the FMP projection
    tables already loaded into `cx` (fmp_clients / fmp_invoices /
    fmp_invoice_items — see dashboard.fmp_orders). `slug_map` is the loaded
    data/fmp_slug_map.json dict; only its 'resolved' {id: slug} map is used
    for id->slug resolution. Returns row/skip counts."""
    resolved = (slug_map or {}).get("resolved") or {}
    exclude = set(str(x) for x in ((slug_map or {}).get("exclude") or []))

    rows = []
    skipped_excluded = 0
    skipped_unmapped = 0
    skipped_noemail = 0

    q = cx.execute(
        "SELECT it.id_pk, it.id_fk_product, inv.invoice_date, cl.email "
        "FROM fmp_invoice_items it "
        "JOIN fmp_invoices inv ON inv.id_pk = it.id_fk_invoice "
        "JOIN fmp_clients cl ON cl.id_pk = inv.id_fk_client")
    for item_id, product_id, invoice_date, email in q.fetchall():
        pid = str(product_id or "")
        if pid in exclude:
            skipped_excluded += 1
            continue
        slug = resolved.get(pid)
        if not slug:
            skipped_unmapped += 1
            continue
        if not (email or "").strip():
            skipped_noemail += 1
            continue
        rows.append((email, slug, invoice_date, item_id))

    n = _ph.replace_source(cx, "fmp", rows)
    return {"rows": n, "skipped_excluded": skipped_excluded,
            "skipped_unmapped": skipped_unmapped, "skipped_noemail": skipped_noemail}
