# scripts/migrate_sasha_family.py
"""One-off: move Sasha's cross-keyed report back under her own account and link
her to Karin's family. Idempotent."""
import os, sqlite3, datetime
from dashboard import family_access as fa

LOG_DB = os.environ.get("LOG_DB", os.path.expanduser("~/deploy-chat/chat_log.db"))
KARIN = "permanentlyyours@hawaii.rr.com"       # real email = family primary
SASHA = "permanentlyyours777@hawaiiantel.net"  # Sasha's own (fake) E4L email
now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

cx = sqlite3.connect(LOG_DB)
fa.init_tables(cx)
fa.upsert_member(cx, KARIN, KARIN, "Karin Takahashi", "human", 0)
fa.upsert_member(cx, KARIN, SASHA, "Sasha (Karin Takahashi's cat)", "pet", 1)
# remove the stopgap cross-keyed report row (Sasha's 2026-07-02 under Karin's email)
cx.execute("DELETE FROM portal_biofield_reports WHERE lower(email)=? AND scan_date=?",
           (KARIN, "2026-07-02"))
cx.commit()
print("linked Sasha under Karin; removed cross-keyed report row")
