"""The hand-off invoice ships free on purpose. Pin it so the derived-shipping
fix doesn't get "completed" by deleting the flag."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

SRC = (repo_root / "dashboard" / "biofield_invoice.py").read_text()


def test_hand_off_invoice_still_sends_pickup_true():
    assert '"pickup": True' in SRC
