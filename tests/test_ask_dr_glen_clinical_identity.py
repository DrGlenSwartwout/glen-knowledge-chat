import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_chat_prompt_forbids_voice_transcript_element_inference():
    text = (ROOT / "app.py").read_text()
    assert "speech-to-text only" in text
    assert "does NOT measure vocal frequency" in text
    assert "Do not recommend any Infoceutical merely from a Five-Element" in text


def test_ed11_and_ed15_catalog_identities_are_canonical():
    products = json.loads((ROOT / "data" / "products.json").read_text())["products"]
    assert products["ed11-liver-driver"]["name"] == "ED11 Liver Driver"
    assert "Pancreas" not in products["ed11-liver-driver"]["pinecone_title"]
    assert products["ed15-pancreas-energetic-driver-infoceutical"]["name"].startswith(
        "ED15 Pancreas")


def test_retrieval_catalog_no_longer_labels_ed11_as_pancreas_driver():
    concepts = json.loads((ROOT / "data" / "atlas-concepts.json").read_text())
    ed11 = next(c for c in concepts["concepts"] if c["id"] == "ed11-liver-driver-pancreas")
    assert ed11["label"] == "ED11 Liver Driver"
    assert "ED15, not ED11" in ed11["summary"]
