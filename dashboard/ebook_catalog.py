"""Static registry of free-ebook Starters. `dir` is the folder under
static/ebooks/ holding the pilot's `pdf` (read) and `audio` (listen) assets."""

CATALOG = {
    "healing-glaucoma-starter": {
        "slug": "healing-glaucoma-starter",
        "title": "Healing Glaucoma — Starter",
        "site": "healingglaucoma.com",
        "dir": "healing-glaucoma-starter",
        "pdf": "starter.pdf",
        "audio": "starter.mp3",
    },
}


def get(slug):
    return CATALOG.get((slug or "").strip())


def all():
    return list(CATALOG.values())
