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
        "audio_key": "ebooks/healing-glaucoma/starter/audio.mp3",
        "condition": "glaucoma",
    },
    "macular-regeneration-starter": {
        "slug": "macular-regeneration-starter",
        "title": "Macular Regeneration — Starter",
        "site": "macularegeneration.com",
        "dir": "macular-regeneration-starter",
        "pdf": "starter.pdf",
        "audio": "starter.mp3",
        "audio_key": "ebooks/macular-regeneration/starter/audio.mp3",
        "condition": "macular",
    },
    "cataract-solutions-starter": {
        "slug": "cataract-solutions-starter",
        "title": "Cataract Solutions — Starter",
        "site": "cataractlab.com",
        "dir": "cataract-solutions-starter",
        "pdf": "starter.pdf",
        "audio": "starter.mp3",
        "audio_key": "ebooks/cataract-solutions/starter/audio.mp3",
        "condition": "cataract",
    },
    "dry-eye-relief-starter": {
        "slug": "dry-eye-relief-starter",
        "title": "Dry Eye Relief — Starter",
        "site": "dryeyelab.com",
        "dir": "dry-eye-relief-starter",
        "pdf": "starter.pdf",
        "audio": "starter.mp3",
        "audio_key": "ebooks/dry-eye-relief/starter/audio.mp3",
        "condition": "dry-eye",
    },
    "refreshing-vision-starter": {
        "slug": "refreshing-vision-starter",
        "title": "Refreshing Vision — Starter",
        "site": "refreshingvision.com",
        "dir": "refreshing-vision-starter",
        "pdf": "starter.pdf",
        "audio": "starter.mp3",
        "audio_key": "ebooks/refreshing-vision/starter/audio.mp3",
        "condition": "vision-improvement",
    },
}


def get(slug):
    return CATALOG.get((slug or "").strip())


def all():
    return list(CATALOG.values())
