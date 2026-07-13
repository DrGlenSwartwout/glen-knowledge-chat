"""Pilot exporter: scrape a small sample of farms and write CSV + JSON.

Usage:  python3 -m scrapers.farm_finder.export_pilot [limit] [out_dir]

No database writes — this is the "show me the data before we build the UI"
pilot artifact. Prints a compact summary table to stdout."""
import csv
import json
import sys

from scrapers.farm_finder.foodforhumans import scrape
from scrapers.farm_finder.mapping import to_practitioner_row


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    rows = scrape(limit=limit, sleep=0.5)
    dicts = [r.to_dict() for r in rows]
    # The integrated shape actually written to the practitioners table.
    practitioner_rows = [to_practitioner_row(r) for r in rows]
    with open(f"{out_dir}/farm_finder_pilot_practitioner_rows.json", "w",
              encoding="utf-8") as fh:
        json.dump(practitioner_rows, fh, indent=2, ensure_ascii=False)

    json_path = f"{out_dir}/farm_finder_pilot.json"
    csv_path = f"{out_dir}/farm_finder_pilot.csv"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(dicts, fh, indent=2, ensure_ascii=False)

    cols = [
        "name", "city", "state", "country", "lat", "lng",
        "practices", "products", "order_options",
        "website", "phone", "email", "source_url",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for d in dicts:
            w.writerow([
                "; ".join(d[c]) if isinstance(d.get(c), list) else d.get(c, "")
                for c in cols
            ])

    print(f"Scraped {len(rows)} farms -> {json_path} , {csv_path}\n")
    geocoded = sum(1 for d in dicts if d.get("lat") is not None)
    with_practices = sum(1 for d in dicts if d.get("practices"))
    print(f"pre-geocoded: {geocoded}/{len(rows)}   with practices: "
          f"{with_practices}/{len(rows)}\n")
    for d, pr in zip(dicts, practitioner_rows):
        loc = ", ".join(x for x in [d.get("city"), d.get("state")] if x)
        prac = ", ".join(d.get("practices", [])[:4])
        print(f"- {d['name']} ({loc}) [{d.get('lat')},{d.get('lng')}] "
              f"tier={pr['tier']}")
        print(f"    specialties: {', '.join(pr['specialties'])}")
        print(f"    products:    {', '.join(d.get('products', []))}")
        print(f"    ordering:    {', '.join(d.get('order_options', []))}")


if __name__ == "__main__":
    main()
