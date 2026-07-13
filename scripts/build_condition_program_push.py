# scripts/build_condition_program_push.py
"""Emit console-push payloads from data/condition_programs_seed.json.
Prod is seeded ONCE-EVER, so seed edits don't reach the live DB — Glen pushes
each program via POST /api/console/condition-programs. Run:
  python scripts/build_condition_program_push.py > /tmp/cp_push.txt
Outputs one JSON body per program + a ready curl per program."""
import json, os
SEED = os.path.join(os.path.dirname(__file__), "..", "data", "condition_programs_seed.json")

def main():
    with open(SEED) as f:
        progs = json.load(f)["condition_programs"]
    for key, p in progs.items():
        body = {"condition_key": key, "label": p.get("label", ""),
                "consult_recommended": bool(p.get("consult_recommended")),
                "items": p.get("items", []), "modifiers": p.get("modifiers", [])}
        print(f"# {key}")
        print("curl -sS -X POST \"$CONSOLE_BASE/api/console/condition-programs\" "
              "-H 'Content-Type: application/json' "
              "-H \"$CONSOLE_AUTH\" -d @- <<'JSON'")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        print("JSON\n")

if __name__ == "__main__":
    main()
