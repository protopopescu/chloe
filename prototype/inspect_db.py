#!/usr/bin/env python3
"""
Read-only inspector for a CHLOE knowledge store.

Every belief CHLOE holds is a row you can read, with the evidence that put
it there. This prints that, for any store the prototype or the web server
has written: the CLI's chloe_data.db, the site's uni_chat.db, demo_chloe.db.

    python3 inspect_db.py uni_chat.db                 beliefs (default)
    python3 inspect_db.py uni_chat.db --people        interlocutors and trust
    python3 inspect_db.py uni_chat.db --questions     what CHLOE still wants to know
    python3 inspect_db.py uni_chat.db --transcript Dan   one person's turns
    python3 inspect_db.py uni_chat.db --all
    python3 inspect_db.py uni_chat.db --json          the same, machine-readable

Nothing here writes to the database.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chloe.dialogue import CONVERSATION_ROLES
from chloe.storage import KnowledgeStore

POLARITY = {1: "supports", -1: "disputes"}


def _people_by_id(store):
    return {p.id: p for p in store.all_people()}


def _evidence(atom, people):
    out = []
    for prov in atom.provenance:
        person = people.get(prov.person_id)
        out.append({
            "person": person.name if person else f"#{prov.person_id}",
            "effect": POLARITY.get(prov.polarity, str(prov.polarity)),
            "at": prov.at,
            "parse_confidence": prov.parse_confidence,
        })
    return out


def collect(store, want):
    people = _people_by_id(store)
    data = {"database": store.path}

    if "beliefs" in want:
        data["beliefs"] = [{
            "id": a.id,
            "statement": a.statement(),
            "subject": a.subject, "relation": a.relation, "object": a.object,
            "scope": a.scope, "domain": a.domain,
            "status": a.status.value, "confidence": round(a.confidence, 3),
            "created_at": a.created_at, "updated_at": a.updated_at,
            "evidence": _evidence(a, people),
        } for a in sorted(store.all_atoms(), key=lambda a: (-a.confidence, a.subject.lower()))]

    if "people" in want:
        data["people"] = [{
            "name": p.name, "trust": p.trust, "created_at": p.created_at,
            "secret_set": p.has_secret,
        } for p in people.values()]

    if "questions" in want:
        data["open_questions"] = [
            {"question": q["question"], "reason": q["reason"]} for q in store.pending_questions()
        ]

    return data


def transcript(store, name):
    person = store.find_person_by_name(name)
    if person is None:
        raise SystemExit(f"no interlocutor called {name!r} in this store")
    return {
        "person": person.name,
        "turns": [
            {"role": i.role, "text": i.text, "at": i.at}
            for i in store.interactions_for_person(person.id)
            if i.role in CONVERSATION_ROLES
        ],
    }


def print_human(data):
    if "beliefs" in data:
        beliefs = data["beliefs"]
        print(f"\nBELIEFS ({len(beliefs)})")
        if not beliefs:
            print("  (none yet)")
        for b in beliefs:
            print(f"\n  #{b['id']}  {b['statement']}")
            print(f"      {b['status']}, confidence {b['confidence']:.2f}, domain {b['domain']}")
            for e in b["evidence"]:
                conf = "" if e["parse_confidence"] is None else f", parse {e['parse_confidence']:.2f}"
                print(f"      - {e['person']} {e['effect']} it ({e['at']}{conf})")
            if not b["evidence"]:
                print("      - no evidence: generated during sleep, not yet verified")

    if "people" in data:
        print(f"\nINTERLOCUTORS ({len(data['people'])})")
        for p in data["people"]:
            trust = ", ".join(f"{k} {v:.2f}" for k, v in p["trust"].items()) or "no trust recorded yet"
            print(f"  {p['name']}: {trust}{'  [secret word set]' if p['secret_set'] else ''}")

    if "open_questions" in data:
        print(f"\nOPEN QUESTIONS ({len(data['open_questions'])})")
        for q in data["open_questions"]:
            print(f"  [{q['reason']}] {q['question']}")

    if "turns" in data:
        print(f"\nTRANSCRIPT — {data['person']} ({len(data['turns'])} turns)")
        for t in data["turns"]:
            who = data["person"] if t["role"] == "human" else "Chloe"
            print(f"  {t['at']}  {who}: {t['text']}")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Inspect a CHLOE knowledge store (read-only).")
    ap.add_argument("database", help="path to the SQLite store, e.g. uni_chat.db")
    ap.add_argument("--people", action="store_true", help="interlocutors and their per-domain trust")
    ap.add_argument("--questions", action="store_true", help="questions CHLOE has queued")
    ap.add_argument("--transcript", metavar="NAME", help="one interlocutor's logged turns")
    ap.add_argument("--all", action="store_true", help="beliefs, people and questions")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if not Path(args.database).exists():
        raise SystemExit(f"no such database: {args.database}")

    want = set()
    if args.all:
        want = {"beliefs", "people", "questions"}
    else:
        if args.people:
            want.add("people")
        if args.questions:
            want.add("questions")
        if not want and not args.transcript:
            want.add("beliefs")

    store = KnowledgeStore(args.database)
    try:
        data = collect(store, want) if want else {}
        if args.transcript:
            data.update(transcript(store, args.transcript))
    finally:
        store.close()

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_human(data)


if __name__ == "__main__":
    main()
