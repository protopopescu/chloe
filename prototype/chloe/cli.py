"""
Interactive REPL entry point.

Usage:
    python -m chloe                  # uses ./chloe_data.db
    python -m chloe --db path.db     # custom storage location
    python -m chloe --declare-transitive is   # opt a relation into transitive hypothesis generation

Type ordinary sentences ('Claire is an artist'), questions ('what is Claire?',
'is Claire an artist?'), or commands: sleep, what do you know, who do you
trust, show open questions, ask more, stop, exit.
"""

import argparse
import sys

from .dialogue import ChloeEngine
from .models import RelationProperties
from .storage import KnowledgeStore


def main(argv=None):
    parser = argparse.ArgumentParser(description="Talk to CHLOE.")
    parser.add_argument("--db", default="chloe_data.db", help="path to the SQLite knowledge store")
    parser.add_argument("--declare-transitive", action="append", default=[],
                         help="relation name to explicitly mark as transitive for hypothesis generation")
    parser.add_argument("--declare-symmetric", action="append", default=[],
                         help="relation name to explicitly mark as symmetric")
    args = parser.parse_args(argv)

    store = KnowledgeStore(args.db)
    for rel in args.declare_transitive:
        store.declare_relation(RelationProperties(name=rel, transitive=True))
    for rel in args.declare_symmetric:
        existing = store.get_relation(rel)
        store.declare_relation(RelationProperties(name=rel, transitive=existing.transitive, symmetric=True))

    engine = ChloeEngine(store)
    name = input("Who are you? ").strip() or "Friend"
    print(" - " + engine.greet(name))

    pending = engine.pending_question()
    if pending:
        print(" - " + pending)

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        reply = engine.turn(text)
        print(" - " + reply)
        if text.strip().lower() in ("exit", "quit", "bye", "goodbye"):
            break

    store.close()


if __name__ == "__main__":
    main(sys.argv[1:])
