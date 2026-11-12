#!/usr/bin/env python3
"""
Merge the core of another CHLOE instance into this one (chloe/ingest.py).

    python3 ingest_core.py uni_chat.db other.db             take other.db in
    python3 ingest_core.py uni_chat.db other.db --dry-run   report what it would change
    python3 ingest_core.py uni_chat.db                      list what has been taken in

The source store is only read, and may be any store this project writes, of
any schema version. Running it again with a later copy of the same source
brings over only what is new.

Stop the web server before ingesting into the store it serves: an
open conversation holds its speaker's trust in memory and writes it back.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chloe.ingest import ingest, open_snapshot
from chloe.storage import KnowledgeStore


def _list(path: str) -> None:
    store = KnowledgeStore(path)
    try:
        rows = store.ingests()
    finally:
        store.close()
    if not rows:
        print("Nothing has been ingested into this store.")
        return
    for row in rows:
        print(f"#{row['id']}  {row['at']}  {row['source']}")
        print("    " + json.dumps(row["summary"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Merge another CHLOE instance's store into this one.")
    ap.add_argument("store", help="the store to merge into")
    ap.add_argument("source", nargs="?", help="the store to take in")
    ap.add_argument("--dry-run", action="store_true", help="ingest into a copy and report, changing nothing")
    args = ap.parse_args(argv)

    if args.source is None:
        _list(args.store)
        return 0
    if not os.path.exists(args.source):
        print(f"No such store: {args.source}", file=sys.stderr)
        return 1
    if os.path.exists(args.store) and os.path.samefile(args.store, args.source):
        print("A store cannot ingest itself.", file=sys.stderr)
        return 1

    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            store = open_snapshot(args.store, tmp) if os.path.exists(args.store) \
                else KnowledgeStore(os.path.join(tmp, "empty.db"))
            try:
                report = ingest(store, args.source)
            finally:
                store.close()
        print("Dry run -- nothing was written.")
    else:
        store = KnowledgeStore(args.store)
        try:
            report = ingest(store, args.source)
        finally:
            store.close()
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
