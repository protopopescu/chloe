"""
SQLite-backed knowledge store.

The 2000 version stored everything in flat per-letter text files
(dict/*.cw, xref/*.xr) rewritten wholesale on every update, which the
original notes already listed as a scaling obstacle. SQLite gives
durability and queryability while staying a single dependency-free file --
the smallest upgrade that keeps this a prototype rather than an
infrastructure build-out. Swapping this module
for a real graph database later would not require changing the models.
"""

import hashlib
import hmac
import json
import secrets
import sqlite3
from typing import Iterable, Optional

from .models import Atom, AtomStatus, Interaction, Person, Provenance, RelationProperties, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    trust TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    secret_hash TEXT,
    secret_salt TEXT
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS atoms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    object TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT 'general',
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    atom_id INTEGER NOT NULL,
    person_id INTEGER NOT NULL,
    interaction_id INTEGER NOT NULL,
    polarity INTEGER NOT NULL,
    at TEXT NOT NULL,
    parse_confidence REAL,
    FOREIGN KEY(atom_id) REFERENCES atoms(id)
);

CREATE TABLE IF NOT EXISTS relations (
    name TEXT PRIMARY KEY,
    transitive INTEGER NOT NULL DEFAULT 0,
    symmetric INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS open_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    reason TEXT NOT NULL,
    related_atom_id INTEGER,
    created_at TEXT NOT NULL,
    asked INTEGER NOT NULL DEFAULT 0
);
"""


def _hash_secret(secret: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256, stdlib only. Secrets are normalised (trimmed and
    casefolded) before hashing, so 'Blue Whale' and 'blue whale' count as
    the same word -- nobody remembers their own casing."""
    normalized = secret.strip().casefold()
    return hashlib.pbkdf2_hmac("sha256", normalized.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()


class KnowledgeStore:
    def __init__(self, path: str, check_same_thread: bool = True):
        # check_same_thread=False is for callers that serialise their own
        # access with a lock (e.g. a threaded web server). It only lifts
        # sqlite3's same-thread check; the connection is still not safe for
        # unsynchronised concurrent use.
        self.conn = sqlite3.connect(path, check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Upgrade databases created before the secret-word identity feature
        existed. CREATE TABLE IF NOT EXISTS in SCHEMA only affects brand-new
        databases, so pre-existing ones (chloe_data.db, uni_chat.db, ...)
        need their `people` table patched in place."""
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(people)")}
        if "secret_hash" not in cols:
            self.conn.execute("ALTER TABLE people ADD COLUMN secret_hash TEXT")
        if "secret_salt" not in cols:
            self.conn.execute("ALTER TABLE people ADD COLUMN secret_salt TEXT")
        # Databases predating the LLM input parser lack this column;
        # existing rows stay NULL, meaning "not recorded".
        prov_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(provenance)")}
        if "parse_confidence" not in prov_cols:
            self.conn.execute("ALTER TABLE provenance ADD COLUMN parse_confidence REAL")

    # ---------------------------------------------------------------- people
    def find_person_by_name(self, name: str) -> Optional[Person]:
        row = self.conn.execute("SELECT * FROM people WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
        return self._row_to_person(row) if row else None

    def get_or_create_person(self, name: str) -> Person:
        existing = self.find_person_by_name(name)
        if existing:
            return existing
        cur = self.conn.execute(
            "INSERT INTO people (name, trust, created_at) VALUES (?, ?, ?)",
            (name, "{}", now_iso()),
        )
        self.conn.commit()
        return Person(id=cur.lastrowid, name=name)

    def save_person(self, person: Person) -> None:
        self.conn.execute(
            "UPDATE people SET trust = ? WHERE id = ?",
            (json.dumps(person.trust), person.id),
        )
        self.conn.commit()

    def set_person_secret(self, person_id: int, secret: str) -> None:
        """Store the secret word as a salted PBKDF2 hash, never the
        plaintext, so identity can be confirmed on a later visit."""
        salt = secrets.token_hex(16)
        digest = _hash_secret(secret, salt)
        self.conn.execute(
            "UPDATE people SET secret_hash = ?, secret_salt = ? WHERE id = ?",
            (digest, salt, person_id),
        )
        self.conn.commit()

    def verify_person_secret(self, person: Person, candidate: str) -> bool:
        if not person.secret_hash or not person.secret_salt:
            return False
        digest = _hash_secret(candidate, person.secret_salt)
        return hmac.compare_digest(digest, person.secret_hash)

    def _row_to_person(self, row) -> Person:
        return Person(
            id=row["id"], name=row["name"], trust=json.loads(row["trust"]), created_at=row["created_at"],
            secret_hash=row["secret_hash"], secret_salt=row["secret_salt"],
        )

    def all_people(self) -> list:
        rows = self.conn.execute("SELECT * FROM people").fetchall()
        return [self._row_to_person(r) for r in rows]

    # ------------------------------------------------------------ interactions
    def log_interaction(self, interaction: Interaction) -> Interaction:
        cur = self.conn.execute(
            "INSERT INTO interactions (person_id, role, text, at) VALUES (?, ?, ?, ?)",
            (interaction.person_id, interaction.role, interaction.text, interaction.at),
        )
        self.conn.commit()
        interaction.id = cur.lastrowid
        return interaction

    # ------------------------------------------------------------------ atoms
    def find_atoms_by_key(self, key: str) -> list:
        subject, relation = key.split("|", 1)
        rows = self.conn.execute(
            "SELECT * FROM atoms WHERE lower(subject) = ? AND lower(relation) = ?",
            (subject, relation),
        ).fetchall()
        return [self._row_to_atom(r) for r in rows]

    def save_atom(self, atom: Atom) -> Atom:
        if atom.id is None:
            cur = self.conn.execute(
                """INSERT INTO atoms (subject, relation, object, scope, domain, status, confidence, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    atom.subject, atom.relation, atom.object, atom.scope, atom.domain,
                    atom.status.value, atom.confidence, atom.created_at, atom.updated_at,
                ),
            )
            atom.id = cur.lastrowid
        else:
            self.conn.execute(
                """UPDATE atoms SET subject=?, relation=?, object=?, scope=?, domain=?, status=?, confidence=?, updated_at=?
                   WHERE id=?""",
                (
                    atom.subject, atom.relation, atom.object, atom.scope, atom.domain,
                    atom.status.value, atom.confidence, now_iso(), atom.id,
                ),
            )
        self.conn.commit()
        return atom

    def delete_atom(self, atom_id: int) -> None:
        self.conn.execute("DELETE FROM provenance WHERE atom_id = ?", (atom_id,))
        self.conn.execute("DELETE FROM atoms WHERE id = ?", (atom_id,))
        self.conn.commit()

    def add_provenance(self, atom_id: int, prov: Provenance) -> None:
        self.conn.execute(
            "INSERT INTO provenance (atom_id, person_id, interaction_id, polarity, at, parse_confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (atom_id, prov.person_id, prov.interaction_id, prov.polarity, prov.at, prov.parse_confidence),
        )
        self.conn.commit()

    def provenance_for(self, atom_id: int) -> list:
        rows = self.conn.execute("SELECT * FROM provenance WHERE atom_id = ? ORDER BY at", (atom_id,)).fetchall()
        return [
            Provenance(
                person_id=r["person_id"], interaction_id=r["interaction_id"], polarity=r["polarity"],
                at=r["at"], parse_confidence=r["parse_confidence"],
            )
            for r in rows
        ]

    def all_atoms(self) -> list:
        rows = self.conn.execute("SELECT * FROM atoms").fetchall()
        return [self._row_to_atom(r) for r in rows]

    def _row_to_atom(self, row) -> Atom:
        atom = Atom(
            id=row["id"], subject=row["subject"], relation=row["relation"], object=row["object"],
            scope=row["scope"], domain=row["domain"], status=AtomStatus(row["status"]),
            confidence=row["confidence"], created_at=row["created_at"], updated_at=row["updated_at"],
        )
        atom.provenance = self.provenance_for(atom.id)
        return atom

    # -------------------------------------------------------------- relations
    def get_relation(self, name: str) -> RelationProperties:
        row = self.conn.execute("SELECT * FROM relations WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
        if not row:
            return RelationProperties(name=name)  # defaults: not transitive, not symmetric
        return RelationProperties(name=row["name"], transitive=bool(row["transitive"]), symmetric=bool(row["symmetric"]))

    def declare_relation(self, props: RelationProperties) -> None:
        self.conn.execute(
            "INSERT INTO relations (name, transitive, symmetric) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET transitive=excluded.transitive, symmetric=excluded.symmetric",
            (props.name, int(props.transitive), int(props.symmetric)),
        )
        self.conn.commit()

    # --------------------------------------------------------- open questions
    def queue_question(self, question: str, reason: str, related_atom_id: Optional[int] = None) -> None:
        # don't queue the same open question twice
        existing = self.conn.execute(
            "SELECT id FROM open_questions WHERE question = ? AND asked = 0", (question,)
        ).fetchone()
        if existing:
            return
        self.conn.execute(
            "INSERT INTO open_questions (question, reason, related_atom_id, created_at, asked) VALUES (?, ?, ?, ?, 0)",
            (question, reason, related_atom_id, now_iso()),
        )
        self.conn.commit()

    def next_question(self):
        row = self.conn.execute(
            "SELECT * FROM open_questions WHERE asked = 0 ORDER BY created_at LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def mark_question_asked(self, question_id: int) -> None:
        self.conn.execute("UPDATE open_questions SET asked = 1 WHERE id = ?", (question_id,))
        self.conn.commit()

    def pending_questions(self) -> list:
        rows = self.conn.execute("SELECT * FROM open_questions WHERE asked = 0 ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
