"""
SQLite-backed knowledge store.

A single dependency-free file, holding the atoms, their provenance, the
people, the entailments and the open questions.
"""

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from typing import Iterable, Optional

from . import grammar
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
    at TEXT NOT NULL,
    origin TEXT,
    ingest_id INTEGER
);

-- One row per core taken in from another instance (ingest.py).
CREATE TABLE IF NOT EXISTS ingests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    at TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT ''
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
    via_atom_id INTEGER,
    void INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(atom_id) REFERENCES atoms(id)
);

-- "Whenever the specific atom holds, so does the general one": Felix is a
-- black cat, so Felix is a cat (entailment.py). `holds` = 0 records that a
-- person was asked and said it does not follow. `confirmed_by` is who said
-- it does or does not; NULL for a link read from the words during sleep.
CREATE TABLE IF NOT EXISTS entailments (
    specific_id INTEGER NOT NULL,
    general_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    holds INTEGER NOT NULL DEFAULT 1,
    confirmed_by INTEGER,
    PRIMARY KEY (specific_id, general_id)
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
    asked INTEGER NOT NULL DEFAULT 0,
    answered INTEGER NOT NULL DEFAULT 0,
    term TEXT,
    other_atom_id INTEGER
);
"""


def interaction_fingerprint(interaction_id: int, person_name: str, role: str, at: str, text: str) -> str:
    """Which turn this is, wherever it is stored (ingest.py). Taken over the
    row as its own store first wrote it, so two copies of one file agree on
    every turn they share and on none they logged apart. An ingested turn
    keeps the fingerprint it came with (Interaction.origin)."""
    key = "\x1f".join((str(interaction_id), person_name, role, at, text))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _hash_secret(secret: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256, stdlib only. Secrets are normalised (trimmed and
    casefolded) before hashing, so 'Blue Whale' and 'blue whale' count as
    the same word."""
    normalized = secret.strip().casefold()
    return hashlib.pbkdf2_hmac("sha256", normalized.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()


class KnowledgeStore:
    def __init__(self, path: str, check_same_thread: bool = True):
        # check_same_thread=False is for callers that serialise their own
        # access with a lock (e.g. a threaded web server). It only lifts
        # sqlite3's same-thread check; the connection is still not safe for
        # unsynchronised concurrent use.
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Bring a database written by an earlier version up to this schema.
        CREATE TABLE IF NOT EXISTS leaves existing tables as they are, so
        columns added since are patched in here."""
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(people)")}
        if "secret_hash" not in cols:
            self.conn.execute("ALTER TABLE people ADD COLUMN secret_hash TEXT")
        if "secret_salt" not in cols:
            self.conn.execute("ALTER TABLE people ADD COLUMN secret_salt TEXT")
        # Existing rows default to 0: still open.
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(open_questions)")}
        if "answered" not in cols:
            self.conn.execute("ALTER TABLE open_questions ADD COLUMN answered INTEGER NOT NULL DEFAULT 0")
        if "term" not in cols:
            self._migrate_questions()
        # A question about two beliefs at once (whether one implies the other).
        if "other_atom_id" not in cols:
            self.conn.execute("ALTER TABLE open_questions ADD COLUMN other_atom_id INTEGER")
        # Existing turns were all spoken here.
        int_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(interactions)")}
        if "origin" not in int_cols:
            self.conn.execute("ALTER TABLE interactions ADD COLUMN origin TEXT")
        if "ingest_id" not in int_cols:
            self.conn.execute("ALTER TABLE interactions ADD COLUMN ingest_id INTEGER")
        self.conn.execute("CREATE INDEX IF NOT EXISTS interactions_origin ON interactions(origin)")
        ent_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(entailments)")}
        if "holds" not in ent_cols:
            self.conn.execute("ALTER TABLE entailments ADD COLUMN holds INTEGER NOT NULL DEFAULT 1")
        if "confirmed_by" not in ent_cols:
            self.conn.execute("ALTER TABLE entailments ADD COLUMN confirmed_by INTEGER")

        # Existing rows stay NULL: not recorded.
        prov_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(provenance)")}
        if "parse_confidence" not in prov_cols:
            self.conn.execute("ALTER TABLE provenance ADD COLUMN parse_confidence REAL")
        # Existing rows are direct evidence, none of it void.
        if "via_atom_id" not in prov_cols:
            self.conn.execute("ALTER TABLE provenance ADD COLUMN via_atom_id INTEGER")
        if "void" not in prov_cols:
            self.conn.execute("ALTER TABLE provenance ADD COLUMN void INTEGER NOT NULL DEFAULT 0")

    def _migrate_questions(self) -> None:
        """Rows that stored a question as a finished sentence. Each keeps its
        text as a label, an unknown term is recovered from it, and an atom
        with several open questions keeps only the newest."""
        self.conn.execute("ALTER TABLE open_questions ADD COLUMN term TEXT")
        for row in self.conn.execute(
                "SELECT id, question FROM open_questions WHERE reason = 'unknown_term'").fetchall():
            m = re.match(r"^\s*(?:what|who)\s+(?:is|are|am)\s+(.+?)\s*\?*\s*$", row["question"], re.I)
            if m:
                self.conn.execute("UPDATE open_questions SET term = ? WHERE id = ?", (m.group(1), row["id"]))
        self.conn.execute(
            "UPDATE open_questions SET answered = 1 WHERE answered = 0 AND related_atom_id IS NOT NULL "
            "AND id NOT IN (SELECT MAX(id) FROM open_questions WHERE answered = 0 "
            "AND related_atom_id IS NOT NULL GROUP BY related_atom_id)")

    # ---------------------------------------------------------------- people
    def find_person_by_name(self, name: str) -> Optional[Person]:
        row = self.conn.execute("SELECT * FROM people WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
        return self._row_to_person(row) if row else None

    def person_by_id(self, person_id: int) -> Optional[Person]:
        row = self.conn.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone()
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

    def add_person(self, person: Person) -> Person:
        """Insert a whole person record -- trust and secret included -- as
        it stood in another store. The id is this store's."""
        cur = self.conn.execute(
            "INSERT INTO people (name, trust, created_at, secret_hash, secret_salt) VALUES (?, ?, ?, ?, ?)",
            (person.name, json.dumps(person.trust), person.created_at, person.secret_hash, person.secret_salt),
        )
        self.conn.commit()
        return Person(id=cur.lastrowid, name=person.name, trust=dict(person.trust),
                      created_at=person.created_at, secret_hash=person.secret_hash,
                      secret_salt=person.secret_salt)

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
            "INSERT INTO interactions (person_id, role, text, at, origin, ingest_id) VALUES (?, ?, ?, ?, ?, ?)",
            (interaction.person_id, interaction.role, interaction.text, interaction.at,
             interaction.origin, interaction.ingest_id),
        )
        self.conn.commit()
        interaction.id = cur.lastrowid
        return interaction

    @staticmethod
    def _row_to_interaction(r) -> Interaction:
        return Interaction(id=r["id"], person_id=r["person_id"], role=r["role"], text=r["text"], at=r["at"],
                           origin=r["origin"], ingest_id=r["ingest_id"])

    def interactions_for_person(self, person_id: int) -> list:
        rows = self.conn.execute(
            "SELECT * FROM interactions WHERE person_id = ? ORDER BY id", (person_id,)
        ).fetchall()
        return [self._row_to_interaction(r) for r in rows]

    def interaction_by_id(self, interaction_id: int) -> Optional[Interaction]:
        r = self.conn.execute("SELECT * FROM interactions WHERE id = ?", (interaction_id,)).fetchone()
        return self._row_to_interaction(r) if r else None

    def all_interactions(self) -> list:
        rows = self.conn.execute("SELECT * FROM interactions ORDER BY id").fetchall()
        return [self._row_to_interaction(r) for r in rows]

    def fingerprint_of(self, interaction: Interaction, person_name: str) -> str:
        """The turn's fingerprint: the one it came with, or the one it has
        as a turn spoken here."""
        return interaction.origin or interaction_fingerprint(
            interaction.id, person_name, interaction.role, interaction.at, interaction.text)

    def fingerprints(self) -> dict:
        """fingerprint -> local id, for every turn in the store."""
        names = {p.id: p.name for p in self.all_people()}
        return {self.fingerprint_of(i, names.get(i.person_id, "")): i.id for i in self.all_interactions()}

    # ------------------------------------------------------------------ atoms
    def find_atoms(self, subject: str, relation: str) -> list:
        """Every atom about this subject under this relation, by
        grammar.identity rather than by spelling."""
        subj, rel = grammar.norm_phrase(subject), grammar.norm_relation(relation)
        rows = self.conn.execute(
            "SELECT * FROM atoms WHERE lower(trim(subject)) = ?", (subj,)).fetchall()
        return [a for a in (self._row_to_atom(r) for r in rows)
                if grammar.norm_relation(a.relation) == rel]

    def find_atoms_about(self, subject: str) -> list:
        """Every atom with this subject, under any relation."""
        rows = self.conn.execute("SELECT * FROM atoms WHERE lower(trim(subject)) = ?",
                                 (grammar.norm_phrase(subject),)).fetchall()
        return [self._row_to_atom(r) for r in rows]

    def atom_by_id(self, atom_id: Optional[int]) -> Optional[Atom]:
        if atom_id is None:
            return None
        row = self.conn.execute("SELECT * FROM atoms WHERE id = ?", (atom_id,)).fetchone()
        return self._row_to_atom(row) if row else None

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
        self.conn.execute("DELETE FROM entailments WHERE specific_id = ? OR general_id = ?",
                          (atom_id, atom_id))
        self.conn.execute("DELETE FROM atoms WHERE id = ?", (atom_id,))
        self.conn.commit()

    def repoint_atom(self, old_id: int, new_id: int) -> None:
        """Everything that refers to one atom now refers to another, for a
        merge: open questions, evidence carried from it, implications."""
        for col in ("related_atom_id", "other_atom_id"):
            self.conn.execute(f"UPDATE open_questions SET {col} = ? WHERE {col} = ? "
                              "AND answered = 0", (new_id, old_id))
        self.conn.execute("UPDATE provenance SET via_atom_id = ? WHERE via_atom_id = ?", (new_id, old_id))
        for col, other in (("specific_id", "general_id"), ("general_id", "specific_id")):
            self.conn.execute(f"UPDATE OR IGNORE entailments SET {col} = ? WHERE {col} = ? "
                              f"AND {other} != ?", (new_id, old_id, new_id))
        self.conn.commit()

    def has_provenance(self, prov: Provenance, atom_id: Optional[int] = None) -> bool:
        """Whether this act of evidence -- this person, in this interaction,
        this way -- is already on file: on the given atom, or on any."""
        sql = ("SELECT 1 FROM provenance WHERE person_id = ? AND interaction_id = ? AND polarity = ?"
               + (" AND atom_id = ?" if atom_id is not None else "") + " LIMIT 1")
        args = (prov.person_id, prov.interaction_id, prov.polarity) + ((atom_id,) if atom_id is not None else ())
        return self.conn.execute(sql, args).fetchone() is not None

    def add_provenance(self, atom_id: int, prov: Provenance) -> None:
        self.conn.execute(
            "INSERT INTO provenance (atom_id, person_id, interaction_id, polarity, at, parse_confidence, "
            "via_atom_id, void) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (atom_id, prov.person_id, prov.interaction_id, prov.polarity, prov.at, prov.parse_confidence,
             prov.via_atom_id, int(prov.void)),
        )
        self.conn.commit()

    def void_disputes(self, atom_id: int, via_atom_id: int) -> list:
        """Void the disputes recorded against one atom because another was
        stated. Returns the person_id of each, so the trust they cost can be
        given back."""
        rows = self.conn.execute(
            "SELECT id, person_id FROM provenance WHERE atom_id = ? AND via_atom_id = ? "
            "AND polarity < 0 AND void = 0", (atom_id, via_atom_id)).fetchall()
        self.conn.executemany("UPDATE provenance SET void = 1 WHERE id = ?", [(r["id"],) for r in rows])
        self.conn.commit()
        return [r["person_id"] for r in rows]

    # ----------------------------------------------------------- entailment
    def add_entailment(self, specific_id: int, general_id: int, holds: bool = True,
                       confirmed_by: Optional[int] = None) -> None:
        self.conn.execute("INSERT OR REPLACE INTO entailments "
                          "(specific_id, general_id, created_at, holds, confirmed_by) VALUES (?, ?, ?, ?, ?)",
                          (specific_id, general_id, now_iso(), int(holds), confirmed_by))
        self.conn.commit()

    def entails(self, atom_id: int) -> list:
        """Ids of the atoms that follow from this one."""
        return [r[0] for r in self.conn.execute(
            "SELECT general_id FROM entailments WHERE specific_id = ? AND holds = 1", (atom_id,))]

    def entailed_by(self, atom_id: int) -> list:
        """Ids of the atoms this one follows from."""
        return [r[0] for r in self.conn.execute(
            "SELECT specific_id FROM entailments WHERE general_id = ? AND holds = 1", (atom_id,))]

    def linked(self, a: int, b: int) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM entailments WHERE holds = 1 AND ((specific_id = ? AND general_id = ?) "
            "OR (specific_id = ? AND general_id = ?))", (a, b, b, a)).fetchone() is not None

    def judged(self, a: int, b: int) -> bool:
        """Whether the pair has been settled either way: linked, or found not
        to follow."""
        return self.conn.execute(
            "SELECT 1 FROM entailments WHERE (specific_id = ? AND general_id = ?) "
            "OR (specific_id = ? AND general_id = ?)", (a, b, b, a)).fetchone() is not None

    def all_entailments(self) -> list:
        return [dict(r) for r in self.conn.execute("SELECT * FROM entailments ORDER BY created_at")]

    def provenance_for(self, atom_id: int) -> list:
        rows = self.conn.execute("SELECT * FROM provenance WHERE atom_id = ? ORDER BY at, id", (atom_id,)).fetchall()
        return [self._row_to_provenance(r) for r in rows]

    def _row_to_provenance(self, row) -> Provenance:
        return Provenance(
            person_id=row["person_id"], interaction_id=row["interaction_id"], polarity=row["polarity"],
            at=row["at"], parse_confidence=row["parse_confidence"],
            via_atom_id=row["via_atom_id"], void=bool(row["void"]),
        )

    def count_atoms(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]

    def has_evidence_from(self, person_id: int) -> bool:
        """Whether this person has ever been a source."""
        return self.conn.execute("SELECT 1 FROM provenance WHERE person_id = ? LIMIT 1",
                                 (person_id,)).fetchone() is not None

    def all_atoms(self) -> list:
        """Every atom, with its provenance, in two queries rather than one
        per atom -- the sleep passes read the whole store."""
        by_atom = {}
        for row in self.conn.execute("SELECT * FROM provenance ORDER BY at, id"):
            by_atom.setdefault(row["atom_id"], []).append(self._row_to_provenance(row))
        atoms = []
        for row in self.conn.execute("SELECT * FROM atoms").fetchall():
            atom = self._row_to_atom(row, provenance=None)
            atom.provenance = by_atom.get(atom.id, [])
            atoms.append(atom)
        return atoms

    def _row_to_atom(self, row, provenance: bool = True) -> Atom:
        atom = Atom(
            id=row["id"], subject=row["subject"], relation=row["relation"], object=row["object"],
            scope=row["scope"], domain=row["domain"], status=AtomStatus(row["status"]),
            confidence=row["confidence"], created_at=row["created_at"], updated_at=row["updated_at"],
        )
        if provenance:
            atom.provenance = self.provenance_for(atom.id)
        return atom

    # -------------------------------------------------------------- relations
    def get_relation(self, name: str) -> RelationProperties:
        row = self.conn.execute("SELECT * FROM relations WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
        if not row:
            return RelationProperties(name=name)  # defaults: not transitive, not symmetric
        return RelationProperties(name=row["name"], transitive=bool(row["transitive"]), symmetric=bool(row["symmetric"]))

    def declared_relations(self) -> list:
        """Every relation with a row of its own, declared either way."""
        return [RelationProperties(name=r["name"], transitive=bool(r["transitive"]), symmetric=bool(r["symmetric"]))
                for r in self.conn.execute("SELECT * FROM relations ORDER BY name")]

    def declare_relation(self, props: RelationProperties) -> None:
        self.conn.execute(
            "INSERT INTO relations (name, transitive, symmetric) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET transitive=excluded.transitive, symmetric=excluded.symmetric",
            (props.name, int(props.transitive), int(props.symmetric)),
        )
        self.conn.commit()

    # --------------------------------------------------------- open questions
    def queue_question(self, label: str, reason: str, related_atom_id: Optional[int] = None,
                       term: Optional[str] = None, other_atom_id: Optional[int] = None) -> bool:
        """Open a question about an atom, a pair of atoms, or an unknown term.

        One open question per subject of doubt: if the atom (the pair, the
        term) already has one, its reason is brought up to date -- the
        newest reason is why CHLOE is unsure now -- and nothing is added.
        `label` is a neutral rendering kept for inspection; the question is
        worded afresh whenever it is asked (see questions.py). Returns True
        if a new question was opened.
        """
        if related_atom_id is not None:
            existing = self.conn.execute(
                "SELECT id FROM open_questions WHERE related_atom_id = ? AND other_atom_id IS ? "
                "AND answered = 0", (related_atom_id, other_atom_id)).fetchone()
        else:
            existing = self.conn.execute(
                "SELECT id FROM open_questions WHERE related_atom_id IS NULL AND answered = 0 "
                "AND lower(trim(term)) = ?", (grammar.norm_phrase(term or ""),)).fetchone()
        if existing:
            self.conn.execute("UPDATE open_questions SET reason = ?, question = ? WHERE id = ?",
                              (reason, label, existing["id"]))
            self.conn.commit()
            return False
        self.conn.execute(
            "INSERT INTO open_questions (question, reason, related_atom_id, term, other_atom_id, "
            "created_at, asked) VALUES (?, ?, ?, ?, ?, ?, 0)",
            (label, reason, related_atom_id, term, other_atom_id, now_iso()),
        )
        self.conn.commit()
        return True

    def has_open_question(self, related_atom_id: Optional[int] = None, term: Optional[str] = None,
                          other_atom_id: Optional[int] = None) -> bool:
        """Whether this subject of doubt already has an open question, by
        the same test queue_question() uses."""
        if related_atom_id is not None:
            row = self.conn.execute(
                "SELECT 1 FROM open_questions WHERE related_atom_id = ? AND other_atom_id IS ? "
                "AND answered = 0", (related_atom_id, other_atom_id)).fetchone()
        else:
            row = self.conn.execute(
                "SELECT 1 FROM open_questions WHERE related_atom_id IS NULL AND answered = 0 "
                "AND lower(trim(term)) = ?", (grammar.norm_phrase(term or ""),)).fetchone()
        return row is not None

    def has_question(self, reason: str, related_atom_id: Optional[int], term: Optional[str],
                     other_atom_id: Optional[int]) -> bool:
        """Whether this exact question has been opened here, answered or not."""
        return self.conn.execute(
            "SELECT 1 FROM open_questions WHERE reason = ? AND related_atom_id IS ? "
            "AND other_atom_id IS ? AND term IS ?", (reason, related_atom_id, other_atom_id, term)
        ).fetchone() is not None

    def add_closed_question(self, label: str, reason: str, related_atom_id: Optional[int],
                            term: Optional[str], other_atom_id: Optional[int], created_at: str) -> None:
        """A question kept for the record of what was once asked."""
        self.conn.execute(
            "INSERT INTO open_questions (question, reason, related_atom_id, term, other_atom_id, "
            "created_at, asked, answered) VALUES (?, ?, ?, ?, ?, ?, 1, 1)",
            (label, reason, related_atom_id, term, other_atom_id, created_at))
        self.conn.commit()

    def all_questions(self) -> list:
        return [dict(r) for r in self.conn.execute("SELECT * FROM open_questions ORDER BY id")]

    def next_question(self):
        """The next open question in queue order. Never-asked ones come
        first; an asked-but-unanswered one comes round again rather than
        being lost. Whether it is still worth asking, and of whom, is
        questions.next_for()'s business."""
        qs = self.pending_questions()
        return qs[0] if qs else None

    def question_by_id(self, question_id: int):
        row = self.conn.execute("SELECT * FROM open_questions WHERE id = ?", (question_id,)).fetchone()
        return dict(row) if row else None

    def mark_question_asked(self, question_id: int) -> None:
        self.conn.execute("UPDATE open_questions SET asked = 1 WHERE id = ?", (question_id,))
        self.conn.commit()

    def mark_question_answered(self, question_id: int) -> None:
        """Closed: answered, or no longer worth asking."""
        self.conn.execute(
            "UPDATE open_questions SET answered = 1, asked = 1 WHERE id = ?", (question_id,)
        )
        self.conn.commit()

    def questions_by_reason(self, reason: str) -> list:
        """Every question opened for this reason, answered or not. The row
        records what the question was about, which outlives the asking."""
        rows = self.conn.execute(
            "SELECT * FROM open_questions WHERE reason = ? ORDER BY id", (reason,)).fetchall()
        return [dict(r) for r in rows]

    def pending_questions(self) -> list:
        """Everything still open, in queue order: never asked first."""
        rows = self.conn.execute(
            "SELECT * FROM open_questions WHERE answered = 0 ORDER BY asked, created_at, id"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- ingests
    def begin_ingest(self, source: str) -> int:
        cur = self.conn.execute("INSERT INTO ingests (source, at) VALUES (?, ?)", (source, now_iso()))
        self.conn.commit()
        return cur.lastrowid

    def finish_ingest(self, ingest_id: int, summary: dict) -> None:
        self.conn.execute("UPDATE ingests SET summary = ? WHERE id = ?", (json.dumps(summary), ingest_id))
        self.conn.commit()

    def ingests(self) -> list:
        rows = self.conn.execute("SELECT * FROM ingests ORDER BY id").fetchall()
        return [dict(r, summary=json.loads(r["summary"]) if r["summary"] else {}) for r in rows]

    def close(self):
        self.conn.close()
