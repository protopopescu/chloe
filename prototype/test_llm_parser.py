"""
Offline tests for the LLM input parser (chloe/llm_nlu.py).

Runs entirely without a real vLLM server: a stdlib mock speaking the
OpenAI-compatible /chat/completions shape is started on a local port and
scripted per test case, so every routing path can be exercised --
LLM parse, honest refusal, low-confidence refusal, contract violation,
malformed reply, server unreachable, command interception -- plus the
storage migration that adds parse_confidence to pre-existing databases.

Run: python3 test_llm_parser.py       (no network, no dependencies)
"""

import json
import os
import sqlite3
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

# The mock's reply for the next request, and a counter of requests seen.
_MOCK = {"reply": None, "hits": 0, "status": 200}


class _MockVLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        _MOCK["hits"] += 1
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"choices": [{"message": {"content": _MOCK["reply"]}}]})
        self.send_response(_MOCK["status"])
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if _MOCK["status"] == 200:
            self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args):  # keep test output clean
        pass


def _start_mock():
    server = HTTPServer(("127.0.0.1", 0), _MockVLLMHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def _payload(**kw):
    base = {"type": "statement", "subject": None, "relation": None, "object": None,
            "scope": None, "confidence": 0.95, "reason": None}
    base.update(kw)
    return json.dumps(base)


class LLMParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.base_url = _start_mock()
        os.environ["VLLM_BASE_URL"] = cls.base_url

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        os.environ.pop("VLLM_BASE_URL", None)

    def setUp(self):
        _MOCK["reply"], _MOCK["hits"], _MOCK["status"] = None, 0, 200
        # import late so VLLM_BASE_URL is already set
        global llm_nlu, UtteranceType
        from chloe import llm_nlu
        from chloe.nlu import UtteranceType

    # ------------------------------------------------------------ LLM parses
    def test_statement_with_scope(self):
        _MOCK["reply"] = _payload(subject="the sky", relation="is", object="blue",
                                  scope="it is daytime", confidence=0.93)
        utt = llm_nlu.parse("You know, whenever it's daytime, the sky takes on a blue colour.")
        self.assertEqual(utt.type, UtteranceType.STATEMENT)
        self.assertEqual((utt.subject, utt.relation, utt.obj, utt.scope),
                         ("the sky", "is", "blue", "it is daytime"))
        self.assertEqual(utt.extra["parser"], "llm")
        self.assertEqual(utt.extra["parse_confidence"], 0.93)
        self.assertEqual(_MOCK["hits"], 1)

    def test_negation(self):
        _MOCK["reply"] = _payload(type="negation", subject="grass", relation="is",
                                  object="purple", confidence=0.9)
        utt = llm_nlu.parse("I'd never say grass is purple, because it isn't.")
        self.assertEqual(utt.type, UtteranceType.NEGATION)
        self.assertTrue(utt.negated)
        self.assertEqual(utt.obj, "purple")

    def test_wh_question(self):
        _MOCK["reply"] = _payload(type="wh_question", subject="the sky", relation="is",
                                  confidence=0.9)
        utt = llm_nlu.parse("what colour is the sky?")
        self.assertEqual(utt.type, UtteranceType.WH_QUESTION)
        self.assertEqual(utt.subject, "the sky")

    # -------------------------------------------------------------- refusals
    def test_honest_refusal_is_final(self):
        _MOCK["reply"] = _payload(type="unknown", confidence=0.9, reason="not_parseable")
        utt = llm_nlu.parse("the sky is blue")  # regex COULD parse this...
        self.assertEqual(utt.type, UtteranceType.UNKNOWN)  # ...but must not get to
        self.assertEqual(utt.extra["reason"], "not_parseable")

    def test_low_confidence_refusal(self):
        _MOCK["reply"] = _payload(subject="the sky", relation="is", object="blue",
                                  confidence=0.3)
        utt = llm_nlu.parse("skyish bluish something?")
        self.assertEqual(utt.type, UtteranceType.UNKNOWN)
        self.assertEqual(utt.extra["reason"], "low_confidence")
        self.assertEqual(utt.extra["parse_confidence"], 0.3)

    def test_multiple_statements_refused(self):
        _MOCK["reply"] = _payload(type="unknown", confidence=0.9, reason="multiple_statements")
        utt = llm_nlu.parse("Felix is a cat and cats are animals")
        self.assertEqual(utt.type, UtteranceType.UNKNOWN)
        self.assertEqual(utt.extra["reason"], "multiple_statements")

    # ---------------------------------------------- contract violations -> fallback
    def test_command_type_rejected_then_pattern_fallback(self):
        _MOCK["reply"] = _payload(type="command", confidence=0.99)
        utt = llm_nlu.parse("the sky is blue")
        # the LLM may not emit commands; broken contract falls back to patterns
        self.assertEqual(utt.type, UtteranceType.STATEMENT)
        self.assertEqual(utt.extra["parser"], "patterns")
        self.assertIn("undeclared utterance type", utt.extra["llm_fallback_reason"])
        self.assertNotIn("parse_confidence", utt.extra)

    def test_malformed_json_falls_back(self):
        _MOCK["reply"] = "Sure! The subject is the sky and..."
        utt = llm_nlu.parse("the sky is blue")
        self.assertEqual(utt.type, UtteranceType.STATEMENT)
        self.assertEqual(utt.extra["parser"], "patterns")

    def test_code_fences_tolerated(self):
        _MOCK["reply"] = "```json\n" + _payload(subject="grass", relation="is", object="green") + "\n```"
        utt = llm_nlu.parse("grass is green")
        self.assertEqual(utt.type, UtteranceType.STATEMENT)
        self.assertEqual(utt.extra["parser"], "llm")

    def test_server_unreachable_falls_back(self):
        old = os.environ["VLLM_BASE_URL"]
        os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:1/v1"  # nothing listens here
        try:
            utt = llm_nlu.parse("the sky is blue")
        finally:
            os.environ["VLLM_BASE_URL"] = old
        self.assertEqual(utt.type, UtteranceType.STATEMENT)
        self.assertEqual(utt.extra["parser"], "patterns")

    # ------------------------------------------------------ command interception
    def test_commands_never_reach_the_llm(self):
        for cmd in ("sleep", "what do you know", "Hi Chloe, who do you trust?"):
            utt = llm_nlu.parse(cmd)
            self.assertEqual(utt.type, UtteranceType.COMMAND, cmd)
        self.assertEqual(_MOCK["hits"], 0)

    # ------------------------------------------------- engine + storage, end to end
    def test_parse_confidence_lands_in_provenance(self):
        from chloe.dialogue import ChloeEngine
        from chloe.storage import KnowledgeStore

        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(os.path.join(tmp, "t.db"))
            engine = ChloeEngine(store)
            engine.greet("Dan")
            engine.turn("no")  # decline the secret-word offer (never hits the LLM)
            self.assertEqual(_MOCK["hits"], 0)

            _MOCK["reply"] = _payload(subject="the sea", relation="is", object="salty",
                                      confidence=0.88)
            reply = engine.turn("Have I mentioned the sea is salty?  It is.")
            self.assertIn("the sea is salty", reply)
            (atom,) = store.all_atoms()
            self.assertEqual(atom.provenance[0].parse_confidence, 0.88)

            # same fact via pattern fallback (malformed reply): confidence stays None
            _MOCK["reply"] = "not json"
            engine.turn("the sea is salty")
            (atom,) = store.all_atoms()
            self.assertIsNone(atom.provenance[1].parse_confidence)
            store.close()

    def test_engine_replies_for_refusals(self):
        from chloe.dialogue import ChloeEngine
        from chloe.storage import KnowledgeStore

        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(os.path.join(tmp, "t.db"))
            engine = ChloeEngine(store)
            engine.greet("Dan")
            engine.turn("no")

            _MOCK["reply"] = _payload(type="unknown", confidence=0.9,
                                      reason="multiple_statements")
            self.assertIn("one at a time", engine.turn("Felix is a cat and cats are animals"))

            _MOCK["reply"] = _payload(subject="x", relation="is", object="y", confidence=0.2)
            self.assertIn("won't store it", engine.turn("xish yish?"))

            self.assertEqual(store.all_atoms(), [])  # nothing was believed
            store.close()

    # ---------------------------------------------------------------- migration
    def test_migration_of_pre_parser_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.db")
            conn = sqlite3.connect(path)
            conn.executescript("""
                CREATE TABLE people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
                    trust TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
                CREATE TABLE interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER NOT NULL,
                    role TEXT NOT NULL, text TEXT NOT NULL, at TEXT NOT NULL);
                CREATE TABLE atoms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT NOT NULL,
                    relation TEXT NOT NULL, object TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT '', domain TEXT NOT NULL DEFAULT 'general',
                    status TEXT NOT NULL, confidence REAL NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE provenance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, atom_id INTEGER NOT NULL,
                    person_id INTEGER NOT NULL, interaction_id INTEGER NOT NULL,
                    polarity INTEGER NOT NULL, at TEXT NOT NULL);
                INSERT INTO people (name, trust, created_at) VALUES ('Dan', '{}', 't0');
                INSERT INTO atoms (subject, relation, object, status, confidence, created_at, updated_at)
                    VALUES ('the sky', 'is', 'blue', 'candidate', 0.5, 't0', 't0');
                INSERT INTO provenance (atom_id, person_id, interaction_id, polarity, at)
                    VALUES (1, 1, 1, 1, 't0');
            """)
            conn.commit()
            conn.close()

            from chloe.storage import KnowledgeStore
            store = KnowledgeStore(path)  # runs _migrate()
            (atom,) = store.all_atoms()
            self.assertIsNone(atom.provenance[0].parse_confidence)  # old row: not recorded
            person = store.find_person_by_name("Dan")
            self.assertFalse(person.has_secret)  # older migration still applies too
            store.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
