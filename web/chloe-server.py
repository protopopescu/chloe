#!/usr/bin/env python3
"""
CHLOE web chat server.

Serves this folder's static site (index.html, chloe.css, script.js) and
two JSON endpoints:

  POST /api/greet  first contact for a browser session: the name the
                   visitor typed, run through ChloeEngine.greet(), which
                   may ask to set up or confirm a secret word (see
                   dialogue.py's AuthState).
  POST /api/chat   every message after that, run through the symbolic
                   engine in ../Prototype/chloe and then, if a vLLM server
                   is configured, phrased naturally.

Also runs "dreaming": a background thread that once a day, during a fixed
window (DREAM_START for DREAM_DURATION_MINUTES), takes the site offline
for chat and runs
chloe.consolidation.sleep() over the shared knowledge store -- merging
duplicate atoms, flagging contradictions, generating hypotheses for
declared-transitive relations, and queuing verification questions. The CLI
only ever ran that pass on an explicit "sleep" command; here it is
scheduled. See the DREAM_* environment variables below.

Stdlib only, as is the chloe package it imports: no pip install to run
this. The only outbound network call is to whatever OpenAI-compatible
server VLLM_BASE_URL points at.

Run:
    export VLLM_BASE_URL=http://<your-llm-host>:8000/v1
    python3 chloe-server.py
See README.md for the full list of environment variables.
"""

import json
import os
import sys
import threading
import time
import hmac
import uuid
from base64 import b64decode
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

try:
    from zoneinfo import ZoneInfo  # stdlib since Python 3.9
except ImportError:  # pragma: no cover
    ZoneInfo = None

SITE_DIR = Path(__file__).resolve().parent
PROTOTYPE_DIR = SITE_DIR.parent / "prototype"
sys.path.insert(0, str(PROTOTYPE_DIR))

from chloe import consolidation, llm_client, persona, questions  # noqa: E402
from chloe.dialogue import CONVERSATION_ROLES, ChloeEngine  # noqa: E402
from chloe.nlu import MAX_INPUT_CHARS  # noqa: E402
from chloe.storage import KnowledgeStore  # noqa: E402
_db_env = os.getenv("CHLOE_DB_PATH")
DB_PATH = Path(_db_env).expanduser() if _db_env else SITE_DIR / "uni_chat.db"
PORT = int(os.getenv("PORT", "5010"))
# Loopback by default: this server is meant to sit behind a reverse proxy,
# and the daily budget trusts X-Forwarded-For only because nothing can reach
# it except through one. Exposing it directly is the deliberate act.
HOST = os.getenv("HOST", "127.0.0.1")

# A fixed window once a day, not a repeating interval: nightly downtime
# rather than a periodic maintenance tick. DREAM_START is "HH:MM" in
# DREAM_TZ (default: the server process's own time zone -- set DREAM_TZ,
# e.g. "Europe/London", if that is not where "night" should fall).
# consolidation.sleep() finishes in well under a second, so CHLOE has
# nothing left to do for most of the window. It stays offline anyway,
# keeping the offline phase a real one rather than waking early.
# The belief browser exposes the whole shared store, so it is off unless a
# password is configured: absent CHLOE_BELIEFS_PASSWORD, /beliefs and
# /api/beliefs do not exist at all rather than existing unprotected. Basic
# auth sends the password in a reversible encoding on every request, so this
# is only meaningful behind the HTTPS reverse proxy the README already asks
# for; it keeps casual visitors out, it is not a security boundary.
BELIEFS_USER = os.getenv("CHLOE_BELIEFS_USER", "chloe")
BELIEFS_PASSWORD = os.getenv("CHLOE_BELIEFS_PASSWORD")
BELIEFS_REALM = "CHLOE beliefs"

# A person can also put CHLOE to sleep on demand with "Sleep, Chloe", as the
# 2000 version allowed. That runs the same consolidation pass through the same
# dream window, only briefly, so the site visibly sleeps and wakes instead of
# the pass happening invisibly inside one request.
NAP_SECONDS = int(os.getenv("CHLOE_NAP_SECONDS", "12"))

# Every visitor supplies their own session_id, so the number of live
# sessions is set by whoever is calling rather than by how many people are
# here. The registry below is bounded: past this many, the least recently
# used is dropped. A dropped session costs its in-memory turn history and
# auth state, nothing in the store, and the next message rebuilds it the
# same way a server restart does.
MAX_SESSIONS = int(os.getenv("CHLOE_MAX_SESSIONS", "500"))

# Content-Length is whatever the caller declares, so it is checked before a
# byte is read. A greet or chat body is a session id, a name and one line
# bounded by MAX_INPUT_CHARS -- well inside this.
MAX_BODY_BYTES = int(os.getenv("CHLOE_MAX_BODY_BYTES", "8192"))

# What one visitor may spend in a day: chat turns, plus the greet that opens
# a session (a greet with a name not on file writes a person row, so it is an
# exchange too; a page reload reuses its session_id and costs nothing). 0
# disables the budget. Health and wake polls, a visitor's own transcript, and
# the authenticated routes are never counted.
DAILY_EXCHANGES = int(os.getenv("CHLOE_DAILY_EXCHANGES", "20"))
OVER_BUDGET_REPLY = "Was nice talking to you. Please come back tomorrow."

# Sleeping is bounded separately and much more tightly: a nap takes the site
# quiet for NAP_SECONDS for everybody, so it costs one of these as well as
# one of the exchanges above. 0 disables the nap budget.
DAILY_NAPS = int(os.getenv("CHLOE_DAILY_NAPS", "5"))
OVER_NAP_REPLY = "I've slept enough for one day. Ask me again tomorrow."

# Addresses the budget cannot use: behind the reverse proxy every request
# arrives from loopback, and the visitor's own address is only in
# X-Forwarded-For. Trusting that header is safe exactly because the listener
# is not reachable except through the proxy -- keep HOST bound to localhost.
_LOOPBACK = ("127.", "::1", "localhost")

# The only files a GET can reach, and what each is served as. Adding a page
# to the site means adding it here.
SITE_FILES = {
    "index.html": "text/html; charset=utf-8",
    "beliefs.html": "text/html; charset=utf-8",
    "chloe.css": "text/css; charset=utf-8",
    "script.js": "application/javascript; charset=utf-8",
    "beliefs.js": "application/javascript; charset=utf-8",
    "chloe-favicon.png": "image/png",
}

DREAM_START = os.getenv("DREAM_START", "22:00")
DREAM_DURATION_MINUTES = float(os.getenv("DREAM_DURATION_MINUTES", "15"))
DREAM_TZ = os.getenv("DREAM_TZ")
_dream_tzinfo = ZoneInfo(DREAM_TZ) if (DREAM_TZ and ZoneInfo) else None


def _dream_hour_minute() -> Tuple[int, int]:
    h, m = DREAM_START.split(":")
    return int(h), int(m)


def _now() -> datetime:
    return datetime.now(_dream_tzinfo) if _dream_tzinfo else datetime.now()


def _next_dream_window(now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """Returns (start, end) of the next dream window -- today's if it
    hasn't ended yet (including "we're currently inside it", e.g. right
    after the server starts), otherwise tomorrow's."""
    now = now or _now()
    hour, minute = _dream_hour_minute()
    start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    end = start + timedelta(minutes=DREAM_DURATION_MINUTES)
    if now >= end:
        start += timedelta(days=1)
        end = start + timedelta(minutes=DREAM_DURATION_MINUTES)
    return start, end

# One shared knowledge store, so knowledge accumulates across visitors;
# one ChloeEngine per browser session, so each visitor has their own
# Person, trust profile and turn history. ThreadingHTTPServer serves each
# request on its own thread and a single sqlite3 connection is not safe for
# unsynchronised concurrent use, so all access below holds _store_lock.
_store = KnowledgeStore(str(DB_PATH), check_same_thread=False)
_store_lock = threading.Lock()


class _LruTable:
    """A dict with a ceiling, keyed by something the caller supplies.

    Anything keyed by a session id or a client address is sized by whoever
    is calling rather than by how many people are here, so the bound belongs
    to the structure rather than to each use of it. Reading counts as use,
    so what gets dropped is always the least recently active.

    Nothing here touches the store, so this lock is always the inner one and
    can never be held while _store_lock is taken.
    """

    def __init__(self, capacity: int, label: str):
        self._capacity = max(1, capacity)
        self._label = label
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, dict]" = OrderedDict()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            item = self._items.get(key)
            if item is not None:
                self._items.move_to_end(key)
            return item

    def put(self, key: str, value: dict) -> dict:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                dropped, _ = self._items.popitem(last=False)
                print(f"[{self._label}] dropped {dropped[:12]} "
                      f"({len(self._items)}/{self._capacity} held)", file=sys.stderr)
            return value

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class _SessionRegistry:
    """Everything held per browser session, in one place.

    Two things used to be keyed by session_id independently: the engine, and
    whether that session had asked CHLOE to sleep and was owed its questions
    on waking. Both grew without limit, and the second was written from the
    request thread and read from the wake poll without a lock.
    """

    def __init__(self, capacity: int):
        self._table = _LruTable(capacity, "sessions")

    def engine(self, session_id: str) -> Optional[ChloeEngine]:
        entry = self._table.get(session_id)
        return entry["engine"] if entry else None

    def knows(self, session_id: str) -> bool:
        return self._table.has(session_id)

    def put(self, session_id: str, engine: ChloeEngine) -> None:
        self._table.put(session_id, {"engine": engine, "owed": False, "client": None})

    def set_client(self, session_id: str, client: Optional[str]) -> None:
        """The address the turn now in flight arrived from. The engine holds
        a nap callback that fires from inside turn(), by which point the
        request that carried the address is no longer in view -- and the
        address can change between turns, so capturing it when the engine is
        built would go stale. Every turn for a session is serialised by
        _store_lock, so at most one can be in flight here."""
        entry = self._table.get(session_id)
        if entry is not None:
            entry["client"] = client

    def client(self, session_id: str) -> Optional[str]:
        entry = self._table.get(session_id)
        return entry["client"] if entry else None

    def mark_owed(self, session_id: str) -> None:
        """This session asked CHLOE to sleep, so it is owed the questions
        waking produces. A session dropped before it wakes is simply not
        owed them any more."""
        entry = self._table.get(session_id)
        if entry is not None:
            entry["owed"] = True

    def owes_questions(self, session_id: str) -> bool:
        entry = self._table.get(session_id)
        return bool(entry and entry["owed"])

    def clear_owed(self, session_id: str) -> None:
        entry = self._table.get(session_id)
        if entry is not None:
            entry["owed"] = False


class _DailyBudget:
    """How much one address has spent today.

    The counter table is bounded like every other client-keyed structure
    here: past its ceiling the least recently seen address is dropped, which
    hands that address a fresh allowance. Cycling through enough addresses to
    cause that is already cycling through enough addresses to ignore a
    per-address budget, so the ceiling costs nothing the budget was buying.
    """

    def __init__(self, allowance: int, label: str, capacity: int = 10000):
        self._allowance = allowance
        self._table = _LruTable(capacity, label)

    def spend(self, client: Optional[str]) -> bool:
        """Charge this address one exchange. False when it has none left.

        An unusable address (the proxy's own, when X-Forwarded-For did not
        arrive) is not charged: a budget that cannot tell visitors apart
        would put every one of them in the same bucket, which is worse than
        no budget. _warn_if_unusable says so in the log instead.
        """
        if self._allowance <= 0 or client is None:
            return True
        today = _now().date().isoformat()
        entry = self._table.get(client)
        if entry is None or entry["day"] != today:
            entry = self._table.put(client, {"day": today, "spent": 0})
        if entry["spent"] >= self._allowance:
            return False
        entry["spent"] += 1
        return True


_sessions = _SessionRegistry(MAX_SESSIONS)
_budget = _DailyBudget(DAILY_EXCHANGES, "budget")
_naps = _DailyBudget(DAILY_NAPS, "naps")
_warned_no_xff = False


def _warn_if_unusable(client: Optional[str]) -> None:
    """Said once, not per request: a budget silently doing nothing is worse
    than one that is off, because it looks like it is on."""
    global _warned_no_xff
    if client is None and DAILY_EXCHANGES > 0 and not _warned_no_xff:
        _warned_no_xff = True
        print("[warn] every request arrives from loopback and no X-Forwarded-For "
              "was set, so the daily budget cannot tell visitors apart and is "
              "not being applied. Check the reverse proxy.", file=sys.stderr)


# Dreaming state, under its own lock so that /api/health never waits
# behind a long-held _store_lock.
_dream_lock = threading.Lock()
_dreaming = False
_last_dream_summary: Optional[str] = None
_last_dream_at: Optional[str] = None
_next_dream_start, _dream_window_end = _next_dream_window()

DREAMING_REPLY = "Zzz... Chloe is dreaming right now, consolidating what it has learned. Try again once it wakes up."
NAP_REPLY = "I'm going to sleep for a moment. Back shortly."


def _is_dreaming() -> bool:
    with _dream_lock:
        return _dreaming


def _dream_status() -> dict:
    with _dream_lock:
        return {
            "dreaming": _dreaming,
            "last_dream_summary": _last_dream_summary,
            "last_dream_at": _last_dream_at,
            "next_dream_start": _next_dream_start.isoformat(),
            "dream_ends_at": _dream_window_end.isoformat() if _dreaming else None,
        }


def _run_dream_window(end: datetime) -> None:
    """Runs the consolidation pass at the start of the window, then
    holds 'dreaming' open (real downtime, not a UI artifact) until `end`,
    since a fixed nightly window was chosen deliberately over a repeating
    interval -- see the DREAM_START comment above."""
    global _dreaming, _last_dream_summary, _last_dream_at
    with _dream_lock:
        _dreaming = True
    print(f"[dream] window started, waking at {end.isoformat()}", file=sys.stderr)
    try:
        with _store_lock:
            report = consolidation.sleep(_store)
        summary = report.summary()
    except Exception as e:  # a failed dream should never take the site down
        print(f"[error] dream pass failed: {e}", file=sys.stderr)
        summary = "Dreaming was interrupted by an internal error."
    remaining = (end - _now()).total_seconds()
    if remaining > 0:
        time.sleep(remaining)
    with _dream_lock:
        _dreaming = False
        _last_dream_summary = summary
        _last_dream_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[dream] Chloe woke up: {summary}", file=sys.stderr)


def _start_nap(session_id: str) -> bool:
    """Put CHLOE to sleep briefly, on request. Returns False if it is
    already dreaming. The window runs on its own thread so the request that
    asked for it returns at once and the browser can show the sleeping
    state, rather than hanging for the duration."""
    with _dream_lock:
        if _dreaming:
            return False
    end = _now() + timedelta(seconds=NAP_SECONDS)
    global _dream_window_end
    _dream_window_end = end
    _sessions.mark_owed(session_id)
    threading.Thread(target=_run_dream_window, args=(end,),
                     name="chloe-nap", daemon=True).start()
    return True


def _nap_reply(session_id: str) -> str:
    """What CHLOE says when asked to sleep: the one place the answer is
    decided, so the reason is never inferred from a bare False.

    A nap is charged to the address that asked for it, apart from the
    exchange the same turn already cost. Losing the race to the scheduler
    after being charged spends one of the day's naps on a sleep that was
    already happening -- rare, and cheaper than holding the nap budget
    inside the dream lock to prevent it.
    """
    if _is_dreaming():
        return DREAMING_REPLY
    if not _naps.spend(_sessions.client(session_id)):
        return OVER_NAP_REPLY
    if not _start_nap(session_id):
        return DREAMING_REPLY
    return NAP_REPLY


def _dream_scheduler() -> None:
    """Runs forever in a daemon thread, started by run(). Polls in short
    increments (rather than one long time.sleep) so it correctly notices
    when 'now' has reached the next window even across DST changes."""
    global _next_dream_start, _dream_window_end
    while True:
        start, end = _next_dream_window()
        _next_dream_start, _dream_window_end = start, end
        now = _now()
        if now < start:
            time.sleep(min((start - now).total_seconds(), 30))
            continue
        # Inside [start, end): either on schedule, or the server started
        # mid-window (a restart at 2:07am) and should serve out the rest of
        # it rather than skip the window.
        _run_dream_window(end)


def _get_or_init_engine(session_id: str, name: str) -> tuple[ChloeEngine, Optional[str]]:
    """Caller must hold _store_lock.

    Returns (engine, greet_reply). greet_reply is non-None only the first
    time an engine is created for this session_id: it is ChloeEngine's own
    greet() text, which may be asking to set up a secret word or asking a
    returning name to confirm theirs. Later calls for the same session_id
    find the engine already there and do not re-run greet(), which would
    re-trigger the secret-word exchange on every page reload.
    """
    engine = _sessions.engine(session_id)
    if engine is not None:
        return engine, None
    # "Sleep, Chloe" starts the dream window rather than the engine's own
    # synchronous sleep(), so the site visibly goes quiet and the questions
    # are put on waking. It still goes through engine.turn(), and is logged
    # like any other turn.
    engine = ChloeEngine(_store, nap=lambda: _nap_reply(session_id))
    reply = engine.greet(name)
    _sessions.put(session_id, engine)
    return engine, reply


def _evidence_row(p, people: dict) -> dict:
    """Caller must hold _store_lock. One piece of evidence, as the belief
    browser and the transcript show it. `note` says where evidence carried
    from another belief came from, and why a voided one no longer counts."""
    via = _store.atom_by_id(p.via_atom_id) if p.via_atom_id else None
    note = None
    if p.void:
        note = f"withdrawn: compatible with '{via.statement()}'" if via else "withdrawn: the belief it came from was removed"
    elif via is not None:
        note = f"via '{via.statement()}'"
    return {
        "person": people[p.person_id].name if p.person_id in people else f"#{p.person_id}",
        "effect": {1: "supports", -1: "disputes"}.get(p.polarity, str(p.polarity)),
        "at": p.at,
        "parse_confidence": p.parse_confidence,
        "void": p.void,
        "note": note,
    }


def collect_beliefs() -> dict:
    """Caller must hold _store_lock. The store as it stands, with the
    evidence behind each belief resolved to names."""
    people = {p.id: p for p in _store.all_people()}
    beliefs = []
    for atom in sorted(_store.all_atoms(), key=lambda a: (-a.confidence, a.subject.lower())):
        beliefs.append({
            "id": atom.id,
            "statement": atom.statement(),
            "subject": atom.subject, "relation": atom.relation, "object": atom.object,
            "scope": atom.scope, "domain": atom.domain,
            "status": atom.status.value, "confidence": round(atom.confidence, 3),
            "created_at": atom.created_at, "updated_at": atom.updated_at,
            "evidence": [_evidence_row(p, people) for p in atom.provenance],
        })
    return {
        "beliefs": beliefs,
        "people": [{"name": p.name, "trust": p.trust} for p in people.values()],
        "open_questions": [{"question": questions.render(_store, q), "reason": q["reason"]}
                           for q in _store.pending_questions() if questions.still_open(_store, q)],
    }


def collect_transcript(session_id: str, name: str) -> dict:
    """Caller must hold _store_lock. One visitor's logged conversation, plus
    the beliefs their turns produced -- the provenance record, not the
    browser's copy of the chat log, so it survives a page reload."""
    engine = _sessions.engine(session_id)
    person = engine.person if engine and engine.person else _store.find_person_by_name(name or "")
    if person is None:
        return {"person": None, "turns": [], "beliefs": [], "open_questions": []}

    people = {p.id: p for p in _store.all_people()}
    mine = []
    for atom in _store.all_atoms():
        if not any(p.person_id == person.id for p in atom.provenance):
            continue
        mine.append({
            "statement": atom.statement(),
            "status": atom.status.value,
            "confidence": round(atom.confidence, 3),
            "evidence": [_evidence_row(p, people) for p in atom.provenance],
        })
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "person": {"name": person.name, "trust": person.trust},
        "turns": [{"role": i.role, "text": i.text, "at": i.at}
                  for i in _store.interactions_for_person(person.id)
                  if i.role in CONVERSATION_ROLES],
        "beliefs": mine,
        # What CHLOE would ask this person now, worded as it would put it.
        "open_questions": [{"question": questions.render(_store, q, person), "reason": q["reason"]}
                           for q in _store.pending_questions()
                           if questions.still_open(_store, q) and questions.for_person(_store, q, person)],
    }


def handle_wake(session_id: str, name: str) -> dict:
    """What CHLOE has to say now that it is awake.

    Called by the browser once its health poll sees the dream window close.
    Only a session that actually asked it to sleep is owed this, so an
    unrelated visitor who happened to be online during the nightly window is
    not ambushed with questions.
    """
    # Still asleep: say nothing and keep the session on the list, so an
    # early poll can't consume the offer before it has actually woken.
    if _is_dreaming() or not _sessions.owes_questions(session_id):
        return {"session_id": session_id, "reply": None}
    _sessions.clear_owed(session_id)
    with _store_lock:
        engine = _sessions.engine(session_id)
        if engine is None or engine.person is None:
            return {"session_id": session_id, "reply": None}
        offer = engine.wake()
    return {"session_id": session_id, "reply": offer}


def handle_greet(payload: dict, client: Optional[str] = None) -> dict:
    """Explicit first-contact step: the website asks a visitor's real
    name before starting the chat proper (see script.js), and this is what
    turns that name into a genuine ChloeEngine.greet() call -- as opposed
    to the old behaviour of silently greeting a synthetic
    'web-visitor-<uuid>' name that the visitor never typed and never saw.
    """
    session_id = str(payload.get("session_id") or uuid.uuid4())
    name = (payload.get("name") or "").strip()
    if _is_dreaming():
        return {"session_id": session_id, "reply": DREAMING_REPLY, "dreaming": True}
    if not name:
        return {"session_id": session_id, "reply": "I didn't catch a name -- what should I call you?"}

    if not _sessions.knows(session_id) and not _budget.spend(client):
        # Opening a session with a name not on file writes a person row, so
        # it is charged. A reload carries its session_id and is not.
        return {"session_id": session_id, "reply": OVER_BUDGET_REPLY,
                "budget_exhausted": True}

    with _store_lock:
        already_had_engine = _sessions.knows(session_id)
        engine, reply = _get_or_init_engine(session_id, name)
        if reply is None:
            # Same session already greeted (e.g. the page was reloaded) --
            # don't make them re-answer the secret-word prompt every time.
            reply = f"Welcome back, {engine.person.name}." if engine.person else f"Welcome back, {name}."

    return {"session_id": session_id, "reply": reply, "already_greeted": already_had_engine}


def handle_chat(payload: dict, client: Optional[str] = None) -> dict:
    session_id = str(payload.get("session_id") or uuid.uuid4())
    message = (payload.get("message") or "").strip()
    name = (payload.get("name") or "").strip()
    if _is_dreaming():
        return {"session_id": session_id, "reply": DREAMING_REPLY, "dreaming": True, "llm_used": False}
    if not message:
        return {"session_id": session_id, "reply": "Say something and I'll respond.", "llm_used": False}

    if not _budget.spend(client):
        return {"session_id": session_id, "reply": OVER_BUDGET_REPLY,
                "budget_exhausted": True, "llm_used": False}

    with _store_lock:
        # /api/greet normally creates the engine before any chat message
        # arrives. This is the safety net for when it has not -- a server
        # restart, or an eviction, drops the in-memory engine -- so chat
        # does not hard-fail just because greet was not replayed.
        engine, _ = _get_or_init_engine(session_id, name or f"guest-{session_id[:8]}")
        _sessions.set_client(session_id, client)
        ground_truth = engine.turn(message)
        # Read inside the lock: it belongs to the turn just taken.
        stance = engine.last_stance
        question = engine.last_question

    if _is_dreaming():
        # The turn put CHLOE to sleep: said as it stands, and the browser
        # shows the sleeping state.
        return {"session_id": session_id, "reply": ground_truth, "dreaming": True, "llm_used": False}

    # A question CHLOE has just put goes out in the core's words: the
    # person's next "yes" or "no" is recorded against it, so it must reach
    # them as the question it is. Only what comes before it is phrased.
    to_phrase, kept = persona.question_to_keep(ground_truth, question)
    phrased = to_phrase
    llm_used = False
    rejected = None
    # The output interface is held to the input interface's length limit as
    # well: naturalise_request puts the person's own message in the user
    # turn, so phrasing a reply to a line the parser refused unread would
    # hand the model the very text the limit exists to keep from it. The
    # engine's refusal goes out as it stands.
    if to_phrase and llm_client.is_configured() and len(message) <= MAX_INPUT_CHARS:
        messages = persona.naturalise_request(message, to_phrase)
        try:
            candidate = llm_client.chat(
                messages, temperature=persona.NATURALISE_TEMPERATURE)
            # The model's licence is linguistic, not epistemic. A reply that
            # repeats the instructions, echoes the person, swaps the
            # speakers, supplies a name or number of its own or leaves out
            # one the core gave, answers a yes/no question the other way
            # from the core, or runs away is dropped for the engine's own
            # text -- the same fallback as an unreachable server.
            rejected = persona.rejection_reason(candidate, message, to_phrase,
                                                stance=stance)
            if rejected:
                print(f"[naturalisation rejected] {rejected}: {candidate[:160]!r}",
                      file=sys.stderr)
            else:
                phrased = candidate.strip()
                llm_used = True
        except llm_client.LLMUnavailable as e:
            print(f"[warn] vLLM unavailable, falling back to engine reply: {e}", file=sys.stderr)

    reply = "\n".join(part for part in (phrased, kept) if part)
    return {"session_id": session_id, "reply": reply, "engine_reply": ground_truth,
            "llm_used": llm_used, "naturalisation_rejected": rejected}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, rel_path: str):
        """Serve one of the site's own files, named in SITE_FILES.

        The rule is the table rather than the directory: SITE_DIR is the
        folder the server runs from, which holds the engine, the knowledge
        store and whatever else lives beside them, so "any regular file
        under SITE_DIR" is the wrong permission to hand a GET. A name is
        either the site's or it is not, and an unlisted one does not exist.
        """
        ctype = SITE_FILES.get(rel_path)
        if ctype is None:
            self.send_error(404)
            return
        path = SITE_DIR / rel_path
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # The site is edited and redeployed often; a cached stylesheet has
        # already been mistaken for a rendering bug. Revalidate every time.
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _client_key(self) -> Optional[str]:
        """The visitor's own address, or None when it cannot be known.

        Apache appends the address it observed to X-Forwarded-For, so the
        last entry is the one it saw and the only one a caller cannot
        forge -- anything a caller puts in the header is pushed left of it.
        With no header and a loopback peer we are behind the proxy and
        blind, which None says plainly rather than lumping every visitor
        under the proxy's address.
        """
        forwarded = self.headers.get("X-Forwarded-For", "")
        if forwarded:
            hops = [h.strip() for h in forwarded.split(",") if h.strip()]
            if hops:
                return hops[-1]
        peer = self.client_address[0] if self.client_address else ""
        if not peer or peer.startswith(_LOOPBACK):
            return None
        return peer

    def _read_json_body(self) -> Optional[dict]:
        """The JSON body of this request, or None when this has already
        answered the caller. The declared length is checked before a byte is
        read: Content-Length is supplied by whoever is calling, and reading
        it blindly is how one request becomes a memory problem."""
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return None
        if length < 0 or length > MAX_BODY_BYTES:
            self._send_json(413, {"error": f"body may not exceed {MAX_BODY_BYTES} bytes"})
            return None
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "invalid JSON body"})
            return None

    def _beliefs_authorised(self) -> bool:
        """Constant-time check of the Basic credentials. Returns False and
        sends the challenge itself when they are missing or wrong."""
        header = self.headers.get("Authorization", "")
        scheme, _, encoded = header.partition(" ")
        supplied = ""
        if scheme.lower() == "basic":
            try:
                supplied = b64decode(encoded, validate=True).decode("utf-8", "replace")
            except Exception:
                supplied = ""
        expected = f"{BELIEFS_USER}:{BELIEFS_PASSWORD}"
        if supplied and hmac.compare_digest(supplied, expected):
            return True
        body = b"Authentication required.\n"
        self.send_response(401)
        self.send_header("WWW-Authenticate", f'Basic realm="{BELIEFS_REALM}", charset="UTF-8"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    def do_GET(self):
        route = self.path.split("?", 1)[0]

        if route == "/api/health":
            payload = {"ok": True, "vllm_configured": llm_client.is_configured(),
                       "max_input_chars": MAX_INPUT_CHARS}
            payload.update(_dream_status())
            self._send_json(200, payload)
            return

        if route == "/api/wake":
            params = parse_qs(urlparse(self.path).query)
            self._send_json(200, handle_wake((params.get("session_id") or [""])[0],
                                             (params.get("name") or [""])[0]))
            return

        if route == "/api/transcript":
            params = parse_qs(urlparse(self.path).query)
            session_id = (params.get("session_id") or [""])[0]
            name = (params.get("name") or [""])[0]
            with _store_lock:
                data = collect_transcript(session_id, name)
            self._send_json(200, data)
            return

        if route in ("/api/beliefs", "/beliefs", "/beliefs.html"):
            # Unconfigured means absent, not unprotected: no 403 to advertise
            # that there is something here worth guessing a password for.
            if not BELIEFS_PASSWORD:
                self.send_error(404)
                return
            if not self._beliefs_authorised():
                return
            if route == "/api/beliefs":
                with _store_lock:
                    self._send_json(200, collect_beliefs())
            else:
                self._send_file("beliefs.html")
            return

        if route in ("/", ""):
            route = "/index.html"
        self._send_file(route.lstrip("/"))

    def do_POST(self):
        if self.path == "/api/greet":
            handler = handle_greet
        elif self.path == "/api/chat":
            handler = handle_chat
        else:
            self.send_error(404)
            return
        payload = self._read_json_body()
        if payload is None:
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "expected a JSON object"})
            return
        client = self._client_key()
        _warn_if_unusable(client)
        try:
            result = handler(payload, client)
        except Exception as e:  # keep the site up even if something breaks
            print(f"[error] {self.path} failed: {e}", file=sys.stderr)
            self._send_json(500, {"error": "internal error"})
            return
        self._send_json(200, result)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")


def run():
    if not llm_client.is_configured():
        print("[warn] VLLM_BASE_URL is not set -- chat will use CHLOE's raw engine "
              "replies without LLM phrasing until it is.", file=sys.stderr)
    threading.Thread(target=_dream_scheduler, name="chloe-dream-scheduler", daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"CHLOE web chat serving on http://{HOST}:{PORT}  (knowledge db: {DB_PATH})")
    window_end_clock = _dream_window_end.strftime("%H:%M")
    print(f"CHLOE dreams daily {DREAM_START}-{window_end_clock} ({DREAM_TZ or 'server local time'}); "
          f"next window starts {_next_dream_start.isoformat()}", file=sys.stderr)
    if DAILY_EXCHANGES > 0:
        naps = f"{DAILY_NAPS} of them sleeps" if DAILY_NAPS > 0 else "sleeps unlimited"
        print(f"Daily budget: {DAILY_EXCHANGES} exchanges per address, {naps} "
              f"(set CHLOE_DAILY_EXCHANGES=0 to disable).", file=sys.stderr)
    else:
        print("[warn] CHLOE_DAILY_EXCHANGES is 0 -- no per-address limit.", file=sys.stderr)
    if BELIEFS_PASSWORD:
        print(f"Belief browser at /beliefs (user {BELIEFS_USER!r}); "
              f"put it behind HTTPS -- basic auth is not encrypted.", file=sys.stderr)
    else:
        print("[warn] CHLOE_BELIEFS_PASSWORD is not set -- /beliefs and /api/beliefs "
              "are disabled.", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _store.close()


if __name__ == "__main__":
    run()
