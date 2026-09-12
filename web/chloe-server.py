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
                   engine in ../prototype/chloe and then, if a vLLM server
                   is configured, phrased naturally.

Also runs "dreaming": a background thread that once a day, during a fixed
window (DREAM_START for DREAM_DURATION_MINUTES, default 02:00 for 15
minutes), takes the site offline for chat and runs
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

from chloe import consolidation, llm_client, persona  # noqa: E402
from chloe.dialogue import CONVERSATION_ROLES, ChloeEngine  # noqa: E402
from chloe.storage import KnowledgeStore  # noqa: E402

DB_PATH = SITE_DIR / "uni_chat.db"
PORT = int(os.getenv("PORT", "8765"))
HOST = os.getenv("HOST", "0.0.0.0")

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

DREAM_START = os.getenv("DREAM_START", "02:00")
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
_engines: dict[str, ChloeEngine] = {}
_histories: dict[str, list] = {}
MAX_HISTORY_TURNS = 8

# Dreaming state, under its own lock so that /api/health never waits
# behind a long-held _store_lock.
_dream_lock = threading.Lock()
_dreaming = False
_last_dream_summary: Optional[str] = None
_last_dream_at: Optional[str] = None
_next_dream_start, _dream_window_end = _next_dream_window()

DREAMING_REPLY = "Zzz... Chloe is dreaming right now, consolidating what she's learned. Try again once she wakes up."


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
    engine = _engines.get(session_id)
    if engine is not None:
        return engine, None
    engine = ChloeEngine(_store)
    reply = engine.greet(name)
    _engines[session_id] = engine
    _histories[session_id] = []
    return engine, reply


def collect_beliefs() -> dict:
    """Caller must hold _store_lock. The store as it stands, with the
    evidence behind each belief resolved to names."""
    people = {p.id: p for p in _store.all_people()}
    effect = {1: "supports", -1: "disputes"}
    beliefs = []
    for atom in sorted(_store.all_atoms(), key=lambda a: (-a.confidence, a.subject.lower())):
        beliefs.append({
            "id": atom.id,
            "statement": atom.statement(),
            "subject": atom.subject, "relation": atom.relation, "object": atom.object,
            "scope": atom.scope, "domain": atom.domain,
            "status": atom.status.value, "confidence": round(atom.confidence, 3),
            "created_at": atom.created_at, "updated_at": atom.updated_at,
            "evidence": [{
                "person": people[p.person_id].name if p.person_id in people else f"#{p.person_id}",
                "effect": effect.get(p.polarity, str(p.polarity)),
                "at": p.at,
                "parse_confidence": p.parse_confidence,
            } for p in atom.provenance],
        })
    return {
        "beliefs": beliefs,
        "people": [{"name": p.name, "trust": p.trust} for p in people.values()],
        "open_questions": [{"question": q["question"], "reason": q["reason"]}
                           for q in _store.pending_questions()],
    }


def collect_transcript(session_id: str, name: str) -> dict:
    """Caller must hold _store_lock. One visitor's logged conversation, plus
    the beliefs their turns produced -- the provenance record, not the
    browser's copy of the chat log, so it survives a page reload."""
    engine = _engines.get(session_id)
    person = engine.person if engine and engine.person else _store.find_person_by_name(name or "")
    if person is None:
        return {"person": None, "turns": [], "beliefs": [], "open_questions": []}

    people = {p.id: p for p in _store.all_people()}
    effect = {1: "supports", -1: "disputes"}
    mine = []
    for atom in _store.all_atoms():
        if not any(p.person_id == person.id for p in atom.provenance):
            continue
        mine.append({
            "statement": atom.statement(),
            "status": atom.status.value,
            "confidence": round(atom.confidence, 3),
            "evidence": [{
                "person": people[p.person_id].name if p.person_id in people else f"#{p.person_id}",
                "effect": effect.get(p.polarity, str(p.polarity)),
                "at": p.at,
                "parse_confidence": p.parse_confidence,
            } for p in atom.provenance],
        })
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "person": {"name": person.name, "trust": person.trust},
        "turns": [{"role": i.role, "text": i.text, "at": i.at}
                  for i in _store.interactions_for_person(person.id)
                  if i.role in CONVERSATION_ROLES],
        "beliefs": mine,
        "open_questions": [{"question": q["question"], "reason": q["reason"]}
                           for q in _store.pending_questions()],
    }


def handle_greet(payload: dict) -> dict:
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

    with _store_lock:
        already_had_engine = session_id in _engines
        engine, reply = _get_or_init_engine(session_id, name)
        if reply is None:
            # Same session already greeted (e.g. the page was reloaded) --
            # don't make them re-answer the secret-word prompt every time.
            reply = f"Welcome back, {engine.person.name}." if engine.person else f"Welcome back, {name}."

    return {"session_id": session_id, "reply": reply, "already_greeted": already_had_engine}


def handle_chat(payload: dict) -> dict:
    session_id = str(payload.get("session_id") or uuid.uuid4())
    message = (payload.get("message") or "").strip()
    name = (payload.get("name") or "").strip()
    if _is_dreaming():
        return {"session_id": session_id, "reply": DREAMING_REPLY, "dreaming": True, "llm_used": False}
    if not message:
        return {"session_id": session_id, "reply": "Say something and I'll respond.", "llm_used": False}

    with _store_lock:
        # /api/greet normally creates the engine before any chat message
        # arrives. This is the safety net for when it has not -- a server
        # restart mid-session wipes the in-memory _engines dict -- so chat
        # does not hard-fail just because greet was not replayed.
        engine, _ = _get_or_init_engine(session_id, name or f"guest-{session_id[:8]}")
        ground_truth = engine.turn(message)

    reply = ground_truth
    llm_used = False
    if llm_client.is_configured():
        history = _histories[session_id]
        messages = [{"role": "system", "content": persona.system_prompt()}]
        messages.extend(history[-MAX_HISTORY_TURNS:])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"The person just said: {message!r}\n\n"
                    f"Ground truth (what actually happened / what you know -- "
                    f"do not contradict or add to this): {ground_truth!r}\n\n"
                    f"Reply to the person now, phrasing that ground truth naturally."
                ),
            }
        )
        try:
            reply = llm_client.chat(messages)
            llm_used = True
        except llm_client.LLMUnavailable as e:
            print(f"[warn] vLLM unavailable, falling back to engine reply: {e}", file=sys.stderr)
            reply = ground_truth

    history = _histories[session_id]
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    del history[: max(0, len(history) - MAX_HISTORY_TURNS * 2)]

    return {"session_id": session_id, "reply": reply, "engine_reply": ground_truth, "llm_used": llm_used}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, rel_path: str):
        path = (SITE_DIR / rel_path).resolve()
        if SITE_DIR not in path.parents and path != SITE_DIR:
            self.send_error(403)
            return
        if not path.is_file():
            self.send_error(404)
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # The site is edited and redeployed often; a cached stylesheet has
        # already been mistaken for a rendering bug. Revalidate every time.
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

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
            payload = {"ok": True, "vllm_configured": llm_client.is_configured()}
            payload.update(_dream_status())
            self._send_json(200, payload)
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
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        try:
            result = handler(payload)
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
