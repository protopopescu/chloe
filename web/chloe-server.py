#!/usr/bin/env python3
"""
CHLOE web chat server.

Serves this folder's static site (index.html/chloe.css/script.js) and two
JSON endpoints:
  POST /api/greet  -- first contact for a browser session: takes the real
                       name the visitor typed and runs it through
                       ChloeEngine.greet(), which may itself ask to set up
                       (or confirm) a secret word -- see
                       prototype/chloe/dialogue.py's AuthState.
  POST /api/chat    -- every message after that, run through the CHLOE
                       prototype's symbolic engine (../prototype/chloe)
                       and then, if a vLLM server is configured, phrased
                       naturally (see prototype/chloe/persona.py for why
                       it's split this way).

Also runs "dreaming": a background thread that, once a day during a fixed
window (DREAM_START for DREAM_DURATION_MINUTES -- default 02:00 for 15
minutes, server-local time), takes the whole site offline for chat and
runs chloe.consolidation.sleep() over the shared knowledge store --
merging duplicate atoms, flagging contradictions, generating hypotheses
for declared-transitive relations, and queuing verification questions.
This was part of the original design ("sleep produced
understanding") but wasn't previously scheduled -- it only ran when a
person typed the "sleep" command mid-conversation. See DREAM_* env vars
below.

Stdlib only -- no pip install needed to run this file itself. The chloe
package it imports is also stdlib-only.

The only thing this file talks to over the network is whatever
OpenAI-compatible server VLLM_BASE_URL points at (see
prototype/chloe/llm_client.py); with no VLLM_BASE_URL set it makes no
network requests at all beyond serving HTTP itself.

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
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # stdlib since Python 3.9
except ImportError:  # pragma: no cover
    ZoneInfo = None

SITE_DIR = Path(__file__).resolve().parent
PROTOTYPE_DIR = SITE_DIR.parent / "prototype"
sys.path.insert(0, str(PROTOTYPE_DIR))

from chloe import consolidation, llm_client, persona  # noqa: E402
from chloe.dialogue import ChloeEngine  # noqa: E402
from chloe.storage import KnowledgeStore  # noqa: E402

DB_PATH = SITE_DIR / "uni_chat.db"
PORT = int(os.getenv("PORT", "8765"))
HOST = os.getenv("HOST", "0.0.0.0")

# CHLOE dreams once a day, in a fixed window -- not on a repeating
# interval -- mirroring an actual nightly downtime rather than a periodic
# maintenance tick. DREAM_START is "HH:MM" in DREAM_TZ (default: the
# server process's own local time zone; set DREAM_TZ, e.g. "Europe/London",
# if the server doesn't run in the time zone "night" should mean). The
# underlying consolidation.sleep() pass finishes in well under a second
# even with a lot of knowledge, so for most of DREAM_DURATION_MINUTES
# CHLOE genuinely has nothing left to do -- the window is honored as real
# downtime anyway, matching the original single-threaded design's
# distinct offline phase, rather than waking back up early.
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

# One shared knowledge store (so CHLOE accumulates knowledge across every
# visitor, per the original design), one ChloeEngine per browser session so
# each visitor gets their own Person/trust identity and turn history.
# ThreadingHTTPServer handles each request on its own thread, and a single
# sqlite3 connection isn't safe for unsynchronized concurrent use, so all
# store/engine access below is serialized with _store_lock.
_store = KnowledgeStore(str(DB_PATH), check_same_thread=False)
_store_lock = threading.Lock()
_engines: dict[str, ChloeEngine] = {}
_histories: dict[str, list] = {}
MAX_HISTORY_TURNS = 8

# Dreaming state, guarded by its own lock (kept separate from _store_lock so
# a status check via /api/health is never stuck waiting behind a long-held
# store lock).
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
    """Runs the consolidation pass once at the start of the window, then
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
    """Runs forever in a daemon thread, started from run(). Polls in short
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
        # now is inside [start, end) -- either right on schedule, or the
        # server started mid-window (e.g. a restart at 2:07am) and should
        # resume dreaming for whatever's left rather than skip the window.
        _run_dream_window(end)


def _get_or_init_engine(session_id: str, name: str) -> tuple[ChloeEngine, Optional[str]]:
    """Caller must hold _store_lock.

    Returns (engine, greet_reply). greet_reply is only non-None the first
    time an engine is created for this session_id -- that's ChloeEngine's
    real greet() text, which (per the secret-word identity feature) may
    already be asking to set up a secret word or asking a returning name to
    confirm theirs. On every later call for the same session_id the engine
    already exists, so greet() is not re-run (that would re-trigger the
    secret-word dance on every page reload) and greet_reply is None.
    """
    engine = _engines.get(session_id)
    if engine is not None:
        return engine, None
    engine = ChloeEngine(_store)
    reply = engine.greet(name)
    _engines[session_id] = engine
    _histories[session_id] = []
    return engine, reply


def handle_greet(payload: dict) -> dict:
    """Explicit first-contact step: the website now asks a visitor's real
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
        # Normally /api/greet has already created the engine by the time any
        # chat message arrives. This is only a safety net -- e.g. the server
        # process restarted mid-session and wiped the in-memory _engines
        # dict -- so chat never hard-fails just because greet wasn't replayed.
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
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/health":
            payload = {"ok": True, "vllm_configured": llm_client.is_configured()}
            payload.update(_dream_status())
            self._send_json(200, payload)
            return
        route = self.path.split("?", 1)[0]
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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _store.close()


if __name__ == "__main__":
    run()
