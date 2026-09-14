# CHLOE — deployable web chat

A self-contained site (index.html/chloe.css/script.js) plus a small stdlib
Python server (`chloe-server.py`) that runs the chat through the CHLOE prototype's
symbolic engine (`../prototype/chloe`) and, when a vLLM server is
configured, asks it to phrase the reply naturally.

The server's only network dependency is optional: a runtime HTTP request
to whatever OpenAI-compatible endpoint `VLLM_BASE_URL` points at (a hosted
vLLM instance, or any server speaking the same `/chat/completions`
protocol). Without it, everything runs fully offline.

## Run it

```
cd web
export VLLM_BASE_URL=http://<your-llm-host>:8000/v1
# export VLLM_API_KEY=...          # only if the server requires one
# export VLLM_MODEL=...                        # optional, has a default
python3 chloe-server.py
```

Then open `http://localhost:8765` (or whatever `HOST`/`PORT` you set).

The site's nav has a "Paper" link pointing at `CHLOE.pdf` alongside
`index.html`. The PDF is not in this repository; drop a copy in this folder
if you want that link to resolve, or remove the link from `index.html`.

If `VLLM_BASE_URL` isn't set, the site still works — chat replies just come
straight from the symbolic engine's own text instead of being phrased by
the model, and the status pill reads "engine only" rather than "online".

## Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `VLLM_BASE_URL` | for LLM phrasing | — | e.g. `http://<your-llm-host>:8000/v1` |
| `VLLM_API_KEY` | no | — | only if the server requires auth |
| `VLLM_MODEL` | no | see `DEFAULT_MODEL` | must match what the server is actually serving |
| `VLLM_TIMEOUT` | no | `30` | seconds |
| `HOST` | no | `0.0.0.0` | |
| `PORT` | no | `8765` | |
| `DREAM_START` | no | `02:00` | daily start time (`HH:MM`, 24h) of CHLOE's dream window |
| `DREAM_DURATION_MINUTES` | no | `15` | how long the dream window lasts |
| `DREAM_TZ` | no | server's local time zone | IANA zone name, e.g. `Europe/London`, if the server doesn't already run in the zone "night" should mean |
| `CHLOE_BELIEFS_PASSWORD` | for the belief browser | — | **unset means `/beliefs` and `/api/beliefs` do not exist**, rather than existing unprotected |
| `CHLOE_BELIEFS_USER` | no | `chloe` | username for the same |

## Downloading a conversation

The "download" control in the chat header saves the conversation as JSON.
It asks the server for `/api/transcript?session_id=...`, which returns the
*provenance record* rather than the browser's copy of the chat log: every
turn as it was logged, with timestamps, plus the beliefs those turns
produced, the evidence behind each one (including other visitors who have
since corroborated or disputed it), and CHLOE's open questions. That
survives a page reload; the on-screen log does not.

If the server can't be reached, the button falls back to dumping what is
visible in the chat window, the same way chat falls back to the engine's own
reply when the language model is unreachable.

## Browsing the belief store

`/beliefs` is a read-only view of everything CHLOE holds: each belief with
its scope, status, confidence and the people who supported or disputed it,
plus per-domain trust and the open questions. `/api/beliefs` returns the
same as JSON.

**Both are disabled unless `CHLOE_BELIEFS_PASSWORD` is set** — with no
password they return 404 rather than 403, so the site does not advertise
that there is anything there. With one set, they are behind HTTP basic auth:

```
export CHLOE_BELIEFS_PASSWORD='...'     # choose your own; never commit it
export CHLOE_BELIEFS_USER=chloe         # optional, this is the default
```

Basic auth sends the password in a reversible encoding on every request, so
this only means anything behind the HTTPS reverse proxy described under
*Deploying elsewhere*. It keeps casual visitors out of the store; it is not
a security boundary. For offline analysis use `../prototype/inspect_db.py`
instead, which needs no server at all.

## How a message flows

1. Browser POSTs `{session_id, message}` to `/api/chat`.
2. `chloe-server.py` runs it through `ChloeEngine.turn()` (`../prototype/chloe/dialogue.py`)
   — this is the authority on what CHLOE actually believes: it stores new
   facts, flags contradictions, updates trust, and produces a short factual
   reply.
3. If `VLLM_BASE_URL` is set, that factual reply is sent to the vLLM server
   (`chloe/llm_client.py`, `chloe/persona.py`) with instructions to phrase it
   naturally without adding or contradicting anything. No conversation
   history goes with it: the model is wording one sentence it was handed, not
   holding up its end of the dialogue. The phrasing it returns is checked
   before it is shown — a reply that quotes the instructions or the visitor,
   swaps the speakers, runs away in length, or introduces a name or number the
   engine did not supply is discarded. If the call fails for any reason
   (server down, timeout, bad response) or the check rejects the phrasing, the
   raw engine reply is used instead — chat never goes down just because the
   model is unreachable.
4. The reply is returned as JSON and appended to the chat log in the browser.

## Dreaming

Once a day, during a fixed window (`DREAM_START` for `DREAM_DURATION_MINUTES`
— default 02:00-02:15, server-local time unless `DREAM_TZ` is set), the
server takes the whole site offline for chat -- `/api/chat` and
`/api/greet` both reply with a "Chloe is dreaming" message instead of
processing the request -- and runs `chloe.consolidation.sleep()` over the
shared knowledge store: merging duplicate atoms, flagging contradictions,
generating hypotheses for declared-transitive relations, and queuing
verification questions for the next conversation. The consolidation pass
itself finishes in well under a second; the rest of the window is honoured
as real downtime anyway, matching a genuine nightly "sleep" rather than a
brief maintenance tick. `GET /api/health` reports `dreaming`,
`last_dream_summary`, `next_dream_start`, and (while dreaming) `dream_ends_at`
so the frontend can show the state; `script.js` polls it every few seconds.
If the server restarts mid-window, it resumes dreaming for whatever time
is left rather than skipping the rest of that night's window.

This is separate from (and in addition to) the manual `sleep` command a
person can still type mid-conversation, which runs the same underlying
`consolidation.sleep()` immediately for that one interaction, without
taking the rest of the site offline.

## Data

Knowledge and per-visitor trust accumulate in `uni_chat.db` (SQLite) in this
folder, created on first run — shared across all visitors, same as
`prototype/chloe_data.db` for the CLI prototype, but a separate file so the
two don't collide.

## Deploying elsewhere

This folder assumes `../prototype/chloe` is present as a sibling directory
(that's how `chloe-server.py` finds the engine). If you copy this folder
somewhere without the rest of the CHLOE repo, bring `prototype/chloe/`
along too, or vendor it into this folder.

For anything beyond a prototype/demo deployment (real traffic, HTTPS,
process supervision), put this behind a reverse proxy and a process
manager (systemd, etc.) — `chloe-server.py`
here is deliberately minimal (Python stdlib `http.server`), not
production-hardened.
