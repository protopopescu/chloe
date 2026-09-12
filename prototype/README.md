# CHLOE — 2026 prototype

A working revival of the architecture described in the original design
notes and the accompanying paper (project started in 2000 by
D. Protopopescu). Pure
Python standard library, no external dependencies, no API keys required.
Optionally, an OpenAI-compatible LLM endpoint (vLLM) can be attached at
both linguistic interfaces — input parsing and output phrasing — see
"Using an LLM at the linguistic interfaces" below; without it, everything
still runs fully offline.

## Run it

```
python3 -m chloe
```

Talk to it: state facts (`the sky is blue`), ask questions
(`what is the sky?`, `is the sky blue?`), or use commands: `sleep`,
`what do you know`, `who do you trust`, `show open questions`, `exit`.

Run a second session as a different name and tell it something that
contradicts the first session — CHLOE will flag the contradiction, adjust
trust, and queue a question to ask about it next time.

## How this maps to the original design

| Design concept | This prototype |
|---|---|
| Knowledge as inspectable "atoms of reasoning" | `models.Atom` — subject/relation/object, never a latent vector |
| Facts have explicit scope, not universal truth | `Atom.scope`, populated from `when`/`if` clauses |
| Provenance for every belief | `models.Provenance`, linked to a logged `Interaction` |
| Domain-specific trust, not one global score | `Person.trust` is a dict keyed by domain |
| Conversation as evidence acquisition | `dialogue.ChloeEngine.turn()` |
| First and second person resolved to who is speaking | `grammar.py` — ports the 2000 `SwapPronoun()`/`AccordTheVerb()`: "I"/"you" resolve to the speaker and to Chloe on the way in, and render back as "you"/"I" with the copula agreed on the way out |
| Sleep / offline consolidation | `consolidation.sleep()` — merges duplicates, flags contradictions, generates hypotheses, queues verification questions |
| Hypotheses become knowledge only via verification | `AtomStatus.HYPOTHESIS` → `CANDIDATE`/`CONFIRMED` only after a person confirms it in a later `turn()` |
| Never assume transitivity/symmetry from language alone | `RelationProperties` — a relation is only used to derive hypotheses if explicitly declared via `store.declare_relation(...)` (see `--declare-transitive` CLI flag) |
| LLM as the linguistic layer, symbolic core as authority | Two symmetric interfaces. **Input:** `llm_nlu.parse(text) -> Utterance` asks the configured LLM to read free text into a structured utterance of a declared type, refusing rather than guessing; the original pattern parser (`nlu.parse`) remains as offline fallback. **Output:** `llm_client.py` + `persona.py` phrase the engine's already-decided factual reply naturally. In both directions the symbolic core stays the authority: the LLM never adds facts, never issues commands, never writes to the store |

## What's deliberately NOT here yet

- **Web browsing** (the design's "navigate the internet and read webpages") — no network access built in.
- **Self-recompilation** (the original `chloe.sh` exit-code-66 respawn loop) — not meaningful for a Python prototype; the closest modern analogue would be CHLOE proposing its own code changes as a diff for review, which is a separate, larger feature.
- **Multi-instance internal dialogue** (the 2000 notes' "dialogue between instances with different experiences") — the data model (per-person provenance, domain-specific trust) supports this; there's just no orchestration of multiple CHLOE instances yet.

## File layout

```
chloe/
  models.py         data model: Atom, Person, Provenance, Interaction, RelationProperties
  storage.py        SQLite-backed KnowledgeStore (replaces the original flat .cw/.xr files)
  nlu.py            pattern-based sentence -> Utterance (offline fallback + command table)
  llm_nlu.py        LLM-based input parser behind the same parse() interface (LLM-first routing)
  grammar.py        pronoun resolution and copula agreement (ported from cGrammar.hh)
  llm_client.py     stdlib client for an OpenAI-compatible /chat/completions endpoint
  persona.py        system prompt for the output-side naturalisation layer
  trust.py          trust updates + confidence scoring
  dialogue.py        ChloeEngine: the conversational loop
  consolidation.py   sleep(): contradiction detection, dedup, hypothesis generation
  cli.py / __main__.py   `python -m chloe`
inspect_db.py       read-only inspector for any CHLOE store (no server needed)
test_llm_parser.py  offline test suite for the LLM parser (uses a local mock server)
test_pronouns.py    offline test suite for grammar.py and the pronoun path
test_inspect.py     offline test suite for the transcript/inspection queries
```

## Using an LLM at the linguistic interfaces

Set the same environment variables the web deployment already uses:

```
export VLLM_BASE_URL=http://<host>:8000/v1     # enables BOTH interfaces
export VLLM_API_KEY=...                        # only if the server needs one
export VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct     # optional, this is the default
export CHLOE_PARSE_MIN_CONFIDENCE=0.6          # optional refusal threshold
```

With `VLLM_BASE_URL` set, `dialogue.ChloeEngine` routes input through
`llm_nlu.parse()`: commands are intercepted deterministically first (the
LLM can never issue one), everything else is read by the model into a
structured utterance and strictly validated. A valid refusal — the model
saying "unknown", or reporting confidence below the threshold — is final:
it is not retried against the pattern parser, because refuse-rather-than-
guess would mean nothing if a refusal fell through to a sloppier parser.
Only a broken exchange (server unreachable, malformed reply) falls back
to `nlu.parse()`, mirroring how the output side falls back to the
engine's plain reply.

Each LLM parse's self-reported confidence is recorded in the belief's
`Provenance.parse_confidence` — inspectable, but deliberately kept out of
the belief-confidence maths, so "CHLOE misunderstood you" stays
distinguishable from "your source was wrong". Pattern-parser rows store
no number there.

Privacy note: with a server configured, ordinary conversational input is
sent to that endpoint. Secret words are not — the identity flow consumes
them before parsing.

Test the parser offline (no server, no network — it starts its own mock):

```
python3 test_llm_parser.py
```

## Inspecting a store

Every belief is a row you can read, with the evidence that put it there.
`inspect_db.py` prints that for any store this project writes — the CLI's
`chloe_data.db`, the web server's `uni_chat.db`, `demo_chloe.db` — and never
writes to it:

```
python3 inspect_db.py demo_chloe.db                 # beliefs, strongest first
python3 inspect_db.py demo_chloe.db --people        # interlocutors and per-domain trust
python3 inspect_db.py demo_chloe.db --questions     # what CHLOE still wants to know
python3 inspect_db.py demo_chloe.db --transcript Dan
python3 inspect_db.py demo_chloe.db --all --json
```

## Try the scripted demo

`demo.py` runs a scripted multi-person session (including a deliberate
contradiction between two sources) against a throwaway database, then
triggers `sleep` and prints the resulting report and the full knowledge
state — a quick way to see the whole cycle end to end without typing.

```
python3 demo.py
```
