# CHLOE

**Computer-Human Language Oriented Experiment** — a persistent epistemic
architecture for inspectable AI.

CHLOE treats a large language model as a *faculty* of an agent, not the
agent itself. The LLM handles language at two replaceable interfaces (reading
free text into structured utterances, and phrasing decided replies
naturally), while a persistent symbolic core remains the authority on what
the agent believes: every belief is an inspectable atom with explicit
provenance, scope, per-domain source trust and confidence. Beliefs are
revised across sessions, contradictions are flagged rather than
overwritten, and an offline consolidation cycle — "sleep" — periodically
merges duplicates, generates hypotheses, and queues verification questions
for future conversations. Hypotheses become knowledge only after
verification.

The project was started in 2000, when the available tools could not meet
the design; the original C++ implementation is preserved here as
provenance. The architecture was revived and implemented in 2026, once
LLMs had solved the language problem the original attempt foundered on.

This repository accompanies the paper *CHLOE: A Persistent Epistemic Architecture for Inspectable AI*
(D. Protopopescu, 2026), <https://doi.org/10.5281/zenodo.22862833>.

## Layout

```
prototype/       the 2026 implementation: pure-stdlib Python, SQLite-backed.
                 Runs fully offline; optionally attaches an OpenAI-compatible
                 LLM endpoint at both linguistic interfaces. Includes a scripted
                 demo and two offline test suites.
web/             a deployable web chat front end for the same engine, with
                 visitor identity binding and a scheduled nightly "dreaming"
                 window. Stdlib-only server.
original-2000/   the first implementation (C++, 2000), verbatim, including
                 the compile-and-respawn supervisor script that was the
                 original self-modification mechanism.
```

Each directory has its own README with details and run instructions.

## Quick start

No dependencies beyond Python 3 — nothing to install:

```
cd prototype
python3 demo.py             # scripted end-to-end walkthrough
python3 -m chloe            # interactive session
python3 -m unittest discover -s tests -t .   # every test suite, offline
```

To attach a language model at the linguistic interfaces, or to run the web
front end, see `prototype/README.md` and `web/README.md`.

## Licence

Apache License 2.0 — see [LICENSE](LICENSE). To cite this work, see
[CITATION.cff](CITATION.cff).
