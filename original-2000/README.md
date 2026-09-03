# CHLOE — original implementation (2000)

This directory preserves, verbatim, the first implementation of CHLOE:
roughly 1,850 lines of C++ written in 2000, when the project was started.
It is kept as historical provenance for the architecture described in the
accompanying paper, not as code anyone should build on today. Nothing here
has been modernised or cleaned up — fixed buffers, hardcoded grammar and
all. That is the point: it documents what the design looked like when the
available tools could not yet meet it.

What it implemented: an introduction dialogue; parsing of simple "X is Y"
sentences into word associations; a per-letter flat-file word bank
(`dict/*.cw`) and association files (`xref/*.xr`) in place of a database;
a trust flag for a designated trusted interlocutor; meaning lookup; and
session logging.

`chloe.sh` is worth a look: a tcsh supervisor script that compiles and
runs `cMain`, and — whenever the program exits with code 66 — recompiles
and respawns it. This was the original self-modification mechanism: the
program could request its own rebuild.

Layout:

```
cMain.cc     entry point
chloe.sh     compile/run/respawn supervisor (exit code 66 = rebuild me)
libs/        cBase, cData, cGrammar, cIO, cMisc, cPerson, cSentence, cWord
```

The compiled binary and editor backup files from the original directory
are not included. The modern reimplementation of the same architecture
lives in `../prototype/`.
