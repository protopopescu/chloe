"""
Scripted end-to-end walkthrough of the CHLOE prototype: multiple sources,
a contradiction, scoped facts, declared-transitive hypothesis generation,
a sleep pass, and verification of a hypothesis in later dialogue.

Run: python3 demo.py
"""

import os

from chloe.dialogue import ChloeEngine
from chloe.models import RelationProperties
from chloe.storage import KnowledgeStore

DB_PATH = "demo_chloe.db"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    store = KnowledgeStore(DB_PATH)
    store.declare_relation(RelationProperties(name="is", transitive=True))

    section("1. Dan states some facts")
    dan = ChloeEngine(store)
    print(dan.greet("Dan"))
    for line in [
        "Claire is an artist",
        "George is a doctor",
        "George is busy when it is a weekday",
        "George is available when it is the weekend",
        "Felix is a cat",
        "a cat is an animal",
    ]:
        print(f"Dan: {line}")
        print(" - " + dan.turn(line))

    section("2. Alice, Bob and Eve deny what Dan said about Claire")
    for name in ["Alice", "Bob", "Eve"]:
        person = ChloeEngine(store)
        print(person.greet(name))
        line = "Claire is not an artist"
        print(f"{name}: {line}")
        print(" - " + person.turn(line))

    section("3. Knowledge state before sleep")
    for atom in store.all_atoms():
        print(f"  #{atom.id} {atom.statement()}  [{atom.status.value}, conf={atom.confidence:.2f}, "
              f"n_provenance={len(atom.provenance)}]")

    section("4. Trust profiles before sleep")
    for p in store.all_people():
        print(f"  {p.name}: {p.trust}")

    section("5. Sleep")
    dan2 = ChloeEngine(store)
    print(dan2.greet("Dan"))
    print(dan2.turn("sleep"))

    section("6. Open questions queued after sleep")
    for q in store.pending_questions():
        print(f"  [{q['reason']}] {q['question']}")

    section("7. Knowledge state after sleep (note new HYPOTHESIS atom)")
    for atom in store.all_atoms():
        print(f"  #{atom.id} {atom.statement()}  [{atom.status.value}, conf={atom.confidence:.2f}]")

    section("8. Next session: Chloe leads with an open question; Dan verifies the hypothesis")
    dan3 = ChloeEngine(store)
    print(dan3.greet("Dan"))
    q = dan3.pending_question()
    print(" - " + q if q else "  (no question surfaced)")
    print("Dan: Felix is an animal.")
    print(" - " + dan3.turn("Felix is an animal."))

    section("9. Final knowledge state")
    for atom in store.all_atoms():
        print(f"  #{atom.id} {atom.statement()}  [{atom.status.value}, conf={atom.confidence:.2f}]")

    section("10. Ask Chloe directly")
    print("Dan: is George available when it is a weekday?")
    print(" - " + dan3.turn("is George available when it is a weekday?"))
    print("Dan: what is Felix?")
    print(" - " + dan3.turn("what is Felix?"))

    store.close()


if __name__ == "__main__":
    main()
