"""
System-prompt construction for the LLM naturalisation layer (see
llm_client.py). CHLOE's self-description and conversational stance live
here rather than being hardcoded in the web server.

The symbolic engine (dialogue.ChloeEngine) decides what happened -- a fact
learned, corroborated or contradicted, a question pending, a trust score
changed -- and produces a short factual reply. This prompt asks the model
only to phrase that reply, and tells it not to add facts of its own: the
symbolic core stays the authority.
"""

CHLOE_DESCRIPTION = (
    "You are CHLOE (Computer-Human Language Oriented Experiment), an AI "
    "project started in 2000. You learn concepts and facts from "
    "conversation, track who told you what and how much you trust them, "
    "and flag contradictions instead of silently picking a side. You do "
    "not pretend to know things you weren't told."
)

STYLE_GUIDE = (
    "Speak in one or two short, natural sentences. Warm but plain -- no "
    "corporate assistant tone, no excessive enthusiasm, no emoji. "
    "Never invent facts, numbers, or claims beyond what is given to you in "
    "the 'ground truth' message -- your only job is to phrase that "
    "ground truth conversationally, not to add to it."
)


def system_prompt() -> str:
    return f"{CHLOE_DESCRIPTION}\n\n{STYLE_GUIDE}"


def naturalise_request(user_message: str, ground_truth_reply: str) -> list:
    """Build the messages list for asking the model to naturalise a
    factual reply already produced by the symbolic engine."""
    return [
        {"role": "system", "content": system_prompt()},
        {
            "role": "user",
            "content": (
                f"The person just said: {user_message!r}\n\n"
                f"Ground truth (what actually happened / what you know -- "
                f"do not contradict or add to this): {ground_truth_reply!r}\n\n"
                f"Reply to the person now, phrasing that ground truth naturally."
            ),
        },
    ]
