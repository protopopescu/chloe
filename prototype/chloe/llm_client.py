"""
Minimal client for an OpenAI-compatible chat-completions endpoint (vLLM).

This is CHLOE's seam for the "LLM as linguistic layer, symbolic core as
authority" idea from the original design and the README's file-layout table:
nlu.parse() turns human text into an Utterance for the symbolic engine, and
this module is the mirror-image piece for the *output* side -- turning the
engine's factual reply into something that reads like natural conversation.

Deliberately stdlib-only (urllib). Configuration is via environment
variables, e.g.:

    export VLLM_BASE_URL=http://<your-llm-host>:8000/v1
    export VLLM_API_KEY=...          # only if the server requires one
    export VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct   # optional, has a default

This only ever speaks HTTP to whatever OpenAI-compatible server
VLLM_BASE_URL points at; with the variable unset it makes no requests.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "30"))


class LLMUnavailable(Exception):
    """Raised when the vLLM server isn't configured or couldn't be reached.
    Callers should catch this and fall back to the symbolic engine's own
    reply rather than let a chat request fail outright."""


def is_configured() -> bool:
    return bool(os.getenv("VLLM_BASE_URL"))


def chat(messages: list, *, temperature: float = 0.7, max_tokens: int = 400,
         model: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Send a chat-completions request, return the reply text.

    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Raises LLMUnavailable on any configuration or network/HTTP problem.
    """
    base_url = os.getenv("VLLM_BASE_URL")
    if not base_url:
        raise LLMUnavailable("VLLM_BASE_URL is not set")

    api_key = os.getenv("VLLM_API_KEY")
    model = model or os.getenv("VLLM_MODEL", DEFAULT_MODEL)

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise LLMUnavailable(f"vLLM server returned HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMUnavailable(f"could not reach vLLM server at {base_url}: {e.reason}") from e
    except TimeoutError as e:
        raise LLMUnavailable(f"vLLM server at {base_url} timed out") from e

    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMUnavailable(f"unexpected response shape from vLLM server: {body}") from e
