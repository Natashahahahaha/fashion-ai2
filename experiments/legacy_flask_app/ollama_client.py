"""
Tiny wrapper around Ollama's local REST API. Standard library only.

Only reachable when Ollama is running on the SAME machine as this app.
Works during local testing; once deployed to Hugging Face Spaces, Ollama
is not reachable (it's running on your machine, not the Space's server) —
app.py handles that by falling back to a template explanation, not by
this file failing loudly.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"


def is_ollama_available(timeout: float = 1.5) -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def call_ollama_chat(model: str, prompt: str, image_path: str | None = None, timeout: float = 90.0) -> str:
    message = {"role": "user", "content": prompt}

    if image_path:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        message["images"] = [image_b64]

    payload = json.dumps({"model": model, "messages": [message], "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Ollama at {OLLAMA_URL} ({e})")
