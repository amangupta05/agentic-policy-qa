from __future__ import annotations

from typing import Dict, List


def _format_context(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        src = c.get("source", "unknown")
        score = round(float(c.get("score", 0.0)), 4)
        text = (c.get("text") or "").strip()
        lines.append(f"[{i}] source={src} score={score}\n{text}")
    return "\n\n".join(lines)


def build_messages_with_context(
    user_query: str,
    history: List[Dict[str, str]],
    chunks: List[Dict[str, Any]],
    system_preamble: str,
):
    # Number and normalize context
    lines = []
    for i, c in enumerate(chunks, start=1):
        src = str(c.get("source", "")).strip()
        txt = str(c.get("text", "")).strip()
        lines.append(f"[{i}] {txt}  (source: {src})")
    ctx_block = "\n".join(lines)

    system = {
        "role": "system",
        "content": f"{system_preamble}\n\nCONTEXT:\n{ctx_block if lines else ''}\n\n"
                   "Format:\n"
                   "- If enough info: single sentence answer ending with citations, e.g., \"X [1]\".\n"
                   "- If not enough info: Not found in docs.",
    }

    user = {"role": "user", "content": user_query.strip()}
    msgs = [system] + history + [user]
    return msgs
