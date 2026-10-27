"""Short notes for the items that made the cut.

The chat model only rewrites text it is given. Links, star counts, versions and
dates are never asked of it: they come straight from the source's own fields.
"""

from __future__ import annotations

import re

from .config import Topic
from .llm import Ollama, OllamaError
from .models import Item, VERDICTS

SYSTEM = (
    "You write short notes for a personal research radar. "
    "Use only the text inside the <source> tags. Do not add facts, numbers, names or links "
    "that are not in that text. If the text does not support a claim, leave the claim out. "
    "Use plain words. Do not use words like powerful, innovative, revolutionary or cutting-edge. "
    "Answer with one JSON object and nothing else."
)

USER_TEMPLATE = """Reader interest ({topic_name}): {topic_description}
Item type: {kind}

<source>
{title}
{text}
</source>

This item was picked because it looks related to the reader interest. Reply with JSON in exactly this shape:
{{"what": "<one sentence saying what this is>", "why": "<one sentence saying how it connects to the reader interest>", "verdict": "<one of: {verdicts}>"}}

Verdicts:
{definitions}
Write the sentences in {language}."""

VERDICT_HELP = {
    "install": "install = a tool the reader could try today",
    "read": "read = worth reading in full",
    "watch": "watch = interesting, worth checking again later",
    "skip": "skip = shares words with the interest but is of no use to the reader; choose it only when the source is clearly unrelated",
}


def allowed_verdicts(kind: str) -> tuple[str, ...]:
    # A paper cannot be installed.
    return tuple(v for v in VERDICTS if not (kind == "paper" and v == "install"))


def normalize_verdict(value: object, kind: str) -> str:
    text = str(value or "").strip().lower()
    allowed = allowed_verdicts(kind)
    if text in allowed:
        return text
    if text == "install" and kind == "paper":
        return "read"
    return default_verdict(kind)


def default_verdict(kind: str) -> str:
    return "read" if kind == "paper" else "watch"


def _first_sentences(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("; "))
    if stop > 80:
        return cut[: stop + 1]
    return cut.rsplit(" ", 1)[0] + "..."


def fallback_note(item: Item, topic: Topic) -> tuple[str, str, str]:
    """Used when no chat model is available. Plain extraction, no generation."""
    what = _first_sentences(item.text) or "No description given."
    why = f"Matched your topic '{topic.name}'."
    return what, why, default_verdict(item.kind)


def _clean_sentence(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def summarize(ollama: Ollama, item: Item, topic: Topic, source_text: str, language: str) -> tuple[str, str, str]:
    """Ask the chat model. Any failure falls back to plain extraction."""
    body = (source_text or item.text)[:3000]
    user = USER_TEMPLATE.format(
        topic_name=topic.name,
        topic_description=" ".join(topic.description.split()),
        kind=item.kind,
        title=item.title,
        text=body,
        verdicts=", ".join(allowed_verdicts(item.kind)),
        definitions="\n".join(VERDICT_HELP[v] for v in allowed_verdicts(item.kind)),
        language=language,
    )
    try:
        answer = ollama.chat_json(SYSTEM, user)
    except OllamaError:
        answer = None
    if not answer:
        return fallback_note(item, topic)
    what = _clean_sentence(answer.get("what"))
    why = _clean_sentence(answer.get("why"))
    if not what:
        what, _, _ = fallback_note(item, topic)
    if not why:
        why = f"Matched your topic '{topic.name}'."
    return what, why, normalize_verdict(answer.get("verdict"), item.kind)
