"""
query_preprocessing.py
========================
The first stage of the pipeline (Query Preprocessor in the architecture
diagram): cheap, deterministic cleanup that runs before any LLM call.

Deliberately no LLM call here — normalization, pronoun resolution, and
language detection are all things a few lines of Python do reliably and
for free, so we save the LLM budget for genuinely ambiguous work
(intent/entity extraction in the Query Analyzer stage).
"""

from __future__ import annotations

import re
import unicodedata

# Pronoun set that, when it appears early in a follow-up question, signals
# "resolve against the previous question" (e.g. "What about its exceptions?",
# "Does that apply to minors?"). Checked against the opening few words only,
# so a pronoun deep in an unrelated sentence doesn't trigger a false splice.
_EARLY_PRONOUN = re.compile(r"\b(it|its|this|that|these|those|they|them)\b", re.IGNORECASE)
_EARLY_WORDS = 4

# A crude but effective signal for non-Latin scripts commonly used for
# Indian languages, so we can flag (not translate) a non-English question.
_NON_LATIN_RANGES = [
    (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, ...)
    (0x0980, 0x09FF),  # Bengali
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]


def normalize(text: str) -> str:
    """Unicode-normalize, collapse whitespace, strip stray punctuation clutter."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([?!.]){2,}", r"\1", text)  # "???" -> "?"
    return text


def detect_language(text: str) -> str:
    """
    Very lightweight heuristic: 'en' unless a meaningful fraction of
    characters fall in a non-Latin Indian-script Unicode range, in which
    case we return that script's ISO-ish tag. This is a detector, not a
    translator — see the Limitations panel in the UI.
    """
    if not text.strip():
        return "unknown"
    script_hits = 0
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _NON_LATIN_RANGES):
            script_hits += 1
    ratio = script_hits / max(len(text), 1)
    return "non_latin_script" if ratio > 0.25 else "en"


def resolve_pronouns(text: str, prior_question: str | None) -> str:
    """
    If the question's opening words contain a bare pronoun ("it", "its",
    "that", ...) and we have a prior question in this session, splice in
    the prior question's text so downstream retrieval has something
    concrete to search for. Checking only the opening words (rather than
    anywhere in the sentence) keeps this conservative — it won't fire on,
    say, "Does the state have the power to restrict it in the interest of
    public order?" where "it" is late and the sentence is self-contained
    enough to search on its own.
    """
    if not prior_question:
        return text
    opening = " ".join(text.split()[:_EARLY_WORDS])
    if not _EARLY_PRONOUN.search(opening):
        return text
    return f"Regarding \"{prior_question}\" — {text}"


def preprocess(raw_question: str, prior_question: str | None = None) -> dict:
    """Run the full preprocessing stage and return the fields the pipeline needs."""
    normalized = normalize(raw_question)
    language = detect_language(normalized)
    resolved = resolve_pronouns(normalized, prior_question)
    return {
        "raw_question": raw_question,
        "normalized_question": normalized,
        "resolved_question": resolved,
        "language": language,
    }
