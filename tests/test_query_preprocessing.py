import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query_preprocessing import detect_language, normalize, preprocess, resolve_pronouns  # noqa: E402


def test_normalize_collapses_whitespace_and_punctuation():
    assert normalize("What   is Article 19???") == "What is Article 19?"


def test_detect_language_flags_devanagari():
    assert detect_language("भारत का संविधान क्या कहता है") == "non_latin_script"


def test_detect_language_defaults_to_english():
    assert detect_language("What does Article 19 protect?") == "en"


def test_resolve_pronouns_splices_prior_question():
    resolved = resolve_pronouns("What about its exceptions?", "What does Article 19 protect?")
    assert "Article 19" in resolved


def test_resolve_pronouns_leaves_standalone_question_untouched():
    resolved = resolve_pronouns("What does Article 21 protect?", "What does Article 19 protect?")
    assert resolved == "What does Article 21 protect?"


def test_preprocess_returns_all_fields():
    result = preprocess("  What about ITS scope??  ", "What does Article 19 protect?")
    assert result["raw_question"]
    assert result["normalized_question"] == "What about ITS scope?"
    assert "Article 19" in result["resolved_question"]
    assert result["language"] == "en"
