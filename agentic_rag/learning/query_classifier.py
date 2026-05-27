"""Rule-based query classifier for DocuMentor AI.

Classifies a user query into one of the supported learning modes.
Rule-based first for reliability; LLM fallback is optional and wired in
but not called by default to keep latency low.

Supported modes:
    explain | code_example | build_with_me | challenge | quiz
    cheat_sheet | debug | interview | general_qa
"""

from __future__ import annotations

import re
from typing import Tuple

from ..rag.utils_logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword rule tables (ordered: more specific patterns first)
# ---------------------------------------------------------------------------

_RULES: list[Tuple[str, list[str]]] = [
    ("debug", [
        r"\b\w+error\b", r"\btraceback\b", r"\bexception\b", r"\bfix\b",
        r"\bbug\b", r"\bdebug\b", r"\bwhy (is|does|am|are)\b",
        r"\bnot working\b", r"\bfailed\b", r"\bcrash\b",
        r"[A-Z][a-z]+Error:",
    ]),
    ("cheat_sheet", [
        r"\bcheat ?sheet\b", r"\bquick ref(erence)?\b", r"\breference card\b",
        r"\bsummary card\b", r"\ball commands\b",
    ]),
    ("interview", [
        r"\binterview\b", r"\binterview (question|prep)\b",
        r"\bhiring\b", r"\btechnical (question|round)\b",
    ]),
    ("quiz", [
        r"\bquiz\b", r"\btest me\b", r"\bcheck (my )?understanding\b",
        r"\bask me\b", r"\bquestion(s)? (for|about|on)\b",
        r"\bflashcard\b",
    ]),
    ("challenge", [
        r"\bchallenge\b", r"\bpractice (task|problem|exercise)\b",
        r"\bgive me (a |an )?task\b", r"\bcoding (exercise|problem)\b",
        r"\bexercise\b",
    ]),
    ("build_with_me", [
        r"\bbuild\b.*\bme\b", r"\bstep.?by.?step\b", r"\bmini.?project\b",
        r"\bwalk.?through\b", r"\bguide me\b", r"\bfrom scratch\b",
        r"\btutorial\b",
    ]),
    ("code_example", [
        r"\bcode (example|sample|snippet)\b", r"\bgive (me )?(a |an )?example\b",
        r"\bshow (me )?(how|code|example)\b", r"\brunnable\b",
        r"\bhow (do|to) (I |you )?(implement|use|write|create)\b",
        r"\bsample code\b",
    ]),
    ("explain", [
        r"\bexplain\b", r"\bwhat is\b", r"\bwhat are\b", r"\bhow does\b",
        r"\bdefine\b", r"\bmeaning of\b", r"\bunderstand\b",
        r"\bwhy is\b", r"\btell me about\b",
    ]),
]

_DEFAULT_MODE = "general_qa"


def classify(query: str) -> str:
    """Return the learning mode for *query* using rule-based matching.

    Falls back to ``general_qa`` if no rule matches.
    """
    q = query.lower()
    for mode, patterns in _RULES:
        for pattern in patterns:
            if re.search(pattern, q):
                logger.debug("Query classified as '%s' via pattern '%s'", mode, pattern)
                return mode
    logger.debug("Query fell through to default mode '%s'", _DEFAULT_MODE)
    return _DEFAULT_MODE
