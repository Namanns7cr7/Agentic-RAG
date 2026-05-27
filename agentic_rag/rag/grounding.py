"""Deterministic grounding verification utilities."""

from __future__ import annotations

import re
from typing import Dict, List, Set


def _tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2}


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def verify_grounding(
    answer: str,
    contexts: List[str],
    *,
    min_sentence_overlap: float = 0.2,
    max_unsupported_ratio: float = 0.4,
) -> Dict[str, object]:
    """Verify that answer sentences are supported by retrieved contexts.

    Uses simple lexical overlap as a deterministic safety check.
    """

    if not answer.strip():
        return {
            "is_grounded": False,
            "unsupported_claims": ["Empty answer"],
            "support_score": 0.0,
            "explanation": "Answer is empty.",
        }

    if not contexts:
        return {
            "is_grounded": False,
            "unsupported_claims": ["No retrieved context"],
            "support_score": 0.0,
            "explanation": "No retrieved context was provided.",
        }

    context_text = " ".join(contexts)
    context_tokens = _tokenize(context_text)
    if not context_tokens:
        return {
            "is_grounded": False,
            "unsupported_claims": ["Empty context"],
            "support_score": 0.0,
            "explanation": "Retrieved context contained no usable tokens.",
        }

    sentences = _split_sentences(answer)
    if not sentences:
        return {
            "is_grounded": False,
            "unsupported_claims": ["No sentences"],
            "support_score": 0.0,
            "explanation": "Answer had no detectable sentences.",
        }

    unsupported: List[str] = []
    supported_count = 0
    for sent in sentences:
        tokens = _tokenize(sent)
        if not tokens:
            continue
        overlap = len(tokens & context_tokens) / max(len(tokens), 1)
        if overlap >= min_sentence_overlap:
            supported_count += 1
        else:
            unsupported.append(sent)

    support_score = supported_count / max(len(sentences), 1)
    is_grounded = support_score >= (1.0 - max_unsupported_ratio)
    explanation = (
        f"Supported {supported_count}/{len(sentences)} sentences."
    )

    return {
        "is_grounded": is_grounded,
        "unsupported_claims": unsupported,
        "support_score": float(round(support_score, 3)),
        "explanation": explanation,
    }
