"""Smart text chunker for splitting large documents into manageable pieces.

Supports multiple strategies:
  - Fixed-size character chunks with overlap
  - Sentence-boundary splitting
  - Paragraph-boundary splitting
"""

from __future__ import annotations

import re
from typing import List

from ..rag.utils_logger import get_logger

logger = get_logger(__name__)

# Defaults
DEFAULT_CHUNK_SIZE = 500       # characters per chunk
DEFAULT_OVERLAP = 50           # overlap between consecutive chunks


def chunk_by_characters(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Split text into fixed-size character chunks with overlap."""
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> List[str]:
    """Split text into chunks at sentence boundaries."""
    if not text:
        return []
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chunk_size:
            current = f"{current} {sentence}".strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def chunk_by_paragraphs(
    text: str,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE * 3,
) -> List[str]:
    """Split text by paragraph boundaries, merging short paragraphs."""
    if not text:
        return []
    paragraphs = re.split(r'\n\s*\n', text)
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= max_chunk_size:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(current)
            # If single paragraph is too long, split by sentences
            if len(para) > max_chunk_size:
                sub = chunk_by_sentences(para, max_chunk_size=max_chunk_size)
                chunks.extend(sub)
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def smart_chunk(
    text: str,
    strategy: str = "sentence",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Chunk text using the chosen strategy.

    Args:
        text: The full document text.
        strategy: 'character', 'sentence', or 'paragraph'.
        chunk_size: Maximum chunk size (meaning depends on strategy).
        overlap: Character overlap (only for character strategy).
    """
    if not text or not text.strip():
        return []

    if strategy == "character":
        return chunk_by_characters(text, chunk_size, overlap)
    elif strategy == "paragraph":
        return chunk_by_paragraphs(text, max_chunk_size=chunk_size * 3)
    else:  # default: sentence
        return chunk_by_sentences(text, max_chunk_size=chunk_size)
