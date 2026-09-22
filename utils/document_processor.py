"""
utils/document_processor.py
============================
COMPONENT: Document Ingestion + Text Chunking
---------------------------------------------
This module handles:
  1. Document Ingestion  — reading raw bytes from PDF or TXT uploads
  2. Text Chunking       — splitting long text into smaller, overlapping chunks
                           so each chunk fits within an embedding model's context

A "chunk" is the atomic unit that gets embedded and stored in the vector store.
"""

import io
import re
from dataclasses import dataclass, field
from typing import List

import pypdf


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """Represents a single piece of text extracted from a document."""
    text: str                          # The actual text content
    source: str                        # Filename this chunk came from
    page_number: int = 0               # 1-based page number (0 = not applicable)
    chunk_index: int = 0               # Sequential index within the document
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> List[dict]:
    """
    Extract text page-by-page from a PDF file.

    Returns a list of dicts with keys 'text' and 'page_number'.
    """
    pages = []
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = _clean_text(raw)
        if cleaned.strip():
            pages.append({"text": cleaned, "page_number": page_num})
    return pages


def extract_text_from_txt(file_bytes: bytes) -> List[dict]:
    """
    Decode a plain-text file and return it as a single 'page'.
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="replace")
    cleaned = _clean_text(text)
    return [{"text": cleaned, "page_number": 1}]


def _clean_text(text: str) -> str:
    """
    Basic text cleaning:
      - Collapse multiple blank lines
      - Strip leading/trailing whitespace
    """
    text = re.sub(r"\n{3,}", "\n\n", text)   # max two consecutive newlines
    text = re.sub(r"[ \t]+", " ", text)       # collapse horizontal whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Split *text* into overlapping chunks of approximately *chunk_size* words.

    TEXT CHUNKING strategy
    ----------------------
    We use a simple word-count window with *chunk_overlap* words repeated at
    the start of each next chunk.  This preserves cross-boundary context so
    that an answer that spans two chunks is still retrievable.
    """
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - chunk_overlap   # overlap: step back before next chunk
    return chunks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_document(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[DocumentChunk]:
    """
    Full document ingestion pipeline for a single file:
      1. Detect file type
      2. Extract text (page-aware for PDF)
      3. Chunk each page's text
      4. Return a flat list of DocumentChunk objects

    Raises ValueError for unsupported file types or empty documents.
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        pages = extract_text_from_pdf(file_bytes)
    elif ext == "txt":
        pages = extract_text_from_txt(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type '.{ext}'. Please upload a PDF or TXT file."
        )

    if not pages:
        raise ValueError(f"No text could be extracted from '{filename}'.")

    chunks: List[DocumentChunk] = []
    chunk_index = 0
    for page_info in pages:
        page_chunks = chunk_text(
            page_info["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for chunk_text_str in page_chunks:
            if chunk_text_str.strip():
                chunks.append(
                    DocumentChunk(
                        text=chunk_text_str,
                        source=filename,
                        page_number=page_info["page_number"],
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

    if not chunks:
        raise ValueError(f"Document '{filename}' appears to be empty after processing.")

    return chunks
