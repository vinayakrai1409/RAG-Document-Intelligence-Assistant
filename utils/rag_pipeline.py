"""
utils/rag_pipeline.py
======================
COMPONENTS: Prompt Augmentation + LLM Generation + Grounded Response + Source Citation
---------------------------------------------------------------------------------------
This module is the "brain" of the RAG system.  It:

  Prompt Augmentation — builds a prompt that injects retrieved document
                        context alongside the user's question
  LLM Generation      — calls the Gemini API and returns the generated text
  Grounded Response   — instructs the model to answer ONLY from the context,
                        avoiding hallucination
  Source Citation     — passes source metadata back to the UI so users can
                        see which chunks were used

Architecture
------------
RAGPipeline is the single public class.  It holds references to:
  - FAISSVectorStore  (already populated)
  - Gemini API key

Call pipeline.query(question, top_k) to run the full RAG cycle.
"""

import os
from typing import List, Tuple
from dataclasses import dataclass

from google import genai
from google.genai import types

from utils.vector_store import FAISSVectorStore
from utils.embeddings import embed_query
from utils.document_processor import DocumentChunk


# ---------------------------------------------------------------------------
# Result data model
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """Everything the UI needs to render a complete answer."""
    question: str
    answer: str
    sources: List[Tuple[DocumentChunk, float]]   # (chunk, distance)
    augmented_prompt: str                         # shown for educational transparency


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a helpful document assistant.
Answer the user's question using ONLY the information provided in the context below.
Do NOT use any outside knowledge or make up information.
If the context does not contain enough information to answer the question, respond with exactly:
"I could not find sufficient information in the uploaded documents to answer this question."

Be concise and accurate. Cite facts directly from the context.
"""

_PROMPT_TEMPLATE = """{system}

CONTEXT (retrieved from uploaded documents):
---
{context}
---

QUESTION: {question}

ANSWER:"""


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Orchestrates the full RAG cycle:
      question → embed → retrieve → augment prompt → LLM → answer
    """

    def __init__(self, vector_store: FAISSVectorStore, api_key: str):
        """
        Parameters
        ----------
        vector_store : FAISSVectorStore — pre-populated with document embeddings
        api_key      : Gemini API key (never hard-coded; passed in at runtime)
        """
        if not api_key or not api_key.strip():
            raise ValueError(
                "Gemini API key is missing. "
                "Set GEMINI_API_KEY in your .env file or Streamlit secrets."
            )
        self._store = vector_store
        self._model_name = "gemini-3.6-flash"
        # Configure the Gemini client — key is read at runtime, never stored in code
        self._client = genai.Client(api_key=api_key.strip())

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """
        Run the complete RAG pipeline for a single question.

        Steps
        -----
        1. Embed the question                    ← Embeddings
        2. Retrieve top-k chunks from FAISS      ← Similarity Search / Retrieval
        3. Build the augmented prompt            ← Prompt Augmentation
        4. Call Gemini                           ← LLM Generation
        5. Return answer + sources               ← Grounded Response + Source Citation

        Parameters
        ----------
        question : str   — the user's natural-language question
        top_k    : int   — how many document chunks to retrieve

        Returns
        -------
        RAGResult
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not self._store.is_ready:
            raise RuntimeError(
                "No documents have been processed yet. "
                "Please upload and process documents before asking questions."
            )

        # ── Step 1: Embed the question ────────────────────────────────
        query_vector = embed_query(question)

        # ── Step 2: Retrieve relevant chunks ─────────────────────────
        sources = self._store.similarity_search(query_vector, top_k=top_k)

        # ── Step 3: Build augmented prompt ────────────────────────────
        context_parts = []
        for i, (chunk, distance) in enumerate(sources, start=1):
            context_parts.append(
                f"[Source {i} | File: {chunk.source} | Page: {chunk.page_number} | "
                f"Chunk: {chunk.chunk_index}]\n{chunk.text}"
            )
        context_str = "\n\n".join(context_parts)

        augmented_prompt = _PROMPT_TEMPLATE.format(
            system=_SYSTEM_PROMPT,
            context=context_str,
            question=question.strip(),
        )

        # ── Step 4: Call Gemini ───────────────────────────────────────
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=augmented_prompt,
                config=types.GenerateContentConfig(
                    temperature=1.0,      # required when thinking_config is set
                    max_output_tokens=8192,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0  # disable thinking for RAG — faster, cheaper
                    ),
                ),
            )
            # gemini-3.6-flash may split output across multiple parts; join them all
            parts = response.candidates[0].content.parts
            answer = "".join(
                p.text for p in parts if hasattr(p, "text") and p.text
            ).strip()
            if not answer:
                answer = "I could not find sufficient information in the uploaded documents to answer this question."
        except Exception as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc

        # ── Step 5: Return grounded result with sources ───────────────
        return RAGResult(
            question=question,
            answer=answer,
            sources=sources,
            augmented_prompt=augmented_prompt,
        )
