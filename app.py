"""
app.py
=======
RAG Document Intelligence Assistant
-------------------------------------
Main Streamlit application.  Run with:  streamlit run app.py

UI Structure
------------
Sidebar : App info | Document upload | API key instructions | Top-K | Reset
Main    : Title | RAG explanation | Document status | Q&A | Chat history
"""

import os
import streamlit as st
from dotenv import load_dotenv

from utils.document_processor import process_document
from utils.embeddings import embed_texts
from utils.vector_store import FAISSVectorStore
from utils.rag_pipeline import RAGPipeline

# ── Load .env (works locally; Streamlit Cloud uses st.secrets) ─────────────
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RAG Document Intelligence Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════
def _init_session():
    """Ensure all required session-state keys exist."""
    defaults = {
        "vector_store": FAISSVectorStore(),
        "processed_files": [],       # list of dicts with file metadata
        "processed_file_ids": set(), # set of file_id strings already processed
        "chat_history": [],          # list of RAGResult objects
        "pipeline": None,            # RAGPipeline instance (created after API key check)
        "processing_error": None,
        "user_api_key": "",          # key entered by the student in the UI
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_api_key() -> str:
    """
    Resolve the Gemini API key with the following priority:
      1. Key entered by the student in the sidebar UI  (session state)
      2. Streamlit Cloud secrets
      3. Local .env / OS environment variable
    """
    # 1. Student-supplied key from the sidebar input
    if st.session_state.get("user_api_key", "").strip():
        return st.session_state["user_api_key"].strip()
    # 2. Streamlit Cloud / Streamlit secrets
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    # 3. Local .env / OS environment
    return os.getenv("GEMINI_API_KEY", "")


def _process_uploaded_files(uploaded_files, chunk_size: int, chunk_overlap: int):
    """
    Process newly uploaded files, embed their chunks, and add to the vector store.

    Uses file_id (Streamlit's unique ID per upload) to detect truly new files.
    File bytes are read immediately — they are only available on the rerun that
    first delivers the upload and become empty on subsequent reruns.
    """
    # Identify files we have not yet processed by their unique upload ID
    new_files = [
        f for f in uploaded_files
        if f.file_id not in st.session_state.processed_file_ids
    ]

    if not new_files:
        return

    # Clear any stale error from a previous failed attempt before we start
    st.session_state.processing_error = None

    progress_placeholder = st.empty()
    total = len(new_files)
    success_count = 0

    for i, uploaded_file in enumerate(new_files):
        progress_placeholder.info(
            f"⏳ Processing {uploaded_file.name} ({i + 1}/{total})…"
        )
        try:
            # Read bytes immediately — buffer is only valid on this rerun
            file_bytes = uploaded_file.read()
            if not file_bytes:
                raise ValueError(
                    "File appears to be empty or could not be read. "
                    "Please remove it and re-upload."
                )
            chunks = process_document(
                file_bytes,
                uploaded_file.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            embeddings = embed_texts([c.text for c in chunks])
            st.session_state.vector_store.add_chunks(chunks, embeddings)
            # Mark as processed ONLY on success so failed files can be retried
            st.session_state.processed_file_ids.add(uploaded_file.file_id)
            st.session_state.processed_files.append(
                {
                    "filename": uploaded_file.name,
                    "num_chunks": len(chunks),
                    "file_size_kb": round(len(file_bytes) / 1024, 1),
                }
            )
            success_count += 1
        except Exception as exc:
            st.session_state.processing_error = (
                f"Error processing '{uploaded_file.name}': {exc}"
            )

    if success_count:
        progress_placeholder.success(
            f"✅ Processed {success_count}/{total} file(s). Ready to answer questions!"
        )
    else:
        progress_placeholder.empty()


def _get_pipeline() -> RAGPipeline | None:
    """Return (and cache) a RAGPipeline, or None if the API key is missing."""
    api_key = _get_api_key()
    if not api_key:
        return None
    # Always re-create so model name changes in code take effect immediately
    try:
        st.session_state.pipeline = RAGPipeline(
            vector_store=st.session_state.vector_store,
            api_key=api_key,
        )
    except ValueError as exc:
        st.error(str(exc))
        return None
    return st.session_state.pipeline


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/120px-ChatGPT_logo.svg.png",
        width=50,
    )
    st.title("📚 RAG Assistant")
    st.caption("Document Intelligence powered by Retrieval Augmented Generation")
    st.divider()

    # ── Document Upload ───────────────────────────────────────────────
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more documents. The assistant will answer questions based on their content.",
    )

    # ── Chunking settings (advanced, collapsed by default) ───────────
    with st.expander("⚙️ Advanced Settings", expanded=False):
        chunk_size = st.slider(
            "Chunk size (words)", min_value=100, max_value=1000,
            value=500, step=50,
            help="Number of words per document chunk.",
        )
        chunk_overlap = st.slider(
            "Chunk overlap (words)", min_value=0, max_value=200,
            value=50, step=10,
            help="Words shared between consecutive chunks.",
        )
        top_k = st.slider(
            "Top-K retrieval", min_value=1, max_value=10,
            value=5, step=1,
            help="Number of document chunks to retrieve per question.",
        )

    # Process files whenever new ones appear
    if uploaded_files:
        _process_uploaded_files(uploaded_files, chunk_size, chunk_overlap)

    # ── API Key input ─────────────────────────────────────────────────
    st.divider()
    st.subheader("🔑 Gemini API Key")

    # Show the text input pre-filled with whatever is already in session state
    entered_key = st.text_input(
        "Enter your Gemini API key",
        value=st.session_state.get("user_api_key", ""),
        type="password",
        placeholder="AIza…",
        help=(
            "Paste your personal Gemini API key here. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        ),
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Save Key", use_container_width=True, type="primary"):
            if entered_key.strip():
                st.session_state.user_api_key = entered_key.strip()
                # Invalidate the cached pipeline so it is rebuilt with the new key
                st.session_state.pipeline = None
                st.success("Key saved!", icon="🔑")
            else:
                st.warning("Please enter a key first.", icon="⚠️")
    with col2:
        if st.button("🗑️ Clear Key", use_container_width=True, type="secondary"):
            st.session_state.user_api_key = ""
            st.session_state.pipeline = None
            st.rerun()

    # Status indicator
    resolved_key = _get_api_key()
    if resolved_key:
        source = (
            "entered above"
            if st.session_state.get("user_api_key", "").strip()
            else "environment / secrets"
        )
        st.success(f"✅ API key active ({source})", icon="✅")
    else:
        st.warning("⚠️ No API key found — enter one above.", icon="⚠️")
        st.caption(
            "Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)"
        )

    # ── Reset button ──────────────────────────────────────────────────
    st.divider()
    if st.button("🗑️ Clear Everything", use_container_width=True, type="secondary"):
        for key in ["vector_store", "processed_files", "processed_file_ids", "chat_history", "pipeline", "processing_error", "user_api_key"]:
            del st.session_state[key]
        _init_session()
        st.rerun()

    if st.button("💬 Clear Chat Only", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Built for educational purposes · RAG Demo")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════

st.title("📚 RAG Document Intelligence Assistant")
st.markdown(
    "_Ask questions about your uploaded documents. "
    "Answers are grounded in the document content, not AI imagination._"
)

# ── What is RAG? (collapsible) ────────────────────────────────────────────
with st.expander("ℹ️ What is Retrieval Augmented Generation (RAG)?", expanded=False):
    st.markdown(
        """
**RAG** combines two AI capabilities:

| Step | What happens |
|------|--------------|
| 📄 **Document ingestion** | Your documents are split into small text chunks |
| 🔢 **Embeddings** | Each chunk is converted to a numerical vector that captures its meaning |
| 🗄️ **Vector store** | Vectors are stored in a searchable index (FAISS) |
| 🔍 **Retrieval** | Your question is embedded and the most similar chunks are found |
| 🧩 **Prompt augmentation** | Retrieved chunks are added to the prompt sent to the LLM |
| 🤖 **LLM generation** | Gemini reads both your question and the context to produce a grounded answer |
| 📎 **Source citation** | The sources used to generate the answer are shown below it |

**Why RAG?**  
Standard LLMs can hallucinate.  RAG anchors the model to real documents, 
making answers verifiable and trustworthy.
        """
    )

# ── RAG Pipeline Visualisation ────────────────────────────────────────────
with st.expander("🔄 RAG Pipeline Flow", expanded=False):
    st.markdown(
        """
```
USER QUESTION
      ↓
QUESTION EMBEDDING  (sentence-transformers/all-MiniLM-L6-v2)
      ↓
VECTOR SIMILARITY SEARCH  (FAISS IndexFlatL2)
      ↓
RELEVANT DOCUMENT CHUNKS  (Top-K retrieved)
      ↓
AUGMENTED PROMPT  (question + context injected)
      ↓
GEMINI LLM  (gemini-1.5-flash)
      ↓
GROUNDED ANSWER + SOURCES
```
        """
    )

# ── Processing errors ─────────────────────────────────────────────────────
if st.session_state.processing_error:
    st.error(st.session_state.processing_error)

# ── Document status ───────────────────────────────────────────────────────
if st.session_state.processed_files:
    st.subheader("📋 Processed Documents")
    import pandas as pd
    df = pd.DataFrame(st.session_state.processed_files)
    df.columns = ["Filename", "Chunks", "Size (KB)"]
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        f"Total chunks indexed: **{st.session_state.vector_store.total_chunks}**"
    )
else:
    st.info(
        "👆 Upload one or more PDF or TXT documents using the sidebar to get started.",
        icon="📂",
    )

st.divider()

# ── Question input ────────────────────────────────────────────────────────
st.subheader("💬 Ask a Question")

with st.form(key="question_form", clear_on_submit=True):
    question = st.text_area(
        "Your question",
        placeholder="e.g. What are the main findings of the document?",
        height=80,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("🔍 Ask", use_container_width=True, type="primary")

if submitted and question.strip():
    # Guard: documents must be uploaded
    if not st.session_state.vector_store.is_ready:
        st.warning("Please upload and process at least one document first.", icon="⚠️")
    else:
        pipeline = _get_pipeline()
        if pipeline is None:
            st.error(
                "Gemini API key not found. "
                "Add GEMINI_API_KEY to your .env file or Streamlit secrets.",
                icon="🔑",
            )
        else:
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    result = pipeline.query(question.strip(), top_k=top_k)
                    # Prepend so the newest answer appears first
                    st.session_state.chat_history.insert(0, result)
                except Exception as exc:
                    st.error(f"Error: {exc}", icon="❌")

# ── Chat history ──────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.subheader("📝 Answers")
    for idx, result in enumerate(st.session_state.chat_history):
        with st.container():
            # Question
            st.markdown(f"**❓ Question {len(st.session_state.chat_history) - idx}:** {result.question}")

            # Answer
            st.markdown("**💡 Answer:**")
            st.info(result.answer)

            # Sources
            with st.expander(
                f"📎 Retrieved Sources ({len(result.sources)} chunks)", expanded=False
            ):
                for rank, (chunk, distance) in enumerate(result.sources, start=1):
                    st.markdown(
                        f"**Source {rank}** | 📄 `{chunk.source}` "
                        f"| Page {chunk.page_number} | Chunk #{chunk.chunk_index} "
                        f"| Distance: `{distance:.4f}`"
                    )
                    st.text_area(
                        label=f"chunk_{idx}_{rank}",
                        value=chunk.text,
                        height=120,
                        disabled=True,
                        label_visibility="collapsed",
                    )

            # Augmented prompt (for educational transparency)
            with st.expander("🔬 View Augmented Prompt (educational)", expanded=False):
                st.code(result.augmented_prompt, language="text")

            st.divider()
