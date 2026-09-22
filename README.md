# 📚 RAG Document Intelligence Assistant

> A beginner-friendly, fully functional Streamlit application that demonstrates **Retrieval Augmented Generation (RAG)** — ask questions about your own documents and get grounded, cited answers powered by Google Gemini.

---

## 📖 What is RAG?

**Retrieval Augmented Generation (RAG)** is a technique that makes Large Language Models (LLMs) more accurate and trustworthy by supplying them with relevant information retrieved from your own documents at query time.

Instead of relying solely on the knowledge baked into the LLM during training (which can be outdated or fabricated), RAG:

1. Stores your documents in a searchable index.
2. When you ask a question, finds the most relevant passages.
3. Gives those passages to the LLM alongside the question.
4. The LLM answers **only from what it was given**, reducing hallucinations.

---

## 🏗️ RAG Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  INDEXING PHASE (done once, at upload time)                  │
│                                                              │
│  Documents → Text Extraction → Chunking →                    │
│  Embedding Model → Vector Store (FAISS)                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  QUERY PHASE (done on every question)                        │
│                                                              │
│  User Question → Embed Question →                            │
│  Similarity Search (FAISS) → Top-K Chunks →                 │
│  Augmented Prompt → Gemini LLM → Grounded Answer + Sources   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🧩 How the Application Works

| # | Step | Where implemented |
|---|------|-------------------|
| 1 | **Document Ingestion** | `utils/document_processor.py` → `process_document()` |
| 2 | **Text Chunking** | `utils/document_processor.py` → `chunk_text()` |
| 3 | **Embeddings** | `utils/embeddings.py` → `embed_texts()` / `embed_query()` |
| 4 | **Vector Database** | `utils/vector_store.py` → `FAISSVectorStore` |
| 5 | **Similarity Search** | `utils/vector_store.py` → `FAISSVectorStore.similarity_search()` |
| 6 | **Retrieval** | `utils/rag_pipeline.py` → `RAGPipeline.query()` step 2 |
| 7 | **Prompt Augmentation** | `utils/rag_pipeline.py` → `_PROMPT_TEMPLATE` + step 3 |
| 8 | **LLM Generation** | `utils/rag_pipeline.py` → Gemini API call, step 4 |
| 9 | **Grounded Response** | `_SYSTEM_PROMPT` instructs the model to use only the context |
| 10 | **Source Citation** | `app.py` → Retrieved Sources expander in the answer section |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Programming language |
| Streamlit | Web interface |
| PyPDF2 | PDF text extraction |
| sentence-transformers | Local embedding model (`all-MiniLM-L6-v2`) |
| FAISS (faiss-cpu) | Vector similarity search |
| Google Generative AI SDK | Gemini LLM API client |
| python-dotenv | Load `.env` files locally |
| pandas | Document status table |
| numpy | Array operations |

---

## 📁 Project Structure

```
project/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for environment variables
├── README.md                     # This file
└── utils/
    ├── __init__.py
    ├── document_processor.py     # Ingestion + chunking
    ├── embeddings.py             # Embedding model wrapper
    ├── vector_store.py           # FAISS vector store
    └── rag_pipeline.py           # Full RAG orchestration
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone or download the project
git clone https://github.com/NarendraEluri123/RAG_API_Key.git
cd RAG_API_Key

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` embedding model (~80 MB). It is cached automatically for subsequent runs.

---

## 🔑 How to Configure the Gemini API Key

### Option A — Enter directly in the app (recommended for students ✅)

No file editing needed! Just:

1. Run `streamlit run app.py`
2. In the **sidebar**, find the **🔑 Gemini API Key** section.
3. Paste your key into the text box and click **✅ Save Key**.
4. The app is immediately ready — upload a document and start asking questions.

Get a free API key at: <https://aistudio.google.com/app/apikey>

### Option B — Local `.env` file

```bash
# Copy the template
cp .env.example .env        # Windows: copy .env.example .env

# Edit .env and replace the placeholder with your real key
# GEMINI_API_KEY=AIza...your_key...
```

### Option C — Streamlit Community Cloud

1. Open your app on Streamlit Cloud.
2. Go to **App settings → Secrets**.
3. Paste:

```toml
GEMINI_API_KEY = "AIza...your_key..."
```

> ⚠️ **Never commit your real API key to git.** The `.env` file is already listed in `.gitignore`.

---

## ▶️ How to Run Locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🚀 How to Deploy on Streamlit Community Cloud

1. Push your code to a public or private GitHub repository (**without** the `.env` file).
2. Go to <https://share.streamlit.io> and click **New app**.
3. Select your repository and set the main file to `app.py`.
4. Under **Advanced settings → Secrets**, add your `GEMINI_API_KEY`.
5. Click **Deploy**.

---

## 💡 Example Questions

After uploading a document, try questions like:

- *"What is the main topic of this document?"*
- *"Summarise the key findings in three bullet points."*
- *"What does the author recommend regarding X?"*
- *"Are there any statistics or figures mentioned?"*
- *"What are the limitations described?"*

---

## ⚠️ Known Limitations

- **PDF quality:** Scanned/image-based PDFs will not extract text (OCR not included).
- **Large files:** Very large PDFs may be slow to process on the first upload.
- **Embedding model:** `all-MiniLM-L6-v2` is optimised for English text.
- **Context window:** Only the top-K retrieved chunks are sent to Gemini; information in non-retrieved chunks will not be used.
- **In-memory index:** The FAISS index is stored in Streamlit session state; it resets on page refresh or server restart.
- **Rate limits:** Free Gemini API tier has request-per-minute limits.

---

## 🔮 Possible Future Enhancements

- [ ] Support for DOCX, CSV, and Markdown files
- [ ] OCR support for scanned PDFs (e.g., Tesseract)
- [ ] Persistent vector store (save/load index to disk)
- [ ] Streaming Gemini responses for faster perceived latency
- [ ] Multi-document comparison queries
- [ ] Re-ranking retrieved chunks with a cross-encoder
- [ ] Conversation-aware retrieval (use chat history to refine queries)
- [ ] LangChain integration for extended chains
- [ ] Authentication for multi-user deployments

---

## 📜 License

This project is released for educational purposes. Feel free to modify and extend it.
