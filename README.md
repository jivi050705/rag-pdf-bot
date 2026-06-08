# RAG PDF Bot 🤖📄

> Ask questions about any PDF using AI — get answers with source page references.

I built this project to explore how RAG (Retrieval-Augmented Generation) works in practice. You upload a PDF, it gets chunked and embedded locally, and then you can ask anything about it in plain English. The LLM only answers from what's actually in the document — no hallucinated facts.

---

## What it does

- Upload any text-based PDF
- Ask questions about its content in natural language
- Get answers grounded in the document, with exact page references
- Handles scanned/image-based PDFs gracefully with a clear error message
- Reset and load a completely different PDF without refreshing

---

## Tech I used

- **FastAPI** — backend API and static file serving
- **LangChain** — RAG pipeline (loading, chunking, retrieval, prompting)
- **FAISS** — vector similarity search
- **HuggingFace `all-MiniLM-L6-v2`** — sentence embeddings
- **Groq API** — LLM inference using `llama-3.3-70b-versatile`
- **Vanilla HTML/CSS/JS** — simple frontend, no framework

---

## How it works

```text
1. PDF Upload
   Text gets extracted page by page

2. Chunking
   The document is split into smaller overlapping chunks

3. Embedding
   Each chunk is converted into a vector using MiniLM

4. Storage
   All vectors are stored in FAISS for semantic search

5. User Question
   The question is embedded and matched against the most relevant chunks

6. Answer Generation
   Groq LLM answers using only the retrieved context

7. Output
   The final answer is returned with source page numbers
```

---

## Project Structure

```bash

rag-pdf-bot/
├── backend/
│ ├── static/
│ │ └── index.html
│ ├── uploaded_pdfs/
│ ├── main.py
│ ├── rag_engine.py
│ ├── requirements.txt
│ └── .env
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- A Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rag-pdf-bot.git
cd rag-pdf-bot/backend
```

### 2. Create a virtual environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run downloads the embedding model once and caches it locally.

### 4. Create `.env`

Inside `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Start the server

```bash
uvicorn main:app --reload
```

### 6. Open the app

http://127.0.0.1:8000/app


> Important: open it through the URL above, not by double-clicking `index.html`.

---

## API Routes

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/` | Health check |
| `GET` | `/app` | Frontend |
| `POST` | `/upload-pdf/?session_id=...` | Upload and process PDF |
| `POST` | `/ask/` | Ask a question |

### Example ask request

```json
{
  "session_id": "session_abc123",
  "question": "What is this document about?"
}
```

### Example response

```json
{
  "answer": "This document explains ...",
  "source_pages":
}
```

---

## Limitations

- Scanned PDFs without selectable text are not supported
- Session data is stored in memory — restarting the server clears all sessions
- No persistent storage — uploaded PDFs are not saved between server restarts
- Single user per session — not designed for concurrent multi-user production use

---

## Future Improvements

- [ ] Support for scanned PDFs using OCR (Tesseract)
- [ ] Persistent session storage with Redis or Supabase
- [ ] Multi-PDF support — query across multiple documents
- [ ] Chat history — remember previous questions in a session
- [ ] Deploy on Render / Railway

---

## 👨‍💻 Author

**Jivitesh Singh**
B.Tech CSE — Amity University Chhattisgarh
