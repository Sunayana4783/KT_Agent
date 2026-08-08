# KT Agent — Knowledge Transfer via PDF + RAG

A production-ready **Knowledge Transfer agent** built with:

| Layer | Technology |
|---|---|
| Agent orchestration | [LangGraph 1.0](https://www.langchain.com/langgraph) |
| LLM & embeddings | OpenAI (`gpt-4o-mini` + `text-embedding-3-small`) |
| Vector store | [ChromaDB](https://www.trychroma.com/) (local persistent) |
| PDF parsing | PyPDF + LangChain `PyPDFLoader` |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Deployment | [Render](https://render.com) (via `render.yaml` Blueprint) |

---

## Architecture

```
PDF files
   │
   ▼
ingest.py  ──►  ChromaDB (vectorstore/)
                     │
User question        │
   │                 ▼
   └──► agent.py (LangGraph graph)
            │
         [retrieve]
            │
         [grade_docs]  ──► irrelevant? ──► [rewrite_query] ──┐
            │                                                  │
         [generate]  ◄────────────────────────────────────────┘
            │
         [hallucination check]
            │
          answer
            │
   ┌────────┴────────┐
main.py (FastAPI)   app.py (Streamlit)
  POST /ask          chat UI
  POST /ingest       PDF uploader
```

---

## Project Structure

```
KT/
├── main.py            # FastAPI backend (API server)
├── app.py             # Streamlit frontend (chat UI)
├── agent.py           # LangGraph RAG agent
├── ingest.py          # PDF loader + ChromaDB ingestion
├── requirements.txt   # Pinned dependencies
├── render.yaml        # Render Blueprint (deploy both services)
├── .env.example       # Environment variable template
├── .gitignore
├── data/              # Put your PDF files here for bulk ingestion
├── uploads/           # PDFs uploaded via the API (auto-created)
└── vectorstore/       # ChromaDB data (auto-created, git-ignored)
```

---

## Local Development

### 1. Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 2. Clone & install

```bash
git clone https://github.com/<your-username>/kt-agent.git
cd kt-agent
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 4. Ingest PDFs (optional bulk load)

Place PDF files in the `data/` folder, then run:

```bash
python ingest.py data/
```

Or ingest a single file:

```bash
python ingest.py data/my-doc.pdf
```

### 5. Start the FastAPI backend

```bash
uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 6. Start the Streamlit UI (separate terminal)

```bash
streamlit run app.py
```

Open: http://localhost:8501

Upload PDFs via the sidebar, then ask questions in the chat.

---

## Deploying to Render

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "feat: initial KT agent"
git remote add origin https://github.com/<your-username>/kt-agent.git
git push -u origin main
```

### Step 2 — Create a new Blueprint on Render

1. Log in to [render.com](https://render.com).
2. Click **New → Blueprint**.
3. Connect your GitHub repository.
4. Render auto-detects `render.yaml` and shows two services:
   - `kt-api` (FastAPI backend)
   - `kt-ui` (Streamlit frontend)
5. Click **Apply**.

### Step 3 — Set environment variables

In the Render dashboard, for the **kt-api** service:

| Key | Value |
|---|---|
| `OPENAI_API_KEY` | `sk-...your-key...` |

> All other variables have sensible defaults. See `.env.example` for full list.

### Step 4 — Update the frontend URL

Once `kt-api` is deployed, copy its `https://kt-api.onrender.com` URL and set it as:

- The `BACKEND_URL` environment variable on the `kt-ui` service.

Or update `render.yaml` → `kt-ui` → `BACKEND_URL` value and redeploy.

### Step 5 — Trigger a redeploy

Push any change to `main` — Render auto-deploys both services.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/ingest` | Upload & ingest a PDF |
| `POST` | `/ask` | Ask the KT agent a question |

### POST /ask — example

```bash
curl -X POST https://kt-api.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the deployment process?"}'
```

Response:

```json
{
  "answer": "The deployment process involves ...",
  "sources": ["deployment-guide.pdf"],
  "question_used": "What is the deployment process?"
}
```

### POST /ingest — example

```bash
curl -X POST https://kt-api.onrender.com/ingest \
  -F "file=@my-document.pdf"
```

---

## LangGraph Agent Flow

The agent uses a **self-correcting RAG loop**:

1. **retrieve** — fetch top-6 chunks from ChromaDB
2. **grade_documents** — LLM grades each chunk for relevance (yes/no)
3. If no relevant chunks → **rewrite_query** → back to retrieve (max 2 retries)
4. **generate** — produce answer grounded strictly in context
5. **hallucination check** — verify answer is supported by context
6. If hallucinated → regenerate; if grounded → return answer

---

## Notes

- **Render free tier** spins down after 15 minutes of inactivity. The first request after spin-down may be slow (~30s). Upgrade to `starter` plan for always-on.
- ChromaDB is persisted on a **Render Disk** (1 GB free). Data survives redeploys.
- The `vectorstore/` directory is git-ignored. Re-ingest PDFs after a fresh clone.
