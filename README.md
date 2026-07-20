# Lakehouse Chatbot YHCT (Traditional RAG)

This project builds a chatbot for traditional Vietnamese medicine lookup using a traditional RAG pipeline:

- Document ingestion from MinIO
- Chunking and metadata extraction
- Embedding and vector storage in Qdrant
- Hybrid retrieval (dense + sparse)
- Answer generation with OpenAI
- Conversation history storage in Postgres

This README focuses only on traditional RAG and does not describe the agentic RAG flow.

## 1. Architecture Overview

![System architecture](image/1.png)

## 2. Traditional RAG Flow

![Chunking and retrieval flow](image/2.png)

Processing summary:

1. Documents (pdf, txt, docx, md, csv, xlsx) are uploaded to a MinIO bucket.
2. The ingestion script reads new files, splits them into chunks, and extracts metadata (disease, herbs).
3. Chunks are embedded and indexed in Qdrant using Hybrid Retrieval mode.
4. Users ask questions in Streamlit.
5. The system preprocesses the question, queries Qdrant, and reranks by disease/herb entities.
6. The LLM composes an answer based on retrieved context and returns source-backed output.
7. Chat history and sessions are stored in Postgres.

## 3. Main Components (Traditional RAG)

- `streamlit_app.py`: chat UI, authentication, session management, and message display.
- `chatbot.py`: core retrieval and generation logic (hybrid search, reranking, prompting, answer composition).
- `database.py`: Postgres connection, table initialization, chat/session persistence.
- `airflow/tasks/embed_v4.py`: new file ingestion from MinIO, chunking, metadata extraction, and Qdrant upsert.
- `source_mapping.py`: maps source filenames to user-friendly document titles.
- `docker-compose.yaml`: infrastructure services (Qdrant, MinIO, Postgres, Airflow, Spark, Trino).

## 4. Environment Requirements

- Python 3.11+
- Docker and Docker Compose
- A valid OpenAI API key

## 5. Quick Setup

### Step 1. Create virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2. Configure keys and environment variables

Current code reads secrets in this order:

1. Environment variables
2. Streamlit secrets
3. `.env/key.py`

You can create `.env/key.py` like this:

```python
OPENAI_API_KEY = "your_openai_api_key"
QDRANT_LOCAL_URL = "http://localhost:6333"
QDRANT_LOCAL_API_KEY = "qdrant_api_key"
QDRANT_LOCAL_COLLECTION = "medical_docs"
LOCAL_DATABASE_URL = "postgresql://admin:admin@localhost:5433/rag_lakehouse"
```

Recommended when running ingestion on host machine (outside containers):

- `MINIO_ENDPOINT=http://localhost:9000`

### Step 3. Start infrastructure with Docker

```bash
docker compose up -d
```

The repository already includes SQL scripts in `postgresscripts/` to initialize required DB users and RAG tables.

## 6. Index Data into Vector Database

Primary ingestion script:

- `airflow/tasks/embed_v4.py`

Two common ways to run:

### Option A: Run directly with Python

```bash
set MINIO_ENDPOINT=http://localhost:9000
python airflow/tasks/embed_v4.py
```

### Option B: Run through an Airflow task (if DAG is configured)

Airflow can call the ingestion logic to scan new files and incrementally index into Qdrant.

Note: The DAG in `airflow/dags/files_into_vectorbase.py` is currently incomplete (missing imports in this repository state), so the stable option is to run `embed_v4.py` directly or create a new DAG that calls `process_new_pdfs`.

## 7. Run the Chat Application

```bash
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

## 8. Quick Checks

- Check Qdrant collection configuration:

```bash
python test.py
```

- Extract chunks by metadata filter (for debugging):

```bash
python filter_qdrant.py
```

## 9. Operational Notes

- Retrieval mode: `HYBRID` (dense + sparse BM25).
- Default embedding model: `text-embedding-3-small`.
- Default answer model: `gpt-4o-mini`.
- Chat sessions are stored in Postgres, and history is reused for contextual question rewriting.
- Prompts are designed to reduce hallucination and prioritize answers grounded in retrieved documents.

## 10. README Scope

This document describes only the traditional RAG part of the repository.
`agentic_rag.py` is intentionally out of scope.

