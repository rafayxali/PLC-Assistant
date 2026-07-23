# Project Aegis-Vision

### Enterprise Multimodal RAG for Field Engineering & Industrial Maintenance

Aegis-Vision is a multimodal Retrieval-Augmented Generation (RAG) system built for **Siemens PLC and industrial automation diagnostics**. It combines text and image retrieval over technical manuals with a stateful LangGraph orchestration pipeline, persistent multi-turn memory, content moderation, and PII redaction — exposed through a FastAPI gateway with a Streamlit chat frontend.

---

## 1. Problem Statement

Heavy industry customers (automotive, energy, aviation, and industrial PLC/automation) maintain large, heterogeneous documentation catalogs: tabular calibration/error-code lists, procedural troubleshooting text, and engineering schematics (wiring diagrams, rack layouts, HMI screens, ladder logic diagrams).

Standard text-only RAG fails here because it drops or misreads diagrams — useless for a technician standing in front of a control cabinet. Aegis-Vision treats schematics and screenshots as retrievable, queryable objects alongside text, so a technician can ask a mixed query like *"How do I calibrate this component?"* while attaching a photo of an HMI error screen, and get an answer grounded in the actual manual page that shows that exact diagram or table.

---

## 2. Features

- **Multi-modal retrieval** — separate vector search over manual text/table chunks and diagram/image descriptions (Qdrant).
- **Stateful conversations** — LangGraph `StateGraph` with per-session memory, checkpointed to Postgres (Neon) for durability across restarts and multi-worker deployments.
- **Security & guardrails** — input moderation and PII redaction run before every LLM call; flagged input is refused before it ever reaches the model. Outbound responses are scrubbed of PII before being returned or stored.
- **Image-aware diagnostics** — technicians can attach a photo of a fault; the pipeline routes it into a combined text+image retrieval and generation flow.
- **Rate limiting** — Redis-backed token-bucket limiter protecting the chat endpoint per client.
- **Secure uploads** — the API never trusts a client-supplied filesystem path; uploads are stored server-side and referenced only by an opaque `file_id`.
- **Document ingestion pipeline** — PDF parsing, semantic text chunking, table extraction, and image captioning, feeding two purpose-built Qdrant collections.
- **Streamlit frontend** — a ready-to-run chat UI with image attachment, source citations, and referenced-diagram previews.

---

## 3. Architecture

```
                ┌───────────────────┐
                │  Streamlit Client  │  app.py
                └─────────┬──────────┘
                          │ HTTP
                ┌─────────▼──────────┐
                │   FastAPI Gateway   │  main.py
                │  /api/v1/upload     │  ── secure file_id-based uploads
                │  /api/v1/chat       │  ── token-bucket rate limited (Redis)
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │   LangGraph Graph   │  orchestration/graph.py
                │                     │
                │  pii_redact         │
                │   ├─► moderation ───┤
                │   └─► input_analysis│
                │         ├─► text_retrieval   ──► Qdrant (plc_manuals_text)
                │         └─► image_retrieval  ──► Qdrant (plc_manuals_images)
                │                     │
                │           synthesis (GPT-4o)
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │  Neon Postgres      │  checkpointer — persistent, multi-worker-safe
                └─────────────────────┘
```

**Request flow:**

1. **`pii_redact`** — strips PII from the incoming query before any downstream component sees it.
2. **`moderation`** and **`input_analysis`** run concurrently on the redacted query.
3. **`text_retrieval`** and **`image_retrieval`** run concurrently, each querying its own Qdrant collection and filtering by similarity threshold.
4. **`synthesis`** assembles retrieved context, applies recent conversation history, and calls GPT-4o to generate a grounded answer — unless moderation flagged the turn, in which case a refusal is returned without an LLM call.
5. The generated answer is PII-redacted before being returned to the client and persisted to conversation history.

---

## 4. Document Ingestion Pipeline

The manuals contain four content categories, each handled by a dedicated ingestion stage:

| Content Type | Examples | Processing |
|---|---|---|
| Text | Concepts, programming instructions, installation/troubleshooting steps | PyMuPDF extraction → semantic chunking → embeddings |
| Tables | Error codes, CPU specs, memory areas, instruction parameters, LED status, wiring tables | Extracted separately, converted to structured Markdown before embedding |
| Images | Wiring diagrams, rack layouts, terminal diagrams, LED reference images, HMI screenshots, flowcharts, ladder logic diagrams | Extracted, captioned via a vision model, caption embedded, metadata stored in Postgres |
| OCR (conditional) | Scanned pages without extractable text | OCR applied before chunking, then treated as Text above |

```
        Siemens Manuals
                    │
                    ▼
              PDF Processing (PyMuPDF)
       ┌────────────┼────────────┐
       ▼            ▼            ▼
     Text        Tables        Images
       │            │            │
       ▼            ▼            ▼
     Chunk      Markdown     Vision Caption
       │            │            │
       └─────┬──────┘            │
             ▼                   ▼
   Embeddings (plc_manuals_text) Embeddings (plc_manuals_images)
             │                   │
             └─────────┬─────────┘
                        ▼
                     Qdrant
                        │
                        ▼
                  LangGraph (Router)
                        │
                        ▼
                Security & Guardrails
                        │
                        ▼
                   User Response
```

Image retrieval uses a **caption-based** approach: each diagram or screenshot is described by a vision model at ingestion time, and that description is embedded with the same model used for text, so both collections are queryable through one embedding pipeline. This keeps ingestion cost and latency low while still making diagrams first-class, retrievable objects.

---

## 5. Database Schema (PostgreSQL / Neon)

### Core schema

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    document_code VARCHAR(100),
    manufacturer VARCHAR(50),
    plc_family VARCHAR(100),
    version VARCHAR(30),
    document_type VARCHAR(100),
    language VARCHAR(20),
    total_pages INT,
    source_url TEXT,
    local_path TEXT,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    page_number INT,
    page_text TEXT,
    has_images BOOLEAN DEFAULT FALSE
);

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    page_id UUID REFERENCES pages(id),
    chunk_index INT,
    section_name TEXT,
    chunk_text TEXT,
    token_count INT,
    qdrant_point_id UUID
);

CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    page_id UUID REFERENCES pages(id),
    image_path TEXT,
    image_type VARCHAR(100),
    caption TEXT,
    description TEXT,
    qdrant_point_id UUID
);

CREATE TABLE tables (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    page_id UUID REFERENCES pages(id),
    image_id UUID REFERENCES images(id),
    markdown TEXT,
    qdrant_point_id UUID
);

CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    title TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    sender VARCHAR(20),
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE retrieval_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id),
    chunk_id UUID REFERENCES chunks(id),
    similarity_score FLOAT,
    rank_position INT
);

CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id),
    rating BOOLEAN,
    comment TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE error_codes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plc_family VARCHAR(100),
    error_code VARCHAR(50),
    title TEXT,
    explanation TEXT,
    recommended_action TEXT,
    document_id UUID REFERENCES documents(id),
    page_number INT
);

CREATE TABLE plc_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    family VARCHAR(100),
    model VARCHAR(100),
    firmware_version VARCHAR(30)
);

CREATE TABLE document_tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    tag VARCHAR(100)
);
```

### Operational schema (guardrails, rate limiting, attachments)

```sql
-- Audit trail of guardrail decisions (moderation flags, PII redactions)
CREATE TABLE guardrail_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id),
    check_type VARCHAR(50),       -- 'moderation' | 'pii_redaction'
    flagged BOOLEAN,
    detail JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audit trail of rate limit decisions (Redis is the live enforcement store)
CREATE TABLE rate_limit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    endpoint VARCHAR(100),
    allowed BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Query-time photos uploaded by technicians (distinct from manual-sourced images)
CREATE TABLE query_attachments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID REFERENCES messages(id),
    image_path TEXT,
    vlm_description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

> LangGraph's own conversation-state checkpointing runs through a separate Postgres connection pool (`checkpoint_pool` in `graph.py`), independent of the application tables above.

---

## 6. Qdrant Configuration

### Collection 1 — `plc_manuals_text`
| Property | Value |
|---|---|
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Distance | Cosine |
| Payload metadata | `postgres_chunk_id`, `document_id`, `page_id`, `section_name`, `text` |

### Collection 2 — `plc_manuals_images`
| Property | Value |
|---|---|
| Dense model | Same embedding model as text collection, applied to the vision-generated caption |
| Distance | Cosine |
| Payload metadata | `postgres_chunk_id`, `document_id`, `page_id`, `section_name`, `text`, `image_url` |

Both collections are filtered at query time by a similarity threshold and capped at the top 3 hits per query.

---

## 7. Project Structure

```
Aegis-Vision/
│
├── data/
│   ├── manuals/                  # source Siemens PDFs
│   ├── images/                   # extracted diagrams/screenshots
│   ├── attachments/               # technician-uploaded query-time photos
│   └── processed/                # chunks, tables, OCR output
│
├── ingestion/
│   ├── pdf_loader.py              # PyMuPDF extraction
│   ├── text_chunker.py            # semantic chunking
│   ├── table_extractor.py         # table → Markdown
│   ├── image_extractor.py         # pulls images out of PDFs
│   ├── image_captioner.py         # vision-model captioning at ingestion time
│   └── embedder.py                # embedding generation, dual-collection upload
│
├── guardrails/
│   ├── moderation.py               # content moderation
│   └── pii_redaction.py            # regex-based PII scrubber
│
├── ratelimit/
│   └── token_bucket.py             # Redis-backed limiter
│
├── orchestration/
│   └── graph.py                    # LangGraph pipeline definition
│
├── app/
│   ├── schemas.py                  # Pydantic request/response models
│   └── uploads.py                  # secure upload storage + path resolution
│
├── database/
│   └── database.py                 # Neon (SQLAlchemy) + Qdrant clients
│
├── main.py                         # FastAPI app entrypoint
├── app.py                          # Streamlit frontend
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 8. Setup

### 1. Create a virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file:

```dotenv
# --- LLM (generation) ---
OPENAI_API_KEY=sk-...

# --- Embeddings ---
HUGGINGFACEHUB_ACCESS_TOKEN=hf_...

# --- Vector store (Qdrant) ---
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=...

# --- Relational store / checkpointing (Neon Postgres) ---
NEON_DATABASE_URL=postgresql://user:password@host/dbname

# --- Rate limiter (Redis) ---
REDIS_URL=redis://localhost:6379/0

# --- CORS (production) ---
CORS_ORIGINS=https://your-frontend-domain.com
```

### 3. Run the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run the frontend

```bash
export AEGIS_API_URL=http://localhost:8000
streamlit run app.py
```

---

## 9. API Reference

### `GET /health`
Liveness check → `{ "status": "healthy", "service": "Aegis-Vision Persistent Engine" }`

### `POST /api/v1/upload`
Uploads a diagnostic image, `multipart/form-data` field `file`. Returns an opaque `file_id` — the API never accepts a raw client-supplied filesystem path or URL, closing off path-traversal / arbitrary-file-read risk.

```json
{ "file_id": "..." }
```

### `POST /api/v1/chat`
Rate-limited to **5 requests/minute per IP**.

**Request**
```json
{
  "conversation_id": "optional-uuid",
  "query": "Why is the input LED blinking red on my S7-1200?",
  "attached_image_path": "optional-file_id-from-upload"
}
```

**Response**
```json
{
  "conversation_id": "uuid",
  "answer": "...",
  "query_type": "text | image | mixed",
  "sources": [
    {
      "postgres_chunk_id": "...",
      "document_id": "...",
      "page_id": "...",
      "section_name": "...",
      "text": "...",
      "similarity_score": 0.81
    }
  ],
  "metadata": {
    "vlm_analysis_summary": "...",
    "available_images": [ { "url": "...", "text": "...", "document_id": "...", "page_id": "..." } ],
    "engine_routing": "langgraph_with_short_term_memory"
  }
}
```

If moderation flags a turn, `answer` returns a standard refusal message instead of a generated response.

---

## 10. Tech Stack

| Layer | Technology |
|---|---|
| API layer | FastAPI |
| Frontend | Streamlit |
| Orchestration | LangGraph |
| Rate limiting | Token Bucket + Redis |
| Guardrails | Custom PII regex + moderation module |
| Relational DB | PostgreSQL (Neon) |
| Vector DB | Qdrant (2 collections: text, image) |
| Text/image embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Generation | OpenAI GPT-4o |
| PDF processing | PyMuPDF |
| Image captioning | Vision model |

---

## 11. License

Add your license here (e.g. MIT, Apache 2.0, or proprietary).