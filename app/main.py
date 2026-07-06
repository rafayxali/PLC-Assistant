import os
import uuid
import logging
from typing import Any

from fastapi import FastAPI, Depends, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.schemas import ChatRequest, ChatResponse, SourceDocument, UploadResponse
from app.uploads import resolve_upload_path, save_upload_file
from database.database import get_db, test_neon, test_qdrant
from orchestration.graph import aegis_graph, checkpoint_pool

# Import the rate limiting dependency components.
# NOTE: we do NOT import the raw `redis_client` variable here anymore - that
# was a bug (see token_bucket.py docstring on close_redis_client). Always go
# through get_redis_client() / close_redis_client() instead.
from ratelimit.token_bucket import get_redis_client, close_redis_client, RateLimitDependency

logger = logging.getLogger("aegis.main")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Aegis-Vision API",
    description="Production industrial RAG API gateway executing state-managed conversation routing.",
    version="1.0.0"
)

# Instantiate rate limiter targeting exactly 5 queries per minute per IP address
# max_tokens=5: Permits immediate 5-turn response spikes when needed
# refill_rate=5/60: Generates tokens continuously, fully refilling over 60s
# (kept as the exact fraction 5/60 rather than a rounded 0.083 so the
# "5 per minute" comment is literally true over long-running processes).
chat_rate_limiter = RateLimitDependency(max_tokens=5, refill_rate=5 / 60)

# Configure allowed origins via env var (comma-separated). Falls back to "*"
# for local development only — wildcard origin + allow_credentials=True is
# both spec-invalid in browsers and a real security smell in production, so
# set CORS_ORIGINS explicitly (e.g. "https://app.example.com") before deploying.
_raw_origins = os.getenv("CORS_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
_allow_credentials = ALLOWED_ORIGINS != ["*"]

if ALLOWED_ORIGINS == ["*"]:
    logger.warning(
        "CORS_ORIGINS not set — defaulting to wildcard '*' with allow_credentials=False. "
        "Set CORS_ORIGINS to your real frontend domain(s) before production."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("\n[System Startup] Verifying active infrastructure connections...")
    test_neon()
    test_qdrant()

    # Establish connection and verify host status using clean, native async/await
    try:
        await get_redis_client().ping()
        print("[System Startup] Native local Redis service connected successfully.")
    except Exception as e:
        logger.error("Failed to connect to Redis during startup: %s", e)
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    if checkpoint_pool is not None:
        checkpoint_pool.close()
        print("[System Shutdown] Closed Postgres checkpoint connection pool.")

    # Gracefully disconnect and clean up active Redis sockets natively.
    # Routed through close_redis_client() (rather than a locally-imported
    # `redis_client` reference) so we always close the actual live client
    # living inside the token_bucket module.
    try:
        await close_redis_client()
        print("[System Shutdown] Closed Redis client pool connections.")
    except Exception as e:
        logger.error("Error closing Redis connection pool: %s", e)

@app.get("/health", status_code=status.HTTP_200_OK, tags=["System Health"])
def health_check():
    return {"status": "healthy", "service": "Aegis-Vision Persistent Engine"}


@app.post("/api/v1/upload", response_model=UploadResponse, tags=["Uploads"])
def upload_diagnostic_image(file: UploadFile = File(...)):
    """
    Stores a technician-uploaded diagnostic photo under a server-controlled
    upload directory and returns an opaque file_id (UUID). Pass this file_id
    back as `attached_image_path` on the /api/v1/chat request — the API never
    accepts or trusts a raw client-supplied filesystem path or URL for this
    field, which closes off path-traversal / arbitrary-file-read risk.
    """
    file_id = save_upload_file(file)
    return UploadResponse(file_id=file_id)


def _safe_str(value: Any) -> str:
    """Avoid turning a genuinely-null payload field into the literal string 'None'."""
    if value is None:
        return ""
    return str(value)


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Orchestration"])
def execution_chat_pipeline(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    _=Depends(chat_rate_limiter)  # Enforces the Token Bucket limit here before processing
):
    """
    Primary API endpoint processing technician queries using persistent, multi-turn LangGraph memory.
    """
    # 1. Bind context session state
    session_id = payload.conversation_id or str(uuid.uuid4())

    # Resolve the client-supplied file_id (a bare UUID from /api/v1/upload)
    # to a real, server-verified path under UPLOAD_DIR. We never pass a raw
    # client-supplied path/URL into the graph.
    resolved_image_path = None
    if payload.attached_image_path:
        resolved_image_path = resolve_upload_path(payload.attached_image_path)
        if resolved_image_path is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid or unknown attached_image_path. Upload the file via "
                    "/api/v1/upload first and pass the returned file_id here."
                )
            )

    determined_route = "text"
    if resolved_image_path:
        determined_route = "mixed" if payload.query else "image"

    # 2. Setup initial dict payload state structure
    initial_state = {
        "user_id": "technician_api_user",
        "user_query": payload.query,
        "attached_image_path": str(resolved_image_path) if resolved_image_path else None,
        "text_search_query": "",
        "image_search_query": "",
        "retrieved_text_chunks": [],
        "retrieved_image_captions": [],
        "vlm_query_description": None,
        "available_images": [],
        "final_response": None
    }

    # 3. Configure execution state memory parameters
    config = {"configurable": {"thread_id": session_id}}

    # 4. Invoke graph execution node processing loops
    try:
        final_state = aegis_graph.invoke(initial_state, config=config)
    except Exception:
        logger.exception("Graph execution failed for session_id=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The diagnostic pipeline failed to complete this request. Please try again."
        )

    # Parse source structural lists, guarding against None values that would
    # otherwise stringify to the literal "None".
    combined_hits = final_state.get("retrieved_text_chunks", []) + final_state.get("retrieved_image_captions", [])
    response_sources = [
        SourceDocument(
            postgres_chunk_id=_safe_str(hit.get("postgres_chunk_id")),
            document_id=_safe_str(hit.get("document_id")),
            page_id=_safe_str(hit.get("page_id")),
            section_name=hit.get("section_name"),
            text=_safe_str(hit.get("text")),
            similarity_score=round(hit.get("similarity_score", 0.0), 4)
        )
        for hit in combined_hits
    ]

    return ChatResponse(
        conversation_id=session_id,
        answer=final_state.get("final_response") or "Unable to complete diagnostic routine.",
        query_type=determined_route,
        sources=response_sources,
        metadata={
            "vlm_analysis_summary": final_state.get("vlm_query_description"),
            "available_images": final_state.get("available_images", []),
            "engine_routing": "langgraph_with_short_term_memory"
        }
    )