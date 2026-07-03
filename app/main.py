import uuid
from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.schemas import ChatRequest, ChatResponse, SourceDocument
from database.database import get_db, test_neon, test_qdrant
from orchestration.graph import aegis_graph

app = FastAPI(
    title="Aegis-Vision API",
    description="Production industrial RAG API gateway executing state-managed conversation routing.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("\n[System Startup] Verifying active infrastructure connections...")
    test_neon()
    test_qdrant()

@app.get("/health", status_code=status.HTTP_200_OK, tags=["System Health"])
def health_check():
    return {"status": "healthy", "service": "Aegis-Vision Persistent Engine"}

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Orchestration"])
def execution_chat_pipeline(payload: ChatRequest, db: Session = Depends(get_db)):
    """
    Primary API endpoint processing technician queries using persistent, multi-turn LangGraph memory.
    """
    # 1. Bind context session state
    session_id = payload.conversation_id or str(uuid.uuid4())
    
    determined_route = "text"
    if payload.attached_image_path:
        determined_route = "mixed" if payload.query else "image"

    # 2. Setup initial dict payload state structure
    initial_state = {
        "user_id": "technician_api_user",
        "user_query": payload.query,
        "attached_image_path": payload.attached_image_path,
        "text_search_query": "",
        "image_search_query": "",
        "retrieved_text_chunks": [],
        "retrieved_image_captions": [],
        "vlm_query_description": None,
        "final_response": None
    }
    
    # 3. Configure execution state memory parameters
    config = {"configurable": {"thread_id": session_id}}
    
    # 4. Invoke graph execution node processing loops
    final_state = aegis_graph.invoke(initial_state, config=config)
    
    # Parse source structural lists
    combined_hits = final_state.get("retrieved_text_chunks", []) + final_state.get("retrieved_image_captions", [])
    response_sources = [
        SourceDocument(
            postgres_chunk_id=str(hit["postgres_chunk_id"]),
            document_id=str(hit["document_id"]),
            page_id=str(hit["page_id"]),
            section_name=hit.get("section_name"),
            text=hit["text"],
            similarity_score=round(hit["similarity_score"], 4)
        )
        for hit in combined_hits
    ]

    return ChatResponse(
        conversation_id=session_id,
        answer=final_state.get("final_response", "Unable to complete diagnostic routine."),
        query_type=determined_route,
        sources=response_sources,
        metadata={
            "vlm_analysis_summary": final_state.get("vlm_query_description"),
            "engine_routing": "langgraph_with_short_term_memory"
        }
    )