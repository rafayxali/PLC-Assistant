import os
import logging
from typing import List, Optional, Dict, Any, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangGraph & LangChain Core
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Ingestion/Retrieval & Generation deps
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

logger = logging.getLogger("aegis.graph")
logging.basicConfig(level=logging.INFO)

# ======================================================
# Config
# ======================================================
# Cap how many prior turns get replayed into the LLM on every call.
# Each "turn" here is one HumanMessage or one AIMessage entry.
MAX_HISTORY_MESSAGES = 20  # ~10 user/assistant exchanges

# ======================================================
# State Schema Setup
# ======================================================
class AegisState(TypedDict):
    user_id: str
    user_query: Optional[str]
    attached_image_path: Optional[str]
    chat_history: Annotated[List[BaseMessage], add_messages]
    text_search_query: str
    image_search_query: str
    retrieved_text_chunks: List[Dict[str, Any]]
    retrieved_image_captions: List[Dict[str, Any]]
    vlm_query_description: Optional[str]
    available_images: List[Dict[str, Any]]
    final_response: Optional[str]

# ======================================================
# Resource Connectors
# ======================================================
HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")

# Initialize Vector Search Model
embeddings_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN
)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
TEXT_COLLECTION = "plc_manuals_text"
IMAGE_COLLECTION = "plc_manuals_images"

# Initialize Groq Engine for ultra-fast generation
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ======================================================
# Helpers
# ======================================================
def _s(value: Any) -> str:
    """Safely coerce a payload value to str, avoiding the literal 'None' string
    that str(None) would otherwise produce for genuinely-null payload fields."""
    if value is None:
        return ""
    return str(value)

# ======================================================
# Graph Nodes (Workers)
# ======================================================

def input_analysis_node(state: AegisState) -> dict:
    """
    Step 1: Analyzes inbound technician data. Processes text requests
    and handles diagnostic files without static mock text.
    """
    raw_query = state.get("user_query") or ""
    image_path = state.get("attached_image_path")

    vlm_desc = None
    if image_path:
        filename = os.path.basename(image_path)
        vlm_desc = f"Technician attached a diagnostic file resource link: ({filename})"

    # Construct clean search context vectors using actual text inputs
    if raw_query and vlm_desc:
        text_search = f"{raw_query} {vlm_desc}"
        image_search = f"{raw_query} {vlm_desc}"
    elif vlm_desc:
        text_search = "PLC hardware interface fault diagnostic"
        image_search = "PLC layout diagnostic"
    else:
        text_search = raw_query
        image_search = raw_query

    messages_update = []
    if raw_query:
        messages_update.append(HumanMessage(content=raw_query))

    return {
        "vlm_query_description": vlm_desc,
        "text_search_query": text_search,
        "image_search_query": image_search,
        "chat_history": messages_update
    }


def text_retrieval_node(state: AegisState) -> dict:
    """
    Step 2: Vector search against manual text chunks.
    """
    search_target = state.get("text_search_query")
    if not search_target:
        return {"retrieved_text_chunks": []}
    try:
        query_vector = embeddings_model.embed_query(search_target)
        hits = qdrant_client.search(collection_name=TEXT_COLLECTION, query_vector=query_vector, limit=3)
        return {"retrieved_text_chunks": [{
            "postgres_chunk_id": _s(hit.payload.get("postgres_chunk_id")),
            "document_id": _s(hit.payload.get("document_id")),
            "page_id": _s(hit.payload.get("page_id")),
            "section_name": hit.payload.get("section_name"),
            "text": _s(hit.payload.get("text")),
            "similarity_score": hit.score
        } for hit in hits]}
    except Exception:
        logger.exception("Text retrieval failed for query: %r", search_target)
        return {"retrieved_text_chunks": []}


def image_retrieval_node(state: AegisState) -> dict:
    """
    Step 3: Vector search against manual diagram descriptions.
    Includes extracting image file targets or payload URLs if present.
    """
    search_target = state.get("image_search_query")
    if not search_target:
        return {"retrieved_image_captions": []}
    try:
        query_vector = embeddings_model.embed_query(search_target)
        hits = qdrant_client.search(collection_name=IMAGE_COLLECTION, query_vector=query_vector, limit=3)
        return {"retrieved_image_captions": [{
            "postgres_chunk_id": _s(hit.payload.get("postgres_chunk_id")),
            "document_id": _s(hit.payload.get("document_id")),
            "page_id": _s(hit.payload.get("page_id")),
            "section_name": hit.payload.get("section_name"),
            "text": _s(hit.payload.get("text")),
            # Extract image_url or local disk source path if populated in database
            "image_url": _s(hit.payload.get("image_url") or hit.payload.get("image_path")),
            "similarity_score": hit.score
        } for hit in hits]}
    except Exception:
        logger.exception("Image retrieval failed for query: %r", search_target)
        return {"retrieved_image_captions": []}


def context_synthesis_node(state: AegisState) -> dict:
    """
    Step 4: Implements the Gemini-styled OmniDoc system architecture.
    Applies logic checks, out-of-context filters, and dynamically serves reference diagrams.
    """
    text_chunks = state.get("retrieved_text_chunks", [])
    img_captions = state.get("retrieved_image_captions", [])
    history = state.get("chat_history", [])

    # Compile the retrieved reference context block
    context_blocks = []
    for c in text_chunks:
        context_blocks.append(f"[Manual Text | Doc: {c['document_id']} | Page: {c['page_id']}]: {c['text']}")

    # Store dynamic references to hardware images we have available so the
    # API layer can also surface them directly (not just via LLM markdown).
    available_images_context = []
    for i in img_captions:
        img_ref = f"[Visual Diagram | Doc: {i['document_id']} | Page: {i['page_id']}]: {i['text']}"
        if i.get("image_url"):
            img_ref += f"\n👉 AVAILABLE IMAGE REFERENCE FILE (URL/Path): {i['image_url']}"
            available_images_context.append({
                "url": i["image_url"],
                "text": i["text"],
                "document_id": i["document_id"],
                "page_id": i["page_id"],
            })
        context_blocks.append(img_ref)

    has_context = len(context_blocks) > 0
    context_str = "\n\n".join(context_blocks) if has_context else "No direct manual context retrieved."

    # Custom Adaptive Gemini/OmniDoc Prompt Layout with strict Image Delivery Protocol
    system_instruction = SystemMessage(content=(
        "You are Aegis-Vision, an advanced, highly perceptive AI diagnostic assistant specializing "
        "in Siemens PLC systems and industrial automation frameworks. Your communication style is accessible, "
        "highly analytical, and supportive—matching a seasoned systems engineer.\n\n"
        "You are provided with a retrieved 'CONTEXT' section containing text chunks from standard PDFs "
        "or automated descriptions of uploaded diagrams.\n\n"
        "CRITICAL DISCREPANCY & TRUTH RULES:\n"
        "1. OBJECTIVE TRUTH & LOGIC FIRST:\n"
        "   Never compromise on objective engineering rules, mathematics, logic, or universal physics facts. "
        "Politely and clearly explain the correct logical framework if inconsistencies exist.\n\n"
        "2. FOR HARD DATA/TEXT DOCUMENTS (PDFs, System Manuals):\n"
        "   The technical manual document is the source of truth ONLY for what that manual claims or specifies.\n\n"
        "3. FOR AUTOMATED VISUAL DESCRIPTIONS (Image Context):\n"
        "   Understand that the context for manual diagrams or uploaded media can come from automated vision parsers. "
        "Acknowledge parsing variances gently, bridge the gap, and troubleshoot the user's corrected subject layout.\n\n"
        "4. DYNAMIC IMAGE DELIVERY PROTOCOL:\n"
        "   - If the user explicitly asks to see a visual diagram, schematic, layout, or image of a device/wiring setup, "
        "     or if providing a visual reference dramatically simplifies troubleshooting the issue:\n"
        "     Look through the CONTEXT block for a line stating 'AVAILABLE IMAGE REFERENCE FILE'. If you find an available image url/path, "
        "     embed it directly into your response text using the exact standard Markdown syntax: ![Image Description](url_or_path_here).\n"
        "   - If the user asks for an image, or you need one to answer a query, but the reference data does NOT provide a valid URL/path "
        "     under the AVAILABLE IMAGE section, you MUST tell the user politely and transparently: 'I don't have that specific visual diagram available in my technical references.'\n"
        "   - If the user uploaded a photo and you find a reference diagram matching their target hardware system closer to the final solution, "
        "     embed that manual diagram to help them cross-reference structural details.\n\n"
        "5. OUT-OF-CONTEXT / UNRELATED QUESTIONS:\n"
        "   If the user asks a question completely unrelated to Siemens PLC systems or industrial automation frameworks, "
        "   politely inform them that this topic is not available within your context or area of expertise.\n\n"
        f"CONTEXT:\n{context_str}"
    ))

    # Trim replayed history so long-running sessions don't blow up token usage
    # or latency on every single turn.
    trimmed_history = history[-MAX_HISTORY_MESSAGES:] if history else history
    full_messages = [system_instruction] + trimmed_history

    try:
        llm_response = llm.invoke(full_messages)
        generated_text = llm_response.content
    except Exception:
        logger.exception("Groq LLM invocation failed")
        generated_text = "⚠️ I encountered an API communication fault while analyzing the system matrix. Please resubmit."

    return {
        "final_response": generated_text,
        "chat_history": [AIMessage(content=generated_text)],
        "available_images": available_images_context,
    }

# ======================================================
# LangGraph Workflow Execution Compilation
# ======================================================
workflow = StateGraph(AegisState)

workflow.add_node("input_analysis", input_analysis_node)
workflow.add_node("text_retrieval", text_retrieval_node)
workflow.add_node("image_retrieval", image_retrieval_node)
workflow.add_node("synthesis", context_synthesis_node)

workflow.add_edge(START, "input_analysis")
workflow.add_edge("input_analysis", "text_retrieval")
workflow.add_edge("input_analysis", "image_retrieval")
workflow.add_edge("text_retrieval", "synthesis")
workflow.add_edge("image_retrieval", "synthesis")
workflow.add_edge("synthesis", END)

# ======================================================
# Checkpointer: persistent, multi-worker-safe (Postgres/Neon)
# ======================================================
# MemorySaver is in-process/in-memory only: it does not survive restarts and
# breaks silently across multiple worker processes (e.g. `uvicorn --workers N`),
# since each worker has its own memory and a session's follow-up turn could
# land on a worker that never saw the earlier turns. We use PostgresSaver,
# backed by the same Neon database already used elsewhere in this project, so
# conversation state is durable and shared correctly across all workers.
#
# checkpoint_pool is exposed at module level so the FastAPI app can close it
# cleanly on shutdown (see main.py's shutdown_event).
checkpoint_pool: Optional[ConnectionPool] = None

if NEON_DATABASE_URL:
    try:
        checkpoint_pool = ConnectionPool(
            conninfo=NEON_DATABASE_URL,
            max_size=20,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=True,
        )
        checkpointer = PostgresSaver(checkpoint_pool)
        checkpointer.setup()  # idempotent: creates checkpoint tables if missing
        logger.info("Checkpointer: using PostgresSaver backed by Neon (persistent, multi-worker safe).")
    except Exception:
        logger.exception(
            "Failed to initialize Postgres checkpointer; falling back to in-memory "
            "MemorySaver. This is NOT safe for multi-worker deployments or restarts — "
            "fix NEON_DATABASE_URL / Postgres connectivity before going to production."
        )
        checkpointer = MemorySaver()
else:
    logger.warning(
        "NEON_DATABASE_URL is not set; falling back to in-memory MemorySaver. "
        "This is NOT safe for multi-worker deployments or restarts."
    )
    checkpointer = MemorySaver()

aegis_graph = workflow.compile(checkpointer=checkpointer)