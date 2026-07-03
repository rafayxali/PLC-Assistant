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
from langchain_openai import ChatOpenAI

# Guardrails
from guardrails.moderation import check_moderation
from guardrails.pii_redaction import redact_pii

load_dotenv()

logger = logging.getLogger("aegis.graph")
logging.basicConfig(level=logging.INFO)

# ======================================================
# Config
# ======================================================
# Cap how many prior turns get replayed into the LLM on every call.
# Tuned down to 5 to prevent token bloat and keep historical contexts immediate.
MAX_HISTORY_MESSAGES = 5  

# Vector search similarity filtering threshold
RETRIEVAL_SIMILARITY_THRESHOLD = 0.35

MODERATION_REFUSAL_MESSAGE = (
    "I'm not able to help with that request. If you think this was flagged "
    "in error, please rephrase or contact support."
)

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
    moderation_flagged: Optional[bool]
    moderation_categories: List[str]
    pii_redaction_counts: Dict[str, int]

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

# Initialize OpenAI Engine for generation
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ======================================================
# Helpers
# ======================================================
def _s(value: Any) -> str:
    """Safely coerce a payload value to str, avoiding the literal 'None' string."""
    if value is None:
        return ""
    return str(value)

# ======================================================
# Graph Nodes (Workers)
# ======================================================

def pii_redact_node(state: AegisState) -> dict:
    """
    Step 0: Local regex-based PII redaction. Runs before the query fans out 
    to moderation and retrieval so both branches see the same redacted text.
    """
    raw_query = state.get("user_query") or ""
    if not raw_query:
        return {"pii_redaction_counts": {}}

    redacted_query, pii_counts = redact_pii(raw_query)
    if pii_counts:
        logger.info(
            "Redacted PII from incoming query for user_id=%s categories=%s",
            state.get("user_id"), list(pii_counts.keys()),
        )

    return {"user_query": redacted_query, "pii_redaction_counts": pii_counts}


def moderation_node(state: AegisState) -> dict:
    """
    Runs concurrently with input_analysis/retrieval.
    Provides network check verification before final synthesis gates execute.
    """
    query = state.get("user_query") or ""
    if not query:
        return {"moderation_flagged": False, "moderation_categories": []}

    mod_result = check_moderation(text=query)
    if mod_result.flagged:
        logger.warning(
            "Moderation flagged incoming query for user_id=%s categories=%s",
            state.get("user_id"), mod_result.categories,
        )
    return {"moderation_flagged": mod_result.flagged, "moderation_categories": mod_result.categories}


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

    return {
        "vlm_query_description": vlm_desc,
        "text_search_query": text_search,
        "image_search_query": image_search
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
        hits = qdrant_client.query_points(collection_name=TEXT_COLLECTION, query=query_vector, limit=3).points
        hits = [h for h in hits if h.score >= RETRIEVAL_SIMILARITY_THRESHOLD]
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
    """
    search_target = state.get("image_search_query")
    if not search_target:
        return {"retrieved_image_captions": []}
    try:
        query_vector = embeddings_model.embed_query(search_target)
        hits = qdrant_client.query_points(collection_name=IMAGE_COLLECTION, query=query_vector, limit=3).points
        hits = [h for h in hits if h.score >= RETRIEVAL_SIMILARITY_THRESHOLD]
        return {"retrieved_image_captions": [{
            "postgres_chunk_id": _s(hit.payload.get("postgres_chunk_id")),
            "document_id": _s(hit.payload.get("document_id")),
            "page_id": _s(hit.payload.get("page_id")),
            "section_name": hit.payload.get("section_name"),
            "text": _s(hit.payload.get("text")),
            "image_url": _s(hit.payload.get("image_url") or hit.payload.get("image_path")),
            "similarity_score": hit.score
        } for hit in hits]}
    except Exception:
        logger.exception("Image retrieval failed for query: %r", search_target)
        return {"retrieved_image_captions": []}


def context_synthesis_node(state: AegisState) -> dict:
    """
    Step 4: Implements the Gemini-styled OmniDoc system architecture.
    Applies logic checks, handles out-of-context blocks, and requests generation from OpenAI.
    """
    text_chunks = state.get("retrieved_text_chunks", [])
    img_captions = state.get("retrieved_image_captions", [])
    history = state.get("chat_history", [])

    context_blocks = []
    for c in text_chunks:
        context_blocks.append(f"[Manual Text | Doc: {c['document_id']} | Page: {c['page_id']}]: {c['text']}")

    available_images_context = []
    for i in img_captions:
        img_ref = f"[Visual Diagram Description | Doc: {i['document_id']} | Page: {i['page_id']}]: {i['text']}"
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

    if state.get("moderation_flagged"):
        logger.info("Skipping LLM call for user_id=%s — moderation flagged this turn.", state.get("user_id"))
        return {
            "final_response": MODERATION_REFUSAL_MESSAGE,
            "chat_history": [AIMessage(content=MODERATION_REFUSAL_MESSAGE)],
            "available_images": [],
        }

    system_instruction = SystemMessage(content=(
        "You are Aegis-Vision, an advanced, highly perceptive AI diagnostic assistant specializing "
        "in Siemens PLC systems and industrial automation frameworks. Your communication style is accessible, "
        "highly analytical, and supportive—matching a seasoned systems engineer.\n\n"
        "You are provided with a retrieved 'CONTEXT' section containing text chunks from standard PDFs "
        "and descriptions of automated diagrams.\n\n"
        "CRITICAL DISCREPANCY & TRUTH RULES:\n"
        "1. OBJECTIVE TRUTH & LOGIC FIRST:\n"
        "   Never compromise on objective engineering rules, mathematics, logic, or universal physics facts. "
        "Politely and clearly explain the correct logical framework if inconsistencies exist.\n\n"
        "2. FOR HARD DATA/TEXT DOCUMENTS (PDFs, System Manuals):\n"
        "   The technical manual document is the source of truth for what that manual claims or specifies.\n\n"
        "3. FOR AUTOMATED VISUAL DESCRIPTIONS (Image Context):\n"
        "   The context block contains structural textual descriptions of diagrams. Use these text descriptions "
        "   as valuable raw data to extract engineering definitions, LED blink patterns, or wiring pin layouts "
        "   to answer the question, even if you do not render the physical image file itself.\n\n"
        "4. DYNAMIC IMAGE DELIVERY PROTOCOL:\n"
        "   Answer the user's technical engineering query immediately using all text data and manual descriptions "
        "   found in the CONTEXT section. Do not refuse to answer simply because a physical image file isn't rendered.\n"
        "   - Only embed a physical image using markdown syntax `![Image Description](url_or_path_here)` if the "
        "     user explicitly asks to see an image/layout, AND a matching line stating 'AVAILABLE IMAGE REFERENCE FILE' "
        "     is present in the context.\n\n"
        "5. OUT-OF-CONTEXT / UNRELATED QUESTIONS:\n"
        "   If the user asks a question completely unrelated to Siemens PLC systems, industrial electronics, or industrial "
        "   automation frameworks, politely inform them that this topic is not available within your context.\n\n"
        f"CONTEXT:\n{context_str}"
    ))

    # Keep only the most recent conversation turns
    trimmed_history = history[-MAX_HISTORY_MESSAGES:] if history else []

    # Current user query (after PII redaction)
    current_query = state.get("user_query") or ""

    # Build the conversation for the LLM
    full_messages = [system_instruction]
    full_messages.extend(trimmed_history)

    # Always append the current user message
    if current_query:
        full_messages.append(HumanMessage(content=current_query))

    try:
        llm_response = llm.invoke(full_messages)
        generated_text = llm_response.content
        if not generated_text:
            logger.warning(
                "OpenAI returned empty content for user_id=%s (finish_reason=%s); "
                "substituting a fallback response instead of an empty string.",
                state.get("user_id"),
                getattr(llm_response, "response_metadata", {}).get("finish_reason"),
            )
            generated_text = (
                "I wasn't able to generate a response to that. Could you rephrase "
                "your question, or let me know if it's related to a Siemens PLC "
                "or industrial automation issue?"
            )
    except Exception:
        logger.exception("OpenAI LLM invocation failed")
        generated_text = "⚠️ I encountered an API communication fault while analyzing the system matrix. Please resubmit."

    redacted_text, pii_counts = redact_pii(generated_text)
    if pii_counts:
        logger.info(
            "Redacted PII from outbound LLM response for user_id=%s categories=%s",
            state.get("user_id"), list(pii_counts.keys()),
        )

    # Persist the latest conversation turn
    messages_to_store = []

    if current_query:
        messages_to_store.append(HumanMessage(content=current_query))

    messages_to_store.append(AIMessage(content=redacted_text))

    return {
        "final_response": redacted_text,
        "chat_history": messages_to_store,
        "available_images": available_images_context,
        "pii_redaction_counts": {
            **state.get("pii_redaction_counts", {}),
            **pii_counts,
        },
    }

# ======================================================
# LangGraph Workflow Execution Compilation
# ======================================================
workflow = StateGraph(AegisState)

workflow.add_node("pii_redact", pii_redact_node)
workflow.add_node("moderation", moderation_node)
workflow.add_node("input_analysis", input_analysis_node)
workflow.add_node("text_retrieval", text_retrieval_node)
workflow.add_node("image_retrieval", image_retrieval_node)
workflow.add_node("synthesis", context_synthesis_node)

workflow.add_edge(START, "pii_redact")

workflow.add_edge("pii_redact", "moderation")
workflow.add_edge("pii_redact", "input_analysis")

workflow.add_edge("input_analysis", "text_retrieval")
workflow.add_edge("input_analysis", "image_retrieval")

workflow.add_edge("moderation", "synthesis")
workflow.add_edge("text_retrieval", "synthesis")
workflow.add_edge("image_retrieval", "synthesis")

workflow.add_edge("synthesis", END)

# ======================================================
# Checkpointer: persistent, multi-worker-safe (Postgres/Neon)
# ======================================================
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
        checkpointer.setup()
        logger.info("Checkpointer: using PostgresSaver backed by Neon (persistent, multi-worker safe).")
    except Exception:
        logger.exception(
            "Failed to initialize Postgres checkpointer; falling back to in-memory "
            "MemorySaver. This is NOT safe for multi-worker deployments or restarts."
        )
        checkpointer = MemorySaver()
else:
    logger.warning(
        "NEON_DATABASE_URL is not set; falling back to in-memory MemorySaver."
    )
    checkpointer = MemorySaver()

aegis_graph = workflow.compile(checkpointer=checkpointer)