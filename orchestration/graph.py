import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from orchestration.schemas import AegisState

load_dotenv()

# ======================================================
# 1. Node Definitions (Workers)
# ======================================================

def input_analysis_node(state: AegisState) -> dict:
    """
    Step 1: Analyzes inputs. If an image is present, the VLM generates a 
    description. Then, cross-retrieval strings are built for BOTH vector collections.
    """
    print("\n[Node: Input Analysis] Parsing query components...")
    raw_query = state.get("user_query", "") or ""
    image_path = state.get("attached_image_path")
    
    vlm_desc = None
    if image_path:
        # MOCK: In production, this calls GPT-4o to analyze the technician's photo
        vlm_desc = "User photo displays a blinking Red SF LED on an S7-1200 central rack unit."
        print(f" -> VLM Photo Analysis: '{vlm_desc}'")

    # --- Cross-Retrieval Logic Construction ---
    # We combine text strings and visual interpretations so that BOTH search types 
    # inherit keywords from the alternative modalities.
    if raw_query and vlm_desc:
        text_search = f"{raw_query} {vlm_desc}"
        image_search = f"{raw_query} {vlm_desc}"
    elif vlm_desc:
        # Even if there is NO text query, the image description searches BOTH spaces
        text_search = vlm_desc
        image_search = vlm_desc
    else:
        text_search = raw_query
        image_search = raw_query

    print(f" -> Text Space Query Target: '{text_search}'")
    print(f" -> Image Space Query Target: '{image_search}'")

    return {
        "vlm_query_description": vlm_desc,
        "text_search_query": text_search,
        "image_search_query": image_search
    }


def text_retrieval_node(state: AegisState) -> dict:
    """
    Queries plc_manuals_text using the cross-retrieval string.
    """
    search_target = state.get("text_search_query")
    print(f"[Node: Text Retrieval] Querying 'plc_manuals_text' vector space with: '{search_target}'...")
    
    # Mocking a text match based on keywords inside the search target
    mock_chunks = []
    if "SF" in search_target or "System Fault" in search_target:
        mock_chunks.append({
            "chunk_text": "S7-1200 Hardware Section 4.1: A red SF (System Fault) light indicates firmware exception or module mismatch.",
            "source": "S7-1200 System Manual, p. 412"
        })
    return {"retrieved_text_chunks": mock_chunks}


def image_retrieval_node(state: AegisState) -> dict:
    """
    Queries plc_manuals_images using the cross-retrieval string.
    """
    search_target = state.get("image_search_query")
    print(f"[Node: Image Retrieval] Querying 'plc_manuals_images' vector space with: '{search_target}'...")
    
    # Mocking an image diagram caption match based on keywords inside the search target
    mock_images = []
    if "rack" in search_target or "LED" in search_target or "SF" in search_target:
        mock_images.append({
            "caption": "Figure 4-2: S7-1200 CPU Module LED indicator layout showing SF location.",
            "image_path": "data/images/fig_4_2.png"
        })
    return {"retrieved_image_captions": mock_images}


def context_synthesis_node(state: AegisState) -> dict:
    """
    Gathers all cross-retrieved information to generate a safe resolution response.
    """
    print("[Node: Context Synthesis] Packaging all collected knowledge blocks...")
    
    text_context = "\n".join([c["chunk_text"] for c in state.get("retrieved_text_chunks", [])])
    img_context = "\n".join([i["caption"] for i in state.get("retrieved_image_captions", [])])
    
    response = (
        "### Aegis Multi-Modal Diagnostic Resolution\n\n"
        f"**Grounded Reference Materials Discovered:**\n"
        f"- *Manual Text Chapter Details:* {text_context if text_context else 'None referenced.'}\n"
        f"- *Manual Diagram Layout Context:* {img_context if img_context else 'None referenced.'}"
    )
    return {"final_response": response}

# ======================================================
# 2. LangGraph Construction (Parallel Flow Architecture)
# ======================================================

workflow = StateGraph(AegisState)

# Add our explicit processing steps
workflow.add_node("input_analysis", input_analysis_node)
workflow.add_node("text_retrieval", text_retrieval_node)
workflow.add_node("image_retrieval", image_retrieval_node)
workflow.add_node("synthesis", context_synthesis_node)

# Step 1: Run Input Analysis
workflow.add_edge(START, "input_analysis")

# Step 2: Fan out into BOTH vector spaces at the same time to ensure cross-retrieval
workflow.add_edge("input_analysis", "text_retrieval")
workflow.add_edge("input_analysis", "image_retrieval")

# Step 3: Direct the output of both collection lookups into the Synthesis engine
workflow.add_edge("text_retrieval", "synthesis")
workflow.add_edge("image_retrieval", "synthesis")

# End path execution boundary
workflow.add_edge("synthesis", END)

app = workflow.compile()

# ======================================================
# 3. Operational Edge Case Testing
# ======================================================
if __name__ == "__main__":
    print("--- Test Case A: IMAGE ONLY Upload (Should search text AND images) ---")
    state_image_only = app.invoke({
        "user_id": "eng_01",
        "user_query": None,
        "attached_image_path": "data/attachments/panel_light.png",
        "retrieved_text_chunks": [],
        "retrieved_image_captions": []
    })
    print(state_image_only["final_response"])

    print("\n--- Test Case B: TEXT ONLY Query (Should search text AND images) ---")
    state_text_only = app.invoke({
        "user_id": "eng_01",
        "user_query": "S7-1200 SF LED status diagnostics",
        "attached_image_path": None,
        "retrieved_text_chunks": [],
        "retrieved_image_captions": []
    })
    print(state_text_only["final_response"])