from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict

class AegisState(TypedDict):
    """
    State tracking object passed through the Aegis-Vision LangGraph pipeline.
    """
    user_id: str
    user_query: Optional[str]
    attached_image_path: Optional[str]  # Path to technician's uploaded photo if present
    
    # Unified Query representations used for searching collections
    text_search_query: str              # What we will pass to the text embedding model
    image_search_query: str             # What we will pass to the image caption embedding model
    
    # Internal context accumulation (Filled concurrently)
    retrieved_text_chunks: List[Dict[str, Any]]
    retrieved_image_captions: List[Dict[str, Any]]
    vlm_query_description: Optional[str]  # VLM's analysis of the incoming photo if present
    
    # Output
    final_response: Optional[str]