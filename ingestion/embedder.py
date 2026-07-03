import os
import time
import uuid
from dotenv import load_dotenv
from sqlalchemy import text
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from database.database import SessionLocal

load_dotenv()

# ---------------------------------------------------------
# INITIALIZE HUGGING FACE EMBEDDINGS & QDRANT
# ---------------------------------------------------------
HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

embeddings_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN
)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Collection names matching your qdrant_setup.py
TEXT_COLLECTION = "plc_manuals_text"
IMAGE_COLLECTION = "plc_manuals_images"


def process_embedding_batch(batch_size: int = 64):
    """
    Pulls unindexed text chunks from PostgreSQL, converts them to vector embeddings 
    via Hugging Face API, and bulk upserts them to the matching Qdrant collections.
    """
    db = SessionLocal()
    print(f"\n[Embeddings Engine] Fetching up to {batch_size} unindexed chunks...")

    # CRITICAL FIX: Changed 'WHERE qdrant_point_id NOT IN' to 'WHERE id NOT IN'
    # This correctly matches the chunks.id primary keys that get stored in retrieval_log.chunk_id
    chunks = db.execute(
        text("""
            SELECT id, document_id, page_id, section_name, chunk_text, qdrant_point_id 
            FROM chunks
            WHERE id NOT IN (
                SELECT DISTINCT chunk_id FROM retrieval_log WHERE chunk_id IS NOT NULL
            )
            LIMIT :limit
        """),
        {"limit": batch_size}
    ).fetchall()

    if not chunks:
        print(" -> All chunks are currently synchronized with Qdrant.")
        db.close()
        return 0

    print(f"Found {len(chunks)} chunks to embed. Generating vectors via Hugging Face...")

    # Extract text strings for batch embedding generation
    texts_to_embed = [row[4] for row in chunks]
    
    try:
        # LangChain handles batching and hits the HF API in a single trip
        vector_embeddings = embeddings_model.embed_documents(texts_to_embed)
    except Exception as e:
        print(f" ❌ Hugging Face Embedding API Error: {e}")
        db.close()
        return 0

    text_points = []
    image_points = []
    processed_chunk_ids = []

    for idx, row in enumerate(chunks):
        c_id, doc_id, page_id, section_name, chunk_text, qdrant_point_id = row
        vector = vector_embeddings[idx]

        # Construct vector payload metadata for your hybrid RAG system
        payload = {
            "postgres_chunk_id": str(c_id),
            "document_id": str(doc_id),
            "page_id": str(page_id),
            "section_name": section_name,
            "text": chunk_text
        }

        # Build Qdrant Point structural wrapper using the unique UUID from chunking
        point = PointStruct(
            id=str(qdrant_point_id),
            vector=vector,
            payload=payload
        )

        # Route the vector based on chunk type
        if section_name and "Visual Entity" in section_name:
            image_points.append(point)
        else:
            text_points.append(point)
            
        processed_chunk_ids.append(c_id)

    # Execute bulk upserts across network channels
    try:
        if text_points:
            qdrant_client.upsert(collection_name=TEXT_COLLECTION, points=text_points)
            print(f" ✅ Bulk-uploaded {len(text_points)} vectors to '{TEXT_COLLECTION}'")

        if image_points:
            qdrant_client.upsert(collection_name=IMAGE_COLLECTION, points=image_points)
            print(f" ✅ Bulk-uploaded {len(image_points)} vectors to '{IMAGE_COLLECTION}'")

        # Flag chunks as finished by registering their primary keys (id) in retrieval_log
        for chunk_uuid in processed_chunk_ids:
            db.execute(
                text("""
                    INSERT INTO retrieval_log (id, message_id, chunk_id, similarity_score, rank_position) 
                    VALUES (gen_random_uuid(), NULL, :chunk_id, 1.0, 0)
                """),
                {"chunk_id": chunk_uuid}
            )
        db.commit()

    except Exception as e:
        db.rollback()
        print(f" ❌ Vector Storage or DB update tracking fault: {e}")
        db.close()
        return 0

    db.close()
    return len(chunks)


def main():
    BATCH_SIZE = 64  # Optimal size for stable cloud endpoint requests
    total_indexed = 0
    batch_num = 1
    consecutive_errors = 0
    max_errors = 5  # Keeps script alive through transient network issues

    print("=" * 70)
    print("STARTING HUGGING FACE + QDRANT VECTOR INGESTION ENGINE")
    print("=" * 70)

    while True:
        print(f"\nStarting Batch #{batch_num}")
        try:
            indexed_count = process_embedding_batch(batch_size=BATCH_SIZE)
            
            if indexed_count == 0:
                print("\nSynchronization task finished. Vector collections are completely up to date.")
                break
                
            total_indexed += indexed_count
            batch_num += 1
            consecutive_errors = 0  # Reset counter on a successful batch
            time.sleep(0.5)    # Pacing delay
            
        except Exception as e:
            consecutive_errors += 1
            print(f"\n⚠️ Network/Database hiccup caught in main loop: {e}")
            print(f"Retrying batch in 10 seconds... (Attempt {consecutive_errors}/{max_errors})")
            
            if consecutive_errors >= max_errors:
                print("\n❌ Critical: Too many consecutive failures. Aborting run to protect execution.")
                break
                
            time.sleep(10)

    print("\n" + "=" * 70)
    print(f"VECTOR INDEXING RUN COMPLETE | Total Chunks Vectorized: {total_indexed}")
    print("=" * 70)


if __name__ == "__main__":
    main()