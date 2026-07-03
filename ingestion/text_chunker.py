import os
import time
import uuid
from dotenv import load_dotenv
from sqlalchemy import text
from database.database import SessionLocal

# Using LangChain's structural text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Configure the splitter for technical documentation
# 1000–1500 characters with a 150-character overlap works well for keeping commands and parameters together
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

def chunk_raw_pages(db, batch_size: int = 50):
    """
    Reads processed pages that haven't been chunked yet, splits them 
    using the recursive character layout, and writes records into 'chunks'.
    """
    print("\n[Chunker: Pages] Scanning for unchunked text pages...")
    
    unread_pages = db.execute(
        text("""
            SELECT p.id, p.document_id, p.page_number, p.page_text 
            FROM pages p
            LEFT JOIN chunks c ON p.id = c.page_id
            WHERE c.id IS NULL AND p.page_text IS NOT NULL AND LENGTH(TRIM(p.page_text)) > 0
            LIMIT :limit
        """),
        {"limit": batch_size}
    ).fetchall()

    if not unread_pages:
        print(" -> All available pages are fully chunked.")
        return 0

    chunks_added = 0
    for page_id, doc_id, page_num, page_text in unread_pages:
        try:
            # Split the raw string structurally
            split_texts = text_splitter.split_text(page_text)
            
            for idx, text_block in enumerate(split_texts):
                chunk_payload = text_block.strip()
                if not chunk_payload:
                    continue
                
                # Estimate token count (standard 4 chars ~ 1 token proxy)
                token_count = len(chunk_payload) // 4
                qdrant_point_id = str(uuid.uuid4())

                db.execute(
                    text("""
                        INSERT INTO chunks (document_id, page_id, chunk_index, section_name, chunk_text, token_count, qdrant_point_id)
                        VALUES (:doc_id, :page_id, :idx, :section, :text, :tokens, :qdrant_id);
                    """),
                    {
                        "doc_id": doc_id,
                        "page_id": page_id,
                        "idx": idx,
                        "section": f"Page {page_num} Block {idx + 1}",
                        "text": chunk_payload,
                        "tokens": token_count,
                        "qdrant_id": qdrant_point_id
                    }
                )
            db.commit()
            chunks_added += len(split_texts)
        except Exception as e:
            db.rollback()
            print(f" ❌ Error character chunking page ID {page_id}: {e}")
            
    print(f" ✅ Successfully ingested {chunks_added} page text chunks.")
    return len(unread_pages)


def chunk_image_and_table_descriptions(db, batch_size: int = 50):
    """
    Treats rich visual summaries and markdown tables as single, unified 
    chunks to guarantee layout context is kept entirely intact.
    """
    print("\n[Chunker: Visuals] Scanning for unchunked image descriptions and tables...")
    
    unread_images = db.execute(
        text("""
            SELECT i.id, i.document_id, i.page_id, i.image_path, i.description 
            FROM images i
            LEFT JOIN chunks c ON i.page_id = c.page_id AND c.section_name LIKE 'Visual Entity%'
            WHERE c.id IS NULL 
              AND i.description IS NOT NULL 
              AND i.description != 'TABLE_CONTENT_EXTRACTED_SEPARATELY'
              AND LENGTH(TRIM(i.description)) > 0
            LIMIT :limit
        """),
        {"limit": batch_size}
    ).fetchall()

    if not unread_images:
        print(" -> All available visual assets are fully chunked.")
        return 0

    visuals_added = 0
    for img_id, doc_id, page_id, img_path, description in unread_images:
        try:
            # Detect if this visual block is a markdown table
            is_table = "|" in description and "-|-" in description
            file_name = os.path.basename(img_path)
            section_label = f"Visual Entity (Table: {file_name})" if is_table else f"Visual Entity (Caption: {file_name})"
            
            token_count = len(description) // 4
            qdrant_point_id = str(uuid.uuid4())

            # Insert as a single chunk to protect formatting integrity
            db.execute(
                text("""
                    INSERT INTO chunks (document_id, page_id, chunk_index, section_name, chunk_text, token_count, qdrant_point_id)
                    VALUES (:doc_id, :page_id, 0, :section, :text, :tokens, :qdrant_id);
                """),
                {
                    "doc_id": doc_id,
                    "page_id": page_id,
                    "section": section_label,
                    "text": description.strip(),
                    "tokens": token_count,
                    "qdrant_id": qdrant_point_id
                }
            )
            
            # Keep structural table track synchronized if needed
            if is_table:
                check_table = db.execute(
                    text("SELECT id FROM tables WHERE image_id = :img_id"), {"img_id": img_id}
                ).fetchone()
                
                if not check_table:
                    db.execute(
                        text("""
                            INSERT INTO tables (document_id, page_id, image_id, markdown, qdrant_point_id)
                            VALUES (:doc_id, :page_id, :img_id, :markdown, :qdrant_id);
                        """),
                        {
                            "doc_id": doc_id,
                            "page_id": page_id,
                            "img_id": img_id,
                            "markdown": description.strip(),
                            "qdrant_id": qdrant_point_id
                        }
                    )

            db.commit()
            visuals_added += 1
        except Exception as e:
            db.rollback()
            print(f" ❌ Error processing image asset ID {img_id}: {e}")

    print(f" ✅ Successfully ingested {visuals_added} visual content chunks.")
    return len(unread_images)


def main():
    BATCH_SIZE = 50
    db = SessionLocal()
    
    print("=" * 70)
    print("STARTING INDUSTRIAL PLC RECURSIVE CHARACTER CHUNKING ENGINE")
    print("=" * 70)
    
    try:
        # Step 1: Process text pages
        while True:
            pages_processed = chunk_raw_pages(db, batch_size=BATCH_SIZE)
            if pages_processed == 0:
                break
            time.sleep(0.1)
            
        # Step 2: Process image captions and tables
        while True:
            visuals_processed = chunk_image_and_table_descriptions(db, batch_size=BATCH_SIZE)
            if visuals_processed == 0:
                break
            time.sleep(0.1)
            
    finally:
        db.close()
        print("\n" + "=" * 70)
        print("CHUNKING PIPELINE CYCLE COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()