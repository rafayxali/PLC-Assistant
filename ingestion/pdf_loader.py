import os
import fitz  # PyMuPDF
from database.database import SessionLocal, engine
from sqlalchemy import text

def init_db_documents(manuals_dir: str = "data/manuals"):
    """
    Scans the manuals directory, reads document structural properties,
    and inserts records into the PostgreSQL documents and pages tables.
    """
    if not os.path.exists(manuals_dir):
        print(f"❌ Manuals directory not found at: {manuals_dir}")
        return

    db = SessionLocal()
    print("\n[Ingestion: PDF Loader] Scanning for manual artifacts...")

    pdf_files = [f for f in os.listdir(manuals_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("⚠️ No PDF documents discovered in target folder.")
        db.close()
        return

    for file_name in pdf_files:
        local_path = os.path.join(manuals_dir, file_name)
        title = os.path.splitext(file_name)[0]
        
        # Check if document already processed to prevent duplicate keys
        check_doc = db.execute(
            text("SELECT id FROM documents WHERE title = :title"),
            {"title": title}
        ).fetchone()

        if check_doc:
            print(f" -> Skimming '{title}' (Already registered in PostgreSQL).")
            continue

        print(f"Processing structural layout of: {file_name}")
        try:
            doc = fitz.open(local_path)
            total_pages = len(doc)

            # Insert metadata record into documents table
            doc_result = db.execute(
                text("""
                    INSERT INTO documents (title, local_path, total_pages, manufacturer, plc_family)
                    VALUES (:title, :local_path, :total_pages, 'Siemens', 'S7-1200/1500')
                    RETURNING id;
                """),
                {
                    "title": title,
                    "local_path": local_path,
                    "total_pages": total_pages
                }
            )
            doc_id = doc_result.fetchone()[0]

            # Parse page granular data blocks
            for page_num in range(total_pages):
                page = doc[page_num]
                page_text = page.get_text()
                
                # Check if page contains embedded pixmaps/image streams
                has_images = len(page.get_images()) > 0

                db.execute(
                    text("""
                        INSERT INTO pages (document_id, page_number, page_text, has_images)
                        VALUES (:document_id, :page_number, :page_text, :has_images);
                    """),
                    {
                        "document_id": doc_id,
                        "page_number": page_num + 1,
                        "page_text": page_text,
                        "has_images": has_images
                    }
                )

            db.commit()
            print(f" ✅ Structural metrics for '{title}' committed with {total_pages} pages.")

        except Exception as e:
            db.rollback()
            print(f" ❌ Critical failure reading structural mapping for {file_name}: {e}")

    db.close()

if __name__ == "__main__":
    init_db_documents()