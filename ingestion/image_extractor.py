import os
import fitz  # PyMuPDF
from database.database import SessionLocal
from sqlalchemy import text

def is_valid_technical_image(base_image: dict, min_width: int = 150, min_height: int = 150) -> bool:
    """
    Pre-filtering logic: Evaluates if an extracted graphic is large enough
    to be a meaningful diagram rather than a spacer, bullet icon, or logo.
    """
    width = base_image.get("width", 0)
    height = base_image.get("height", 0)
    image_bytes = base_image.get("image", b"")
    
    # 1. Drop tiny assets (arrows, bullet points, tiny icons)
    if width < min_width or height < min_height:
        return False
        
    # 2. Drop extremely small payloads (mostly blank boxes or single-color lines)
    if len(image_bytes) < 2048:  # Less than 2 KB
        return False
        
    return True

def extract_manual_images(manuals_dir: str = "data/manuals", output_images_dir: str = "data/images"):
    """
    Extracts high-value visual graphics from manuals and syncs metadata to PostgreSQL.
    """
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    db = SessionLocal()
    print("\n[Ingestion: Image Extractor] Scanning for graphic extraction jobs...")

    documents = db.execute(text("SELECT id, title, local_path FROM documents;")).fetchall()

    if not documents:
        print("⚠️ No documents available for image parsing. Run pdf_loader first.")
        db.close()
        return

    for doc_row in documents:
        doc_id, title, local_path = doc_row
        print(f"\nProcessing visual entities from: '{title}'")
        
        try:
            doc = fitz.open(local_path)
            
            page_records = db.execute(
                text("SELECT id, page_number FROM pages WHERE document_id = :doc_id"),
                {"doc_id": doc_id}
            ).fetchall()
            
            page_id_map = {p_num: p_id for p_id, p_num in page_records}

            for page_num in range(len(doc)):
                page_index = page_num + 1
                page_id = page_id_map.get(page_index)
                if not page_id:
                    continue
                
                image_list = doc[page_num].get_images(full=True)
                
                for img_idx, img_meta in enumerate(image_list):
                    xref = img_meta[0]
                    base_image = doc.extract_image(xref)
                    
                    # --- EXECUTE IMAGE PREPROCESSING FILTER ---
                    if not is_valid_technical_image(base_image, min_width=150, min_height=150):
                        continue # Silently skip junk assets
                    
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    img_filename = f"{doc_id}_p{page_index}_img{img_idx}.{image_ext}"
                    img_filepath = os.path.join(output_images_dir, img_filename)

                    check_img = db.execute(
                        text("SELECT id FROM images WHERE image_path = :path"),
                        {"path": img_filepath}
                    ).fetchone()
                    
                    if check_img:
                        continue

                    # Save actual asset payload to disk storage
                    with open(img_filepath, "wb") as f:
                        f.write(image_bytes)

                    # Update PostgreSQL record parameters
                    db.execute(
                        text("""
                            INSERT INTO images (document_id, page_id, image_path, image_type)
                            VALUES (:document_id, :page_id, :image_path, :image_type);
                        """),
                        {
                            "document_id": doc_id,
                            "page_id": page_id,
                            "image_path": img_filepath,
                            "image_type": f"embedded/{image_ext}"
                        }
                    )
            
            db.commit()
            print(f" ✅ Image extraction completed safely for: '{title}'")

        except Exception as e:
            db.rollback()
            print(f" ❌ Extraction fault during processing of {title}: {e}")

    db.close()

if __name__ == "__main__":
    extract_manual_images()