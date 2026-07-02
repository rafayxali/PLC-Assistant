import os
import base64
import time

from dotenv import load_dotenv
from groq import Groq
from sqlalchemy import text

from database.database import SessionLocal

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


TABLE_PROMPT = """
You are extracting structured knowledge from a PLC manual.

This image contains a TABLE.

Extract EVERY piece of information from the table.

Requirements:

- Preserve rows and columns.
- Output valid GitHub Markdown table format.
- Preserve merged cells as best as possible.
- Preserve units.
- Preserve model numbers.
- Preserve addresses.
- Preserve error codes.
- Preserve module names.
- Preserve catalog numbers.
- Preserve notes.
- Preserve every visible value.

Ignore:

- Decorative borders
- Logos
- Headers/footers
- Page numbers
- Watermarks

Output ONLY the markdown table.

No explanation.
"""


def encode_image(path: str):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_table(image_path: str):

    image64 = encode_image(image_path)

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": TABLE_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image64}"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content.strip()


def process_tables(batch_size=20):

    db = SessionLocal()

    print("\nSearching for images requiring table extraction...")

    rows = db.execute(
        text("""
            SELECT id, image_path
            FROM images
            WHERE description='TABLE_CONTENT_EXTRACTED_SEPARATELY'
            LIMIT :limit
        """),
        {"limit": batch_size}
    ).fetchall()

    if not rows:
        print("No pending table images.")
        db.close()
        return 0

    print(f"Found {len(rows)} table image(s).")

    processed = 0

    for image_id, image_path in rows:

        image_path = image_path.replace("\\", "/")

        if not os.path.exists(image_path):
            print(f"Missing file: {image_path}")
            continue

        print(f"\nExtracting table: {image_path}")

        try:

            markdown = extract_table(image_path)

            db.execute(
                text("""
                    UPDATE images
                    SET description=:description
                    WHERE id=:id
                """),
                {
                    "description": markdown,
                    "id": image_id
                }
            )

            db.commit()

            processed += 1

            print("✓ Table extracted")
            print(markdown[:200] + "...")

            time.sleep(0.5)

        except Exception as e:

            db.rollback()

            print(f"Failed: {e}")

            time.sleep(2)

    db.close()

    return processed


if __name__ == "__main__":

    BATCH_SIZE = 20

    total = 0
    batch = 1

    while True:

        print("\n" + "=" * 60)
        print(f"Batch {batch}")
        print("=" * 60)

        count = process_tables(BATCH_SIZE)

        if count == 0:
            break

        total += count
        batch += 1

        print(f"\nProcessed this batch: {count}")
        print(f"Total processed: {total}")

        time.sleep(2)

    print("\nFinished.")
    print(f"Total tables extracted: {total}")