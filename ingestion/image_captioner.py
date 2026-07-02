import os
import time
import base64

from dotenv import load_dotenv
from groq import Groq
from sqlalchemy import text

from database.database import SessionLocal

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

PROMPT = """
You are generating knowledge for an industrial PLC Retrieval-Augmented Generation (RAG) system.

Your task is to convert this image into a rich, retrieval-friendly textual description.

Analyze the image carefully and include ALL relevant technical information.

If the image contains:

• PLC hardware:
  - Identify the PLC family (e.g. Siemens S7-1200, S7-1500) if visible.
  - Identify CPUs, modules, expansion cards, communication modules, power supplies and other hardware.
  - Mention model numbers, serial numbers or product identifiers whenever visible.
  - Describe ports, terminals, LEDs, buttons, switches and connectors.

• Wiring diagrams:
  - Explain the wiring layout.
  - Describe signal flow.
  - Mention digital inputs, digital outputs, analog inputs, analog outputs, relays, sensors, actuators, motors, power supplies and communication lines.

• Network diagrams:
  - Describe communication between devices.
  - Mention Ethernet, PROFINET, PROFIBUS, Modbus, RS-232, RS-485 or any visible communication protocols.

• Software screenshots:
  - Identify the software if recognizable.
  - Describe menus, toolbars, ladder logic, function block diagrams, structured text, tags, variables, projects and configuration windows.

• Electrical schematics:
  - Explain the purpose of the schematic.
  - Mention electrical symbols, components, labels and connections.

• Flowcharts or block diagrams:
  - Explain the process flow.
  - Describe relationships between components.

If the image is primarily a table or spreadsheet, return exactly:
TABLE_CONTENT_EXTRACTED_SEPARATELY

OCR REQUIREMENT:
Extract every visible piece of readable text except decorative page elements.

Include:
- PLC model numbers
- Module names
- Labels
- Pin names
- Terminal identifiers
- Variable names
- Error codes
- Addresses
- Buttons
- Menu names
- Captions

Ignore:
- Decorative borders
- Company logos
- Page numbers
- Headers
- Footers
- Watermarks

Output requirements:
- Produce ONE continuous paragraph.
- Do NOT use markdown.
- Do NOT use bullet points.
- Do NOT say "The image shows..."
- Do NOT mention that you are an AI.
- Write concise but information-dense text optimized for semantic search and vector embeddings.
"""


def encode_image(path: str):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_caption(image_path: str):

    image64 = encode_image(image_path)

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image64}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content.strip()


def caption_extracted_images(batch_limit: int = 20):

    db = SessionLocal()

    print("\n[Groq Image Captioner] Looking for uncaptioned images...")

    images = db.execute(
        text("""
            SELECT id, image_path
            FROM images
            WHERE (description IS NULL OR description = '')
              AND LOWER(image_path) NOT LIKE '%.jpx'
            LIMIT :limit
        """),
        {"limit": batch_limit},
    ).fetchall()

    if not images:
        db.close()
        return 0

    print(f"Found {len(images)} image(s).")

    processed = 0

    for img_id, img_path in images:

        img_path = img_path.replace("\\", "/")

        # Extra safety (should never happen because SQL filters them)
        if img_path.lower().endswith(".jpx"):
            print(f"⏭ Skipping JPX image: {img_path}")
            continue

        if not os.path.exists(img_path):
            print(f"⚠ Missing image: {img_path}")
            continue

        print(f"\nProcessing: {img_path}")

        try:

            description = generate_caption(img_path)

            db.execute(
                text("""
                    UPDATE images
                    SET description = :description
                    WHERE id = :id
                """),
                {
                    "description": description,
                    "id": img_id,
                },
            )

            db.commit()

            processed += 1

            print("✅ Caption saved successfully.")
            print(description[:200] + "...")

            # avoid hammering Groq API
            time.sleep(0.5)

        except Exception as e:

            db.rollback()

            print(f"❌ Error processing {img_path}")
            print(e)

            # small backoff before continuing
            time.sleep(2)

    db.close()

    return processed


if __name__ == "__main__":

    BATCH_SIZE = 20

    total_processed = 0
    batch_number = 1

    start_time = time.time()

    while True:

        print("\n" + "=" * 70)
        print(f"Starting Batch #{batch_number}")
        print("=" * 70)

        processed = caption_extracted_images(batch_limit=BATCH_SIZE)

        if processed == 0:
            break

        total_processed += processed
        batch_number += 1

        print("\nBatch completed.")
        print(f"Processed this batch : {processed}")
        print(f"Total processed      : {total_processed}")

        # pause between batches
        time.sleep(2)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("IMAGE CAPTIONING COMPLETE")
    print("=" * 70)
    print(f"Total images captioned : {total_processed}")
    print(f"Elapsed time           : {elapsed:.2f} seconds")
    print("=" * 70)