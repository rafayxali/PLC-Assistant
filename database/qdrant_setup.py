from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

EMBED_DIM = 384

try:
    client.create_collection(
        collection_name="plc_manuals_text",
        vectors_config=VectorParams(
            size=EMBED_DIM,
            distance=Distance.COSINE
        ),
    )
    print("Text collection created")
except Exception as e:
    print(e)

try:
    client.create_collection(
        collection_name="plc_manuals_images",
        vectors_config=VectorParams(
            size=EMBED_DIM,
            distance=Distance.COSINE
        ),
    )
    print("Image collection created")
except Exception as e:
    print(e)

print(client.get_collections())
print("hello")