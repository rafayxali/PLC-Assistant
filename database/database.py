import os
import logging
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from qdrant_client import QdrantClient

# ======================================================
# Load Environment Variables
# ======================================================

load_dotenv()

logger = logging.getLogger("aegis.database")
logging.basicConfig(level=logging.INFO)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")

if not QDRANT_URL or not NEON_DATABASE_URL:
    logger.warning(
        "One or more required env vars are missing (QDRANT_URL / NEON_DATABASE_URL). "
        "Connections will fail until these are set."
    )

# ======================================================
# Qdrant Connection
# ======================================================

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ======================================================
# Neon PostgreSQL Connection
# ======================================================

engine = create_engine(
    NEON_DATABASE_URL,
    pool_pre_ping=True,
    future=True
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

# ======================================================
# Helper Function
# ======================================================

def get_db():
    """
    Returns a SQLAlchemy session.
    Use:
        db = next(get_db())
    or
        with SessionLocal() as db:
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ======================================================
# Connection Tests
# ======================================================

def test_neon():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT NOW();"))
            print("✅ Connected to Neon PostgreSQL")
            print("Server Time:", result.scalar())
    except Exception as e:
        print("❌ Neon Connection Failed")
        print(e)


def test_qdrant():
    try:
        collections = qdrant.get_collections()
        print("✅ Connected to Qdrant")
        print("Collections:")
        for collection in collections.collections:
            print(f"   • {collection.name}")
    except Exception as e:
        print("❌ Qdrant Connection Failed")
        print(e)


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":
    print("\n========== Database Connections ==========\n")

    test_neon()
    print()

    test_qdrant()

    print("\n==========================================\n")