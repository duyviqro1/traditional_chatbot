import os
import sys
from pathlib import Path
from qdrant_client import QdrantClient, models

env_path = Path(__file__).resolve().parent / ".env"
sys.path.insert(0, str(env_path))

try:
    from key import QDRANT_LOCAL_URL as KEY_FILE_QDRANT_LOCAL_URL
except ImportError:
    KEY_FILE_QDRANT_LOCAL_URL = None
try:
    from key import QDRANT_LOCAL_API_KEY as KEY_FILE_QDRANT_LOCAL_API_KEY
except ImportError:
    KEY_FILE_QDRANT_LOCAL_API_KEY = None
try:
    from key import QDRANT_LOCAL_COLLECTION as KEY_FILE_QDRANT_LOCAL_COLLECTION
except ImportError:
    KEY_FILE_QDRANT_LOCAL_COLLECTION = None

QDRANT_URL = (
    os.getenv("QDRANT_LOCAL_URL")
    or os.getenv("QDRANT_URL")
    or KEY_FILE_QDRANT_LOCAL_URL
    or "http://qdrant:6333"
)
QDRANT_COLLECTION = (
    os.getenv("QDRANT_LOCAL_COLLECTION")
    or os.getenv("QDRANT_COLLECTION")
    or KEY_FILE_QDRANT_LOCAL_COLLECTION
    or "medical_docs"
)
QDRANT_API_KEY = (
    os.getenv("QDRANT_LOCAL_API_KEY")
    or KEY_FILE_QDRANT_LOCAL_API_KEY
    or "qdrant_api_key"
)
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))

print("QDRANT_URL =", QDRANT_URL)
print("QDRANT_COLLECTION =", QDRANT_COLLECTION)
print("HAS_API_KEY =", bool(QDRANT_API_KEY))

if QDRANT_API_KEY:
    print("API_KEY_PREFIX =", QDRANT_API_KEY[:8])
    
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=QDRANT_TIMEOUT
)

client.create_payload_index(
    collection_name=QDRANT_COLLECTION,
    field_name="metadata.disease",
    field_schema=models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        lowercase=True,
    ),
    wait=True
)

client.create_payload_index(
    collection_name=QDRANT_COLLECTION,
    field_name="metadata.herbs",
    field_schema=models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        lowercase=True,
    ),
    wait=True
)

print("Đã tạo payload index cho metadata.disease và metadata.herbs")
