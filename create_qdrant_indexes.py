import os
import sys
from pathlib import Path
from qdrant_client import QdrantClient, models

env_path = Path(__file__).resolve().parent / ".env"
sys.path.insert(0, str(env_path))

try:
    from key import QDRANT_CLOUD_API_KEY as KEY_FILE_QDRANT_CLOUD_API_KEY
except ImportError:
    KEY_FILE_QDRANT_CLOUD_API_KEY = None

QDRANT_URL = os.getenv(
    "QDRANT_CLOUD_URL",
    "https://1a2c93a3-63cc-4363-bcf0-ccf4f0640ed1.us-east-1-1.aws.cloud.qdrant.io",
)

QDRANT_COLLECTION = os.getenv("QDRANT_CLOUD_COLLECTION", "medical_docs")

QDRANT_API_KEY = (
    os.getenv("QDRANT_CLOUD_API_KEY")
    or KEY_FILE_QDRANT_CLOUD_API_KEY
    or os.getenv("QDRANT_API_KEY")
)

print("QDRANT_URL =", QDRANT_URL)
print("QDRANT_COLLECTION =", QDRANT_COLLECTION)
print("HAS_API_KEY =", bool(QDRANT_API_KEY))

if QDRANT_API_KEY:
    print("API_KEY_PREFIX =", QDRANT_API_KEY[:8])
    
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120
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
