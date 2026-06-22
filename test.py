import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient


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

QDRANT_URL = (
    os.getenv("QDRANT_LOCAL_URL")
    or os.getenv("QDRANT_URL")
    or KEY_FILE_QDRANT_LOCAL_URL
    or "http://qdrant:6333"
)
QDRANT_API_KEY = (
    os.getenv("QDRANT_LOCAL_API_KEY")
    or KEY_FILE_QDRANT_LOCAL_API_KEY
    or "qdrant_api_key"
)
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=QDRANT_TIMEOUT,
)

print("QDRANT_URL =", QDRANT_URL)
print(qdrant_client.get_collections())
