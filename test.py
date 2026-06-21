import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient


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
QDRANT_API_KEY = (
    os.getenv("QDRANT_CLOUD_API_KEY")
    or os.getenv("QDRANT_API_KEY")
    or KEY_FILE_QDRANT_CLOUD_API_KEY
)
QDRANT_TIMEOUT = int(os.getenv("QDRANT_CLOUD_TIMEOUT", "120"))

if not QDRANT_API_KEY:
    raise RuntimeError("Missing QDRANT_CLOUD_API_KEY in environment or .env/key.py.")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=QDRANT_TIMEOUT,
)

print(qdrant_client.get_collections())
