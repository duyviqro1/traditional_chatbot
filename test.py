import os
from qdrant_client import QdrantClient

client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection = os.getenv("QDRANT_COLLECTION", "medical_docs")
info = client.get_collection(collection)

print("Dense vectors:")
print(info.config.params.vectors)

print("Sparse vectors:")
print(info.config.params.sparse_vectors)