from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://f8631513-7fca-42c8-a65c-163a78013622.sa-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YWIyODMxYWYtOTc5MS00ZjhlLThmMGQtNDA4N2I1YjY4NDY4In0.VRYVIimFXav-ritZXyaYHiv5tLLB7JF6VycfpSjIwSs",
)

print(qdrant_client.get_collections())

