import uuid
import httpx
import chromadb
from pathlib import Path
from langfuse.decorators import observe

class FeedbackRAGStore:
    """Stores and retrieves human feedback using image embeddings via ChromaDB."""
    
    def __init__(self, db_path: str = "/app/runs/chroma"):
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="image_feedback",
            metadata={"hnsw:space": "cosine"}
        )

    @observe()
    def _get_embedding(self, image_b64: str, backend_url: str, backend_model: str) -> list[float]:
        try:
            resp = httpx.post(
                f"{backend_url}/embeddings",
                json={
                    "model": backend_model,
                    "input": f"data:image/jpeg;base64,{image_b64}"
                },
                timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                print(f"Embedding error: {data['error']}")
                return []
            return data["data"][0]["embedding"]
        except Exception as e:
            print(f"Embedding request failed: {e}")
            return []

    @observe()
    def add_feedback(self, image_b64: str, feedback_text: str, backend_url: str, backend_model: str) -> None:
        emb = self._get_embedding(image_b64, backend_url, backend_model)
        if not emb:
            return
        doc_id = uuid.uuid4().hex
        self.collection.add(
            ids=[doc_id],
            embeddings=[emb],
            documents=[feedback_text]
        )

    @observe()
    def search_similar(self, image_b64: str, backend_url: str, backend_model: str, k: int = 2) -> list[str]:
        if self.collection.count() == 0:
            return []
        emb = self._get_embedding(image_b64, backend_url, backend_model)
        if not emb:
            return []
        # Filter out results with high distance (cosine distance: smaller is more similar)
        # We only want to use feedback if the image is actually somewhat similar.
        results = self.collection.query(
            query_embeddings=[emb],
            n_results=min(k, self.collection.count())
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        
        # Only keep results where distance is reasonably close
        docs = []
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            if dist < 0.3:  # Cosine distance threshold
                docs.append(doc)
        return docs
