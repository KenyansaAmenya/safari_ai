from sentence_transformers import SentenceTransformer

from src.config import settings
from src.domain.exceptions import EmbeddingError

class EmbeddingGenerator:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        self.dimension = settings.embedding_dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )    
            return embeddings.tolist()
        except Exceptions as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}")

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]            