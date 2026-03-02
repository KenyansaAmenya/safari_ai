import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    
    # Initialize the embedding generator with a specified model
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        
        # Try to load the model and log the embedding dimension
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    # Generate embeddings for a list of texts
    def generate(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Generate embeddings with progress bar and normalization for cosine similarity
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
            batch_size=32
        )
        
        # Log the shape of the generated embeddings
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    # Generate embeddings for a list of chunks
    def generate_for_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        texts = [chunk['content'] for chunk in chunks]
        return self.generate(texts)
    
    # Encode a single query string into an embedding
    def encode_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding[0]