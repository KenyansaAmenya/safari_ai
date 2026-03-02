import chromadb
import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# VectorStore class to manage vector storage and retrieval using ChromaDB
class VectorStore:
    
    # Initialize the vector store with a specified type and persistence path
    def __init__(self, 
                 store_type: str = "chroma",
                 persist_path: str = "data/vector_store"):
        self.store_type = store_type.lower()
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Common attributes
        self.collection = None
        self.index = None
        self.chunks = []
        self.dimension = None
        
        # ChromaDB client
        self.chroma_client = None

    # Initialize the vector store based on the specified type    
    def initialize(self, dimension: int):
        self.dimension = dimension
        

        if self.store_type == "chroma":
            self._init_chroma()
        elif self.store_type == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"Unknown store type: {self.store_type}")
    
    # Initialize ChromaDB collection and client - FIXED VERSION
    def _init_chroma(self):
        chroma_path = self.persist_path / "chroma"
        
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        
        
        try:
            self.chroma_client.delete_collection("tourism_kb")
            logger.info("Deleted existing ChromaDB collection")
        except:
            pass
        
        self.collection = self.chroma_client.create_collection(
            name="tourism_kb",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("ChromaDB initialized")
    
    def _init_faiss(self):
        
        # Using IndexFlatIP for cosine similarity with normalized vectors
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info("FAISS index initialized")
    
    def add_documents(self, 
                      chunks: List[Dict[str, Any]], 
                      embeddings: np.ndarray):
        """Add documents and their embeddings to the store."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        self.chunks = chunks
        
        if self.store_type == "chroma":
            self._add_to_chroma(chunks, embeddings)
        else:
            self._add_to_faiss(embeddings)
        
        # Save metadata
        self._save_metadata()
    
    def _add_to_chroma(self, chunks: List[Dict], embeddings: np.ndarray):
        """Add documents to ChromaDB."""
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [c['content'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        
        # Convert to list for Chroma
        embeddings_list = embeddings.tolist()
        
        # Add in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                embeddings=embeddings_list[i:end_idx]
            )
            
            logger.info(f"Added batch {i//batch_size + 1}: {i}-{end_idx}")
        
        logger.info(f"Added {len(chunks)} documents to ChromaDB")
    
    def _add_to_faiss(self, embeddings: np.ndarray):
        """Add vectors to FAISS index."""
        # FAISS expects float32
        vectors = embeddings.astype('float32')
        self.index.add(vectors)
        logger.info(f"Added {len(vectors)} vectors to FAISS")
    
    def search(self, 
               query_embedding: np.ndarray, 
               top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filters: Metadata filters (ChromaDB only)
            
        Returns:
            List of results with content, metadata, and score
        """
        if self.store_type == "chroma":
            return self._search_chroma(query_embedding, top_k, filters)
        else:
            return self._search_faiss(query_embedding, top_k)
    
    def _search_chroma(self, 
                       query_embedding: np.ndarray,
                       top_k: int,
                       filters: Optional[Dict]) -> List[Dict]:
        """Search ChromaDB."""
        # Build where clause for metadata filtering
        where_clause = None
        if filters:
            where_clause = {}
            for key, value in filters.items():
                if value:
                    where_clause[key] = {"$eq": value}
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted
    
    def _search_faiss(self, 
                      query_embedding: np.ndarray,
                      top_k: int) -> List[Dict]:
        """Search FAISS index."""
    
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_vector, top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'id': f"chunk_{idx}",
                    'content': chunk['content'],
                    'metadata': chunk['metadata'],
                    'score': float(score)  
                })
        
        return results
    
    def _save_metadata(self):
        """Save chunk metadata for later retrieval."""
        metadata_path = self.persist_path / "chunks_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def persist(self):
        """Explicitly persist the vector store."""
        if self.store_type == "chroma":
            pass
        else:
            # Save FAISS index
            faiss_path = self.persist_path / "faiss.index"
            faiss.write_index(self.index, str(faiss_path))
            logger.info(f"Saved FAISS index to {faiss_path}")
    
    def load(self):
        """Load existing vector store."""
        metadata_path = self.persist_path / "chunks_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"No existing vector store found at {self.persist_path}")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        if self.store_type == "chroma":
            self._load_chroma()
        else:
            self._load_faiss()
        
        logger.info(f"Loaded vector store with {len(self.chunks)} documents")
    
    def _load_chroma(self):
        """Load ChromaDB collection - FIXED VERSION."""
        chroma_path = self.persist_path / "chroma"
        
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        
        self.collection = self.chroma_client.get_collection("tourism_kb")
    
    # Load FAISS index from disk
    def _load_faiss(self):
        faiss_path = self.persist_path / "faiss.index"
        
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
        
        self.index = faiss.read_index(str(faiss_path))
        self.dimension = self.index.d
    
    # Get statistics about the vector store
    def get_stats(self) -> Dict[str, Any]:
        return {
            'store_type': self.store_type,
            'total_documents': len(self.chunks),
            'dimension': self.dimension,
            'persist_path': str(self.persist_path)
        }