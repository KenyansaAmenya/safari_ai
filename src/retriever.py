from typing import List, Dict, Any, Optional
import numpy as np
import logging

from .vector_store import VectorStore
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

# Simple retriever with optional reranking
class Retriever:
    
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 5,
                 filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:

        # Encode query
        query_embedding = self.embedder.encode_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k, filters)
        
        logger.info(f"Retrieved {len(results)} documents for: {query[:50]}...")
        return results
    
    # 
    def retrieve_with_rerank(self, 
                            query: str,
                            initial_k: int = 10,
                            final_k: int = 5) -> List[Dict[str, Any]]:

        # Get more candidates initially
        candidates = self.retrieve(query, initial_k)
        
        if not candidates:
            return []
        
        # Simple reranking: boost scores based on keyword matches
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        # Rerank candidates
        for doc in candidates:
            score_boost = 0.0
            content_lower = doc.get('content', '').lower()
            meta = doc.get('metadata', {})
            
            # Boost for exact phrase match
            if query_lower in content_lower:
                score_boost += 0.3
            
            # Boost for individual term matches
            content_terms = set(content_lower.split())
            term_overlap = len(query_terms & content_terms) / len(query_terms)
            score_boost += term_overlap * 0.2
            
            # Boost for metadata matches
            for field in ['title', 'location', 'category', 'activities']:
                field_val = str(meta.get(field, '')).lower()
                if any(term in field_val for term in query_terms):
                    score_boost += 0.1
            
            # Apply boost
            original_score = doc.get('score', 0) or 0
            doc['score'] = original_score + score_boost
            doc['rerank_score'] = score_boost
        
        # Re-sort by boosted score
        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return candidates[:final_k]
    
    
    def hybrid_search(self,
                     query: str,
                     top_k: int = 5) -> List[Dict[str, Any]]:
 
        # For now, just use semantic search with reranking
        return self.retrieve_with_rerank(query, initial_k=top_k*2, final_k=top_k)
    
    # Format retrieved results into context string for LLM
    def get_context_string(self, 
                          results: List[Dict[str, Any]], 
                          max_length: int = 3000) -> str:

        contexts = []
        current_length = 0
        
        # Format each document with metadata and content
        for i, doc in enumerate(results, 1):
            meta = doc.get('metadata', {})
            title = meta.get('title', 'Unknown')
            content = doc.get('content', '')
            
            formatted = f"[{i}] {title}\n{content}\n\n"
            
            # Check if adding this document exceeds max length
            if current_length + len(formatted) > max_length:
                # Truncate last document if needed
                remaining = max_length - current_length
                if remaining > 100:
                    truncated = formatted[:remaining-50] + "... [truncated]\n\n"
                    contexts.append(truncated)
                break
            
            contexts.append(formatted)
            current_length += len(formatted)
        
        return "".join(contexts)