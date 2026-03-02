from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer
import logging

# Set up logging
logger = logging.getLogger(__name__)

# DocumentChunker: A class to split documents into overlapping chunks based on token counts.
class DocumentChunker:    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 400,
                 chunk_overlap: int = 50):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded")
    
    # Public method to chunk a list of documents
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        
        # Process each document and create chunks
        for doc_idx, doc in enumerate(documents):
            chunks = self._chunk_single_document(doc, doc_idx)
            all_chunks.extend(chunks)
            
            # Log progress every 100 documents
            if (doc_idx + 1) % 100 == 0:
                logger.info(f"Processed {doc_idx + 1}/{len(documents)} documents")
        
        # Final log with total chunks created
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    # Internal method to chunk a single document
    def _chunk_single_document(self, 
                               document: Dict[str, Any], 
                               doc_idx: int) -> List[Dict[str, Any]]:
        
        # Extract content and metadata
        content = document['content']
        metadata = document.get('metadata', {})
        
        # Tokenize content
        tokens = self.tokenizer.encode(
            content, 
            add_special_tokens=False,
            truncation=False
        )
        
        # If content fits in one chunk, don't split
        if len(tokens) <= self.chunk_size:
            return [{
                'content': content,
                'metadata': {
                    **metadata,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'doc_index': doc_idx,
                    'token_count': len(tokens)
                }
            }]
        
        # Create overlapping chunks
        chunks = []
        start_idx = 0
        chunk_idx = 0
        
        while start_idx < len(tokens):
            # Calculate end index
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Decode chunk back to text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Clean up partial words at boundaries (except for first/last chunk)
            if chunk_idx > 0 and start_idx > 0:
                # Remove incomplete word at start
                first_space = chunk_text.find(' ')
                if first_space > 0:
                    chunk_text = chunk_text[first_space:].strip()
            
            if end_idx < len(tokens):
                # Remove incomplete word at end
                last_space = chunk_text.rfind(' ')
                if last_space > 0:
                    chunk_text = chunk_text[:last_space].strip()
            
            # Create chunk
            chunk = {
                'content': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': chunk_idx,
                    'doc_index': doc_idx,
                    'token_count': len(chunk_tokens),
                    'char_start': start_idx,
                    'char_end': end_idx
                }
            }
            chunks.append(chunk)
            
            # Move start position with overlap
            start_idx = end_idx - self.chunk_overlap
            chunk_idx += 1
        
        # Update total_chunks in all chunks
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
    
    # Method to calculate statistics about chunks
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict[str, Any]:
    
        token_counts = []
        for chunk in chunks:
            tokens = self.tokenizer.encode(chunk['content'], add_special_tokens=False)
            token_counts.append(len(tokens))
        
        # Log statistics
        return {
            'total_chunks': len(chunks),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'median_tokens': sorted(token_counts)[len(token_counts) // 2]
        }