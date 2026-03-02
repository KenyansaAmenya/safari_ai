from typing import List, Dict, Any, Optional
import logging

from .config import config
from .data_loader import TourismDataLoader
from .chunker import DocumentChunker
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .grok_client import GrokClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Main RAG Pipeline
# Orchestrates data loading, chunking, embedding, retrieval, and generation.
class TourismRAG:

    # Initialize all components to None. They will be set during build or load.
    def __init__(self):
        self.config = config
        self.data_loader: Optional[TourismDataLoader] = None
        self.chunker: Optional[DocumentChunker] = None
        self.embedder: Optional[EmbeddingGenerator] = None
        self.vector_store: Optional[VectorStore] = None
        self.grok: Optional[GrokClient] = None
        
        self.is_initialized = False

    # Build the knowledge base from scratch    
    def build_knowledge_base(self, force_rebuild: bool = False) -> int:
       
        logger.info(" Building knowledge base...")
        
        # 1. Load data
        logger.info("Step 1/4: Loading data...")
        self.data_loader = TourismDataLoader(self.config.DATA_DIR)
        documents = self.data_loader.load_all()
        
        if not documents:
            raise ValueError("No documents loaded. Check your CSV files.")
        
        stats = self.data_loader.get_statistics()
        logger.info(f"Loaded {stats['total_documents']} documents")
        logger.info(f"Categories: {stats['categories']}")
        
        # 2. Chunk documents
        logger.info("Step 2/4: Chunking documents...")
        self.chunker = DocumentChunker(
            self.config.EMBEDDING_MODEL,
            self.config.CHUNK_SIZE,
            self.config.CHUNK_OVERLAP
        )
        chunks = self.chunker.chunk_documents(documents)
        
        chunk_stats = self.chunker.get_chunk_statistics(chunks)
        logger.info(f"Chunk statistics: {chunk_stats}")
        
        # 3. Generate embeddings
        logger.info("Step 3/4: Generating embeddings...")
        self.embedder = EmbeddingGenerator(self.config.EMBEDDING_MODEL)
        embeddings = self.embedder.generate_for_chunks(chunks)
        
        # 4. Store in vector DB
        logger.info("Step 4/4: Storing in vector database...")
        self.vector_store = VectorStore(
            self.config.VECTOR_DB,
            self.config.VECTOR_STORE_PATH
        )
        self.vector_store.initialize(self.embedder.dimension)
        self.vector_store.add_documents(chunks, embeddings)
        self.vector_store.persist()
        
        # Initialize Grok client
        self.grok = GrokClient(self.config.GROK_API_KEY, self.config.GROK_MODEL)
        
        self.is_initialized = True
        logger.info("Knowledge base built successfully!")
        
        return len(chunks)
    
    # Load pre-built knowledge base.
    def load_knowledge_base(self):
        logger.info("Loading existing knowledge base...")
        
        # Load vector store
        self.embedder = EmbeddingGenerator(self.config.EMBEDDING_MODEL)
        self.vector_store = VectorStore(
            self.config.VECTOR_DB,
            self.config.VECTOR_STORE_PATH
        )
        self.vector_store.load()
        self.grok = GrokClient(self.config.GROK_API_KEY, self.config.GROK_MODEL)
        
        # 
        self.is_initialized = True
        logger.info(" Knowledge base loaded")
    
    # Main query function
    def query(self, 
              question: str,
              top_k: int = None,
              filters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:

        # Check initialization
        if not self.is_initialized:
            raise RuntimeError("Knowledge base not initialized. Run build_knowledge_base() or load_knowledge_base() first.")
        
        top_k = top_k or self.config.TOP_K_RETRIEVAL
        
        logger.info(f"Processing query: {question[:60]}...")
        
        # 1. Retrieve relevant documents
        query_embedding = self.embedder.encode_query(question)
        retrieved_docs = self.vector_store.search(query_embedding, top_k, filters)
        
        # If no documents are retrieved, return a default response
        if not retrieved_docs:
            return {
                'query': question,
                'answer': "I couldn't find any relevant information about that in my knowledge base.",
                'sources': [],
                'retrieved_count': 0
            }
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # 2. Generate response
        response = self.grok.generate(question, retrieved_docs)
        
        return {
            'query': question,
            'answer': response['answer'],
            'sources': response.get('sources', []),
            'retrieved_documents': retrieved_docs,
            'retrieved_count': len(retrieved_docs),
            'model': response.get('model'),
            'success': response.get('success', False)
        }
    
    # Simple chat interface that can be extended to maintain conversation history.
    def chat(self, message: str, history: List[Dict] = None) -> str:

        result = self.query(message)
        return result['answer']
    
    # Get system statistics for monitoring and debugging.
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            'initialized': self.is_initialized,
            'config': {
                'chunk_size': self.config.CHUNK_SIZE,
                'embedding_model': self.config.EMBEDDING_MODEL,
                'vector_db': self.config.VECTOR_DB
            }
        }
        
        # Add vector store stats
        if self.vector_store:
            stats['vector_store'] = self.vector_store.get_stats()
        
        # Add data loader stats
        if self.data_loader and self.data_loader.documents:
            stats['data'] = self.data_loader.get_statistics()
        
        return stats

# Helper function to create and optionally build the RAG system.
def create_rag_system(build: bool = True) -> TourismRAG:
    rag = TourismRAG()
    
    if build:
        try:
            rag.load_knowledge_base()
        except FileNotFoundError:
            logger.info("No existing KB found, building new one...")
            rag.build_knowledge_base()
    
    return rag