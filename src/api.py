from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

from .rag_pipeline import TourismRAG, create_rag_system
from .config import config

# Initialize FastAPI app
app = FastAPI(
    title="Kenya Tourism RAG API",
    description="Retrieval-Augmented Generation API for Kenya travel information",
    version="1.0.0" # Just a placeholder version
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_system: Optional[TourismRAG] = None

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    filters: Optional[Dict[str, str]] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_count: int
    success: bool

class HealthResponse(BaseModel):
    status: str
    initialized: bool
    config: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    global rag_system
    try:
        rag_system = create_rag_system()
    except Exception as e:
        print(f"Warning: Could not auto-initialize: {e}")

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Kenya Tourism RAG API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy",
        initialized=rag_system is not None and rag_system.is_initialized,
        config={
            "chunk_size": config.CHUNK_SIZE,
            "embedding_model": config.EMBEDDING_MODEL,
            "vector_db": config.VECTOR_DB
        }
    )

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """Execute a RAG query."""
    if not rag_system or not rag_system.is_initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(
            question=request.question,
            top_k=request.top_k,
            filters=request.filters
        )
        
        return QueryResponse(
            query=result['query'],
            answer=result['answer'],
            sources=result.get('sources', []),
            retrieved_count=result.get('retrieved_count', 0),
            success=result.get('success', False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query", response_model=QueryResponse, tags=["RAG"])
async def query_get(
    question: str = Query(..., description="The question to ask"),
    top_k: int = Query(5, description="Number of documents to retrieve"),
    location: Optional[str] = Query(None, description="Filter by location")
):
    """GET endpoint for queries."""
    filters = {'location': location} if location else None
    return await query(QueryRequest(question=question, top_k=top_k, filters=filters))

@app.get("/stats", tags=["Admin"])
async def get_stats():
    """Get system statistics."""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_system.get_stats()

@app.post("/admin/init", tags=["Admin"])
async def initialize_system(force_rebuild: bool = False):
    """Initialize or rebuild the knowledge base."""
    global rag_system
    
    try:
        rag_system = TourismRAG()
        
        if not force_rebuild:
            try:
                rag_system.load_knowledge_base()
                return {"status": "loaded", "message": "Existing KB loaded"}
            except FileNotFoundError:
                pass
        
        count = rag_system.build_knowledge_base()
        return {
            "status": "built",
            "message": f"Knowledge base built with {count} chunks"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reload", tags=["Admin"])
async def reload_system():
    """Reload the knowledge base from disk."""
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        rag_system.load_knowledge_base()
        return {"status": "success", "message": "Knowledge base reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))