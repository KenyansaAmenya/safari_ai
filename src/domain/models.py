from datetime import datetime
from pydantic import BaseModel, Field, validator
from typing import Any, Optional

# This class represents an ingested document(Raw document)
class document(BaseModel):
    id: Optional[str] = None
    source_name: str
    source_type: str # pdf, csv, docx, txt
    title: Optional[str] = None
    content: str
    location: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[datetime] = None

# splitting large documents to chunks
class Chunk(BaseModel):
    id: Optional[str] = None
    document_id: str
    text: str
    embedding: Optional[list[float]] = None
    chunk_index: int
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] = Field(default_factory=dict)   

# what the user asks the system
class Query(BaseModel):
    text: str
    top_k: int = 5
    filters: Optional[dict[str, Any]] = None

# represents relevant chunks found in the vectorDB
class Source(BaseModel):
    text: str
    metadata: dict[str, Any]
    score: float    

# Represents a complete response returned to the user
class Response(Basemodel):
    query: str
    answer: str
    sources: list[Source]
    latency_ms: float
    model_used:str = "grok"

# track individual test case results
class EvaluationResult(BaseModel):  
    question: str
    answer: str
    source_count: int
    latency_ms: float
    relevance_score: Optional[float] = None
    hallucination_flag: bool = False

# Aggregate statistics for system quality monitoring
class EvaluationReport(BaseModel):
    timestamp: datetime
    results: list[EvaluationResult]
    avg_latency_ms: float
    avg_sources: float
    hallucination_rate: float      

# Input Validation Model
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="User question about Kenya tourism"
    )   
    top_k: int = Field(default=5, ge=1, le=10)
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Metadata filters like {'location': 'maasai_mara'}"
    )
    
    # sanitize the query to prevent injection attacks and remove unwanted characters
    # prevents xss (Cross-site scripting) attacks by removing script tags and other HTML elements
    @validator('query')
    def sanitize_query(cls, v):
        import re
        v = re.sub(r'<script.*?>.*?</script>', '', v, flags=re.DOTALL | re.IGNORECASE)
        V = re.sub(r'<.*?>', '', v)
        return v.strip()

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    vector_count: int
    version: str = "1.0.0"      

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None          



