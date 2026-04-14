from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Supabase
    supabase_url: str
    supabase_service_key: str
    supabase_db_connection: str

    # Vector Store
    vector_collection: str = "safari_chunks"
    embedding_dimension: int = 384

    # GROK
    grok_api_key = str
    grok_api_url = str = "https://api.x.ai/v1/chat/completions"
    grok_model = "grok-3.5-mini"

    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 50

    # Retrieval 
    top_k: int = 5
    score_threshold: float = 0.5

    # App
    log_level: str = "INFO"
    
    # security
    api_key: Optional[str] = None  # Set to require API key authentication (to change for production)
    rate_limit_per_minute: int = 30
    max_query_length: int = 500  
    allowed_origins: str = "*"  # CORS origins (to add origins for production)

    class Config:
        env_file = ".env"

    settings = Settings()    