import os
from pathlib import Path 
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    # Paths
    DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    VECTOR_STORE_PATH: str = "data/vector_store"

    # Text processing
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50

    # Retriever settings
    TOP_K_RETRIEVAL: int = 5

    # Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB: str = "chroma"

    # Grok API
    GROK_API_KEY: Optional[str] = None
    GROK_MODEL: str = "grok-2-latest"
    GROK_API_BASE: str = "https://api.groq.com/openai/v1/"

    # Logging 
    LOG_LEVEL: str = "INFO"

    def __post_init__(self):
        # Load sensitive information from environment variables
        self.GROK_API_KEY = os.getenv("GROK_API_KEY", self.GROK_API_KEY)    
        self.GROK_MODEL = os.getenv("GROK_MODEL", self.GROK_MODEL)
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", self.CHUNK_SIZE))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", self.CHUNK_OVERLAP))
        self.TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", self.TOP_K_RETRIEVAL))
        self.VECTOR_DB = os.getenv("VECTOR_DB", self.VECTOR_DB)
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", self.EMBEDDING_MODEL)


    def validate(self) -> List[str]:
        errors = []

        if not self.GROK_API_KEY or self.GROK_API_KEY == "your_grok_api_key_here":
            errors.append("GROK_API_KEY is not set. Get your API key from https://api.groq.com/openai/v1/ and set it in the .env file.")

        if not os.path.exists(self.DATA_DIR):
            errors.append(f"DATA_DIR Data directory not found: {self.DATA_DIR}")

        else:
            csv_files = list(Path(self.DATA_DIR).glob("*.csv"))
            if not csv_files:
                errors.append(f"No CSV files found in DATA_DIR: {self.DATA_DIR}")

            return errors

    def is_valid(self) -> bool:
        """check if the configuration is valid by ensuring there are no validation errors."""
        return len(self.validate()) == 0                    
    

    def print_config(self):
        print("\n" + "="*40)
        print("Current Configuration:")
        print("="*40)
        for key, value in self.__dict__.items():
            if "KEY" in key and value:
                value = value[:10] + "..." if len(value) > 10 else "***"
            print(f"{key}: {value}")

        print("="*40 + "\n")


config = Config()                