# Base Exception
class SafariAIError(Exception):
    pass

# failed to load error
class DocumentLoadError(SafariAIError):
    pass

# failed to generate embedding
class EmbeddingError(SafariAIError):
    pass

# supabase vector operation failed
class VectorStoreErrror(SafariAIError):
    pass

# GROK API error
class LLMError(SafariAIError):
    pass

# Search operation failed
class RetrievalError(SafariAIError):
    pass