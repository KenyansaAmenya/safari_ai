import os
import requests
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GrokClient:
    # my (fallbacks)
    WORKING_MODELS = [
        "qwen/qwen3-32b",
        "llama-3.3-70b-versatile", 
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "groq/compound-mini"
    ]
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,  # Made optional for auto-detection
                 api_base: str = "https://api.groq.com/openai/v1"):
       
        self.api_key = api_key or os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY or GROK_API_KEY environment variable."
            )
        
        self.api_base = api_base
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Set model with priority:
        # 1. Explicitly passed model
        # 2. Environment variable
        # 3. Auto-detection from API
        # 4. Default fallback
        self.model = model or os.getenv("GROQ_MODEL") or self._get_working_model()
        
        logger.info(f"Groq client initialized (model: {self.model})")
    
    def _get_working_model(self) -> str:
       
        try:
            response = requests.get(
                f"{self.api_base}/models",
                headers=self.headers,
                timeout=5
            )
            
            if response.status_code == 200:
                available_models = [m['id'] for m in response.json().get('data', [])]
                logger.info(f"Available models from API: {available_models}")
                
                # Try preferred models first
                for preferred in self.WORKING_MODELS:
                    if preferred in available_models:
                        logger.info(f"Selected preferred model: {preferred}")
                        return preferred
                
                # Fallback to first available text model (excluding special-purpose ones)
                exclude_keywords = ["whisper", "guard", "canopylabs/orpheus", "allam"]
                for model in available_models:
                    if not any(keyword in model.lower() for keyword in exclude_keywords):
                        logger.info(f"Selected fallback text model: {model}")
                        return model
                
                # Last resort: first available model
                if available_models:
                    logger.info(f"Selected first available model: {available_models[0]}")
                    return available_models[0]
                    
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to Groq API for model detection: {e}")
        except Exception as e:
            logger.warning(f"Error during model auto-detection: {e}")
        
        # Ultimate fallback to first working model
        logger.info(f"Using default fallback model: {self.WORKING_MODELS[0]}")
        return self.WORKING_MODELS[0]
    
    
    def generate(self,
                 query: str,
                 context: List[Dict[str, Any]],
                 temperature: float = 0.3,
                 max_tokens: int = 1024) -> Dict[str, Any]:
        
        system_prompt = """You are a knowledgeable Kenya tourism assistant. Your answers must be strictly based on the provided context documents.

CRITICAL RULES:
1. Answer ONLY using the information in the provided context
2. If the context doesn't contain the answer, say: "I don't have specific information about that in my knowledge base."
3. Never make up information or use external knowledge
4. Always cite your sources using [Source: Title] format
5. Be concise but comprehensive
6. If multiple sources provide conflicting information, acknowledge the discrepancy
7. For locations, include relevant details like accessibility, best time to visit, and entrance fees if available in context"""

        context_str = self._format_context(context)
        
        user_message = f"""Based on the following context documents, please answer the question.

{context_str}

Question: {query}

Provide a helpful answer based strictly on the context above. Include source citations using [Source: Title] format. If the information comes from multiple sources, cite all relevant ones."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            logger.info(f"Sending request to Groq API with model: {self.model}")
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                },
                timeout=60
            )
            
            # Log the request for debugging
            logger.debug(f"Request payload: {{'model': '{self.model}', 'temperature': {temperature}, 'max_tokens': {max_tokens}}}")
            
            response.raise_for_status()
            result = response.json()
            
            answer = result['choices'][0]['message']['content']
            
            # Extract sources with metadata
            sources = []
            for doc in context:
                meta = doc.get('metadata', {})
                sources.append({
                    'title': meta.get('title', 'Unknown'),
                    'location': meta.get('location'),
                    'category': meta.get('category'),
                    'score': doc.get('score'),
                    'content_preview': doc.get('content', '')[:200] + '...' if doc.get('content') else ''
                })
            
            return {
                'answer': answer,
                'model': self.model,
                'sources': sources,
                'usage': result.get('usage', {}),
                'success': True
            }
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error {response.status_code}: {e}")
            error_detail = ""
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            
            # Handle specific HTTP errors
            if response.status_code == 400:
                # Model might be decommissioned - try to get working model
                logger.warning("Model may be decommissioned, attempting to find working model...")
                new_model = self._get_working_model()
                if new_model != self.model:
                    logger.info(f"Retrying with new model: {new_model}")
                    self.model = new_model
                    # Recursive retry with new model
                    return self.generate(query, context, temperature, max_tokens)
                else:
                    return {
                        'answer': f"Error: The model '{self.model}' is not available. Please check your model configuration.",
                        'success': False,
                        'error': 'model_unavailable',
                        'details': str(error_detail)
                    }
            elif response.status_code == 401:
                return {
                    'answer': "Error: Invalid API key. Please check your GROQ_API_KEY.",
                    'success': False,
                    'error': 'authentication_failed'
                }
            elif response.status_code == 429:
                return {
                    'answer': "Error: Rate limit exceeded. Please try again later.",
                    'success': False,
                    'error': 'rate_limited'
                }
            else:
                return {
                    'answer': f"Error calling Groq API: {str(e)}",
                    'success': False,
                    'error': 'api_error',
                    'details': str(error_detail)
                }
                
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            return {
                'answer': "Error: Request timed out. Please try again.",
                'success': False,
                'error': 'timeout'
            }
        except requests.exceptions.ConnectionError:
            logger.error("Connection error")
            return {
                'answer': "Error: Could not connect to Groq API. Please check your network.",
                'success': False,
                'error': 'connection_error'
            }
        except Exception as e:
            logger.error(f"Unexpected error generating response: {e}")
            return {
                'answer': f"Error generating response: {str(e)}",
                'success': False,
                'error': 'unknown'
            }
    
    def _format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents for the prompt with better structure."""
        if not documents:
            return "No relevant documents found."
        
        parts = []
        for i, doc in enumerate(documents, 1):
            meta = doc.get('metadata', {})
            title = meta.get('title', 'Untitled')
            location = meta.get('location', 'Unknown location')
            category = meta.get('category', '')
            
            # Create header with metadata
            header = f"[SOURCE {i}] {title}"
            if location and location.lower() != 'unknown':
                header += f" - Location: {location}"
            if category:
                header += f" - Category: {category}"
            
            # Get content and truncate if too long
            content = doc.get('content', '').strip()
            if len(content) > 2000:  # Increased from 1500 for better context
                content = content[:2000] + "... [content truncated]"
            
            parts.append(f"{header}\n{content}\n")
        
        return "\n---\n".join(parts)
    
    def test_connection(self) -> bool:
        """Test API connectivity and authentication."""
        try:
            response = requests.get(
                f"{self.api_base}/models",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                models = response.json().get('data', [])
                logger.info(f"Connection successful. Found {len(models)} models.")
                return True
            else:
                logger.error(f"Connection test failed with status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """Get list of all available models from Groq API."""
        try:
            response = requests.get(
                f"{self.api_base}/models",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                models = [m['id'] for m in response.json().get('data', [])]
                return models
            else:
                logger.error(f"Failed to list models: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []