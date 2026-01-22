"""
LLaMA4-Ollama Integration - January Phase
Configures LLaMA4 with Ollama framework for offline operation
"""

import time
from typing import Optional, Dict, List, Any
import json

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama Python client not available. Install with: pip install ollama")

try:
    from langchain_community.llms import Ollama
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_OLLAMA_AVAILABLE = False
    print("⚠️  LangChain Ollama integration not available.")


class OllamaLLM:
    """
    Wrapper for Ollama LLM integration with LLaMA4 support.
    """
    
    def __init__(self, model_name: str = "llama3.2", 
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.2,
                 max_tokens: Optional[int] = None):
        """
        Initialize Ollama LLM.
        
        Args:
            model_name: Name of the model (e.g., "llama3.2", "llama4" when available)
            base_url: Ollama API base URL
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (None for model default)
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama not available. Install with: pip install ollama")
        
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Test connection
        self._check_connection()
        
        print(f"   ✅ Ollama LLM initialized: {model_name}")
        print(f"   Base URL: {base_url}")
    
    def _check_connection(self) -> None:
        """Check if Ollama is running and model is available."""
        try:
            # List available models
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            if self.model_name not in model_names:
                print(f"   ⚠️  Model '{self.model_name}' not found in Ollama.")
                print(f"   Available models: {', '.join(model_names)}")
                print(f"   Pull model with: ollama pull {self.model_name}")
            else:
                print(f"   ✅ Model '{self.model_name}' is available")
        except Exception as e:
            print(f"   ⚠️  Could not connect to Ollama at {self.base_url}")
            print(f"   Error: {e}")
            print(f"   Make sure Ollama is running: ollama serve")
    
    def generate(self, prompt: str, 
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                stream: bool = False) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stream: Whether to stream the response
            
        Returns:
            Generated text
        """
        params = {
            'model': self.model_name,
            'prompt': prompt,
            'options': {
                'temperature': temperature if temperature is not None else self.temperature,
            }
        }
        
        if max_tokens is not None or self.max_tokens is not None:
            params['options']['num_predict'] = max_tokens or self.max_tokens
        
        if stream:
            # Stream response
            response_text = ""
            for chunk in ollama.generate(**params, stream=True):
                if 'response' in chunk:
                    response_text += chunk['response']
            return response_text
        else:
            # Get full response
            response = ollama.generate(**params)
            return response.get('response', '')
    
    def chat(self, messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            stream: bool = False) -> str:
        """
        Chat completion interface.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            stream: Whether to stream the response
            
        Returns:
            Assistant's response
        """
        params = {
            'model': self.model_name,
            'messages': messages,
            'options': {
                'temperature': temperature if temperature is not None else self.temperature,
            }
        }
        
        if self.max_tokens:
            params['options']['num_predict'] = self.max_tokens
        
        if stream:
            response_text = ""
            for chunk in ollama.chat(**params, stream=True):
                if 'message' in chunk and 'content' in chunk['message']:
                    response_text += chunk['message']['content']
            return response_text
        else:
            response = ollama.chat(**params)
            return response['message']['content']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            models = ollama.list()
            for model in models.get('models', []):
                if model['name'] == self.model_name:
                    return model
            return {}
        except:
            return {}


class LangChainOllamaLLM:
    """
    LangChain wrapper for Ollama (alternative interface).
    """
    
    def __init__(self, model_name: str = "llama3.2",
                 temperature: float = 0.2,
                 streaming: bool = False):
        """
        Initialize LangChain Ollama LLM.
        
        Args:
            model_name: Name of the model
            temperature: Sampling temperature
            streaming: Whether to enable streaming
        """
        if not LANGCHAIN_OLLAMA_AVAILABLE:
            raise ImportError("LangChain Ollama not available")
        
        callback_manager = None
        if streaming:
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            callback_manager=callback_manager
        )
        
        self.model_name = model_name
        self.temperature = temperature
        
        print(f"   ✅ LangChain Ollama LLM initialized: {model_name}")
    
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        return self.llm(prompt)
    
    def __call__(self, prompt: str) -> str:
        """Make the LLM callable."""
        return self.generate(prompt)


def test_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """
    Test if Ollama is running and accessible.
    
    Args:
        base_url: Ollama API base URL
        
    Returns:
        True if connection successful
    """
    if not OLLAMA_AVAILABLE:
        return False
    
    try:
        models = ollama.list()
        return True
    except:
        return False


def list_available_models(base_url: str = "http://localhost:11434") -> List[str]:
    """
    List available Ollama models.
    
    Args:
        base_url: Ollama API base URL
        
    Returns:
        List of model names
    """
    if not OLLAMA_AVAILABLE:
        return []
    
    try:
        models = ollama.list()
        return [m['name'] for m in models.get('models', [])]
    except:
        return []

