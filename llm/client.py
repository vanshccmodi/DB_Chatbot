"""
LLM Client - Unified interface for Groq, OpenAI, and local models.

Groq is the DEFAULT provider (free tier available).
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class GroqClient(LLMClient):
    """
    Groq API client - FREE and FAST inference.
    
    Available models:
    - llama-3.3-70b-versatile (recommended)
    - llama-3.1-8b-instant (faster)
    - mixtral-8x7b-32768
    - gemma2-9b-it
    """
    
    AVAILABLE_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile", 
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client
    
    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0
        )
    
    def is_available(self) -> bool:
        try:
            # Simple test call
            self.client.models.list()
            return True
        except Exception as e:
            logger.warning(f"Groq availability check failed: {e}")
            return False


class OpenAIClient(LLMClient):
    """OpenAI API client (paid)."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0
        )
    
    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False


class LocalLLaMAClient(LLMClient):
    """Local LLaMA/Phi model client via transformers."""
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._pipeline = None
    
    @property
    def pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            logger.info(f"Loading local model: {self.model_name}")
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        return self._pipeline
    
    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        output = self.pipeline(
            messages,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True
        )
        generated_text = output[0]["generated_text"][-1]["content"]
        # Approximate tokens for local (or use tokenizer if available)
        return LLMResponse(
            content=generated_text,
            input_tokens=0, # Local pipeline generic usually doesn't give this easily without more access
            output_tokens=0,
            total_tokens=0
        )
    
    def is_available(self) -> bool:
        try:
            _ = self.pipeline
            return True
        except Exception:
            return False
            
def create_llm_client(provider: str = "groq", **kwargs) -> LLMClient:
    """
    Factory function to create LLM client.
    
    Args:
        provider: "groq" (default, free), "openai", or "local"
        **kwargs: Provider-specific arguments
        
    Returns:
        Configured LLMClient instance
    """
    if provider == "groq":
        return GroqClient(**kwargs)
    elif provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "local":
        return LocalLLaMAClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'groq', 'openai', or 'local'")
