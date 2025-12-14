"""
LLM Factory
Factory for creating LLM instances
"""

from config import Config
from llm import Gemini, ChatGPT, Ollama


class LLMFactory:
    """Factory for creating LLM instances"""
    
    @staticmethod
    def create_llm(llm_type: str, config: Config):
        """Create LLM instance"""
        if llm_type == "ollama":
            return Ollama("qwen3:8b", base_url="https://b84f92e0aabb.ngrok-free.app")
        elif llm_type == "chatgpt":
            return ChatGPT("gpt-4.1-mini", config.OPENAI_API_KEY)
        else:
            return Gemini("gemini-2.0-flash-lite", config.GOOGLE_API_KEY)
