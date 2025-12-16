from langchain_ollama import ChatOllama
from .base import BaseLLM
from ragas.llms import LangchainLLMWrapper


class Ollama(BaseLLM):
    def __init__(self, model_name: str, base_url: str = "https://b84f92e0aabb.ngrok-free.app"):
        self.base_url = base_url
        super().__init__(model_name, api_key="ollama-local")
        
    def _initialize_llm(self):
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.1,
            num_predict=1000,
        )
    
    def get_ragas_llm(self):
        return LangchainLLMWrapper(ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.1,
            num_predict=1000,
        ))
