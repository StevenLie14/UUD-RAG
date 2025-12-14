from langchain_openai import ChatOpenAI
from .base import BaseLLM
from ragas.llms import LangchainLLMWrapper

class ChatGPT(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        
    def _initialize_llm(self):
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
        )
    
    def get_ragas_llm(self):
        return LangchainLLMWrapper(ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
        ))
