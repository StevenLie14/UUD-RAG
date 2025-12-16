from langchain_core.prompts.chat import ChatPromptTemplate
from typing import Dict
import time

class BaseLLM:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        if not api_key:
            raise ValueError("API key must be provided.")
        self.api_key = api_key
        self.model = self._initialize_llm()

    def answer(self, PROMPT: ChatPromptTemplate, input : Dict, max_retries: int = 3) -> str:
        runnable = PROMPT | self.model
        
        for attempt in range(max_retries):
            try:
                return runnable.invoke(input).content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[WARNING] LLM request failed (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                    print(f"[WARNING] Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] LLM request failed after {max_retries} attempts")
                    raise
        
    def _initialize_llm(self):
        raise NotImplementedError("Subclasses must implement this method.")