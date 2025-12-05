from google import genai
from .base import BaseLLM
from langchain_core.prompts.chat import ChatPromptTemplate
from typing import Dict


class GeminiLive(BaseLLM):
    """Gemini Live API implementation using google-genai package"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp", api_key: str = None):
        """
        Initialize Gemini Live client
        
        Args:
            model_name: Name of the Gemini model (default: gemini-2.0-flash-exp)
            api_key: Google API key
        """
        if not api_key:
            raise ValueError("API key must be provided for Gemini Live.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.api_key = api_key
        self.model = None  # Not needed for this implementation
    
    def _initialize_llm(self):
        """Not needed for Gemini Live client - we use self.client directly"""
        return None
    
    def answer(self, PROMPT: ChatPromptTemplate, input: Dict) -> str:
        """
        Generate answer using Gemini Live API
        
        Args:
            PROMPT: ChatPromptTemplate with system and user messages
            input: Dictionary of input variables for the prompt
            
        Returns:
            str: Generated response content
        """
        try:
            # Format the prompt with input variables
            messages = PROMPT.format_messages(**input)
            
            # Extract system and user messages
            system_instruction = None
            user_content = ""
            
            for msg in messages:
                if msg.type == "system":
                    system_instruction = msg.content
                elif msg.type == "human" or msg.type == "user":
                    user_content = msg.content
            
            # Generate response using Gemini Live client
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config={
                    "system_instruction": system_instruction,
                    "temperature": 0.1,
                    "max_output_tokens": 1000
                }
            )
            
            return response.text
            
        except Exception as e:
            raise RuntimeError(f"Error generating response with Gemini Live: {e}")
    
    def get_ragas_llm(self):
        """
        Get RAGAS-compatible LLM wrapper
        Note: This uses LangChain's implementation since RAGAS requires it
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper
        
        langchain_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",  # Use stable model for RAGAS
            api_key=self.api_key,
            temperature=0.1,
            max_tokens=1000
        )
        
        return LangchainLLMWrapper(langchain_llm)
