from dotenv import load_dotenv
import os

class Config:
    def __init__(self):
        load_dotenv()
        self.GOOGLE_API_KEY = self.set_env("GOOGLE_API_KEY")
        self.GROQ_API_KEY = self.set_env("GROQ_API_KEY")
        self.QDRANT_API_KEY = self.set_env("QDRANT_API_KEY",False)
        self.QDRANT_HOST = self.set_env("QDRANT_HOST")
        self.OPENAI_API_KEY = self.set_env("OPENAI_API_KEY")
        self.HF_TOKEN = self.set_env("HF_TOKEN", False)
        
    
    def set_env(self, key, required: bool = True) -> str:
        env = os.getenv(key)
        if env is None and required:
            raise ValueError(f"Environment variable {key} is not set.")
        return env
    