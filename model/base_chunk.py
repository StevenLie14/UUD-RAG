from pydantic import BaseModel

class BaseChunk(BaseModel):
    id: str

    def get_context(self) -> str:
        raise NotImplementedError
    
    def get_payload(self) -> dict:
        raise NotImplementedError


    
