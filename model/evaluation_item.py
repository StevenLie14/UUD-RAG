from pydantic import BaseModel
from typing import List

class EvaluationItem(BaseModel):
    user_input: str
    retrieved_contexts: List[str]
    response: str
    reference: str
