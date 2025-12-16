from pydantic import BaseModel, Field
from typing import List

class EvaluationItem(BaseModel):
    user_input: str = Field(..., description="The question or query from the user")
    retrieved_contexts: List[str] = Field(..., description="List of retrieved context strings")
    response: str = Field(..., description="The generated response/answer")
    reference: str = Field(..., description="The ground truth or reference answer")
