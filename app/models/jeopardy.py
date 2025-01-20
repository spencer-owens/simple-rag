from pydantic import BaseModel
from typing import List, Optional

class JeopardySource(BaseModel):
    """A source from the Jeopardy dataset used to answer the question."""
    content: str
    metadata: dict

class JeopardyRequest(BaseModel):
    """A request for a Jeopardy-based question."""
    question: str
    user_id: Optional[str] = None

class JeopardyResponse(BaseModel):
    """A response to a Jeopardy-based question."""
    answer: str
    sources: List[JeopardySource]
    cached: bool = False 