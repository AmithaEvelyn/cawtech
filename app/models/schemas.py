from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class Document(BaseModel):
    id: str
    filename: str
    content_type: str
    chunks: List[str]
    metadata: Dict[str, str]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Question(BaseModel):
    text: str
    context: Optional[str] = None
    conversation_id: Optional[str] = None

class Answer(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Dict]]
    conversation_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Feedback(BaseModel):
    answer_id: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Conversation(BaseModel):
    id: str
    questions: List[Question]
    answers: List[Answer]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow) 