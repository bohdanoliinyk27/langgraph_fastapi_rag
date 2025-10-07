# models.py
from typing import List, Literal, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

Role = Literal["user", "assistant"]

class Message(BaseModel):
    role: Role
    content: Any
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatCreate(BaseModel):
    title: Optional[str] = None

class ChatSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages_count: int

class Chat(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    history: List[Message] = Field(default_factory=list)


class ChatAskRequestV2(BaseModel):
    question: str
    attached: List[str] = Field(default_factory=list)
    use_web: bool = False
    use_pubmed: bool = False
    template_id: Optional[str] = None

class ChatAnswer(BaseModel):
    answer: str
    trace: Optional[list] = None

class FileInfo(BaseModel):
    name: str
    size: int

class UploadResult(BaseModel):
    message: str
    namespace: str
