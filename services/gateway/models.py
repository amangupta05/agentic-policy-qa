from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, conlist


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    messages: Optional[List[ChatMessage]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=32, le=2048)
    stream: bool = Field(default=True)
    api_key: Optional[str] = None

class Citation(BaseModel):
    doc_id: str
    page: Optional[int]
    span: str
    score: Optional[float]

class ChatResponse(BaseModel):
    request_id: str
    query: str
    context_count: int
    output: str
    latency_ms: float
    citations: List[Citation] = []


class Health(BaseModel):
    status: str


class Ready(BaseModel):
    status: str
    details: dict
