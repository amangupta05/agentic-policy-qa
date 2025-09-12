# services/gateway/config.py
from __future__ import annotations
from typing import ClassVar
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # env-driven settings
    VLLM_URL: str = "http://vllm:8000"
    VLLM_API_KEY: str = ""
    VLLM_MODEL: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    ENFORCE_ONE_SENTENCE: bool = True
    
    RETRIEVER_URL: str = "http://retriever:9000"
    RETRIEVER_TOP_K: int = 8

    RERANK_MIN_SCORE: float = 0.0
    RERANK_TOP_K_POST: int = 1

    REDIS_URL: str = "redis://redis:6379/0"
    RATE_BUCKET_CAPACITY: int = 5
    RATE_BUCKET_FILL_RATE: float = 1.0

    # constants (not fields)
    SYSTEM_PROMPT: ClassVar[str] = (
        "Answer ONLY from CONTEXT. If no relevant CONTEXT, reply exactly: Not found in docs.\n"
        "Output must be ≤1 sentence and end with bracketed citations like [1]. No preamble."
    )


    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

def get_settings() -> Settings:
    return Settings()
