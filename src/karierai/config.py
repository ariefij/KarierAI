from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    llm_model: str = os.getenv('LLM_MODEL', 'gpt-4.1-mini')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    langfuse_public_key: str = os.getenv('LANGFUSE_PUBLIC_KEY', '')
    langfuse_secret_key: str = os.getenv('LANGFUSE_SECRET_KEY', '')
    langfuse_host: str = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
    qdrant_url: str = os.getenv('QDRANT_URL', '')
    qdrant_api_key: str = os.getenv('QDRANT_API_KEY', '')
    qdrant_collection_name: str = os.getenv('QDRANT_COLLECTION_NAME', 'jobs_collection')
    sqlite_path: str = os.getenv('SQLITE_PATH', 'dataset/jobs.db')
    chunk_size: int = int(os.getenv('CHUNK_SIZE', '800'))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', '120'))
    top_k_default: int = int(os.getenv('TOP_K_DEFAULT', '5'))
    api_host: str = os.getenv('API_HOST', '0.0.0.0')
    api_port: int = int(os.getenv('API_PORT', os.getenv('PORT', '8080')))
    api_url: str = os.getenv('API_URL', 'http://localhost:8080')
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')

    @property
    def sqlite_file(self) -> Path:
        path = Path(self.sqlite_path)
        return path if path.is_absolute() else self.project_root / path

    @property
    def jobs_path(self) -> Path:
        return self.project_root / 'dataset' / 'jobs.jsonl'


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
