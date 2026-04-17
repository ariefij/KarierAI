from __future__ import annotations

from typing import Any

from ..config import get_settings


def get_embeddings() -> Any:
    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception as exc:
        raise RuntimeError(f"langchain_openai belum tersedia: {exc}") from exc
    settings = get_settings()
    return OpenAIEmbeddings(model=settings.embedding_model, openai_api_key=settings.openai_api_key)


def get_qdrant_client() -> Any:
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:
        raise RuntimeError(f"qdrant_client belum tersedia: {exc}") from exc
    settings = get_settings()
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL belum diisi")
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def ensure_collection(vector_size: int = 1536) -> None:
    try:
        from qdrant_client.models import Distance, VectorParams
    except Exception as exc:
        raise RuntimeError(f"qdrant_client belum tersedia: {exc}") from exc
    settings = get_settings()
    client = get_qdrant_client()
    collections = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection_name not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def reset_vector_store_collection(vector_size: int = 1536) -> None:
    settings = get_settings()
    client = get_qdrant_client()
    try:
        client.delete_collection(settings.qdrant_collection_name)
    except Exception:
        pass
    ensure_collection(vector_size=vector_size)


def get_vector_store() -> Any:
    try:
        from langchain_qdrant import QdrantVectorStore
    except Exception as exc:
        raise RuntimeError(f"langchain_qdrant belum tersedia: {exc}") from exc
    settings = get_settings()
    ensure_collection()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_collection_name,
        embedding=get_embeddings(),
    )
