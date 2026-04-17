from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_MULTI_WS_RE = re.compile(r'\s+')


def _normalize_text(value: str) -> str:
    return _MULTI_WS_RE.sub(' ', value or '').strip()


class ChatRequest(BaseModel):
    query: str = Field(..., description='Pertanyaan user')
    history: str = Field(default='', description='Ringkasan chat sebelumnya')

    @field_validator('query', 'history', mode='before')
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        if value is None:
            return ''
        return str(value)

    @field_validator('query')
    @classmethod
    def _validate_query(cls, value: str) -> str:
        normalized = _normalize_text(value)
        if len(normalized) < 2:
            raise ValueError('Query chat tidak boleh kosong.')
        if len(normalized) > 2000:
            raise ValueError('Query chat terlalu panjang. Ringkas maksimal 2000 karakter.')
        return normalized

    @field_validator('history')
    @classmethod
    def _validate_history(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) > 12000:
            raise ValueError('History chat terlalu panjang. Ringkas maksimal 12000 karakter.')
        return normalized


class ChatResponse(BaseModel):
    response: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_messages: list[str] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)

    @field_validator('response')
    @classmethod
    def _validate_response(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError('Respons chat tidak boleh kosong.')
        return normalized


class IngestResponse(BaseModel):
    jobs_inserted: int
    chunks_inserted: int
    collection_name: str
    raw_rows_loaded: int | None = None
    replace_existing: bool | None = None
    vector_reset: bool | None = None


class CVAnalyzeRequest(BaseModel):
    cv_text: str

    @field_validator('cv_text')
    @classmethod
    def _validate_cv_text(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 10:
            raise ValueError('Teks CV terlalu pendek untuk dianalisis.')
        return normalized


class CVAnalyzeResponse(BaseModel):
    profile: dict[str, Any]
    response_text: str | None = None


class RecommendationMatch(BaseModel):
    job_id: str | None = None
    job_title: str | None = None
    company_name: str | None = None
    location: str | None = None
    work_type: str | None = None
    salary_raw: str | None = None
    score: float = 0.0
    matched_skills: list[str] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
    job_excerpt: str | None = None


class RecommendationRequest(BaseModel):
    cv_text: str
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator('cv_text')
    @classmethod
    def _validate_cv_text(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 10:
            raise ValueError('Teks CV terlalu pendek untuk rekomendasi.')
        return normalized


class RecommendationResponse(BaseModel):
    profile: dict[str, Any]
    search_query: str
    matches: list[RecommendationMatch] = Field(default_factory=list)
    response_text: str | None = None


class ConsultationRequest(BaseModel):
    cv_text: str
    target_role: str

    @field_validator('cv_text')
    @classmethod
    def _validate_cv_text(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 10:
            raise ValueError('Teks CV terlalu pendek untuk konsultasi.')
        return normalized

    @field_validator('target_role')
    @classmethod
    def _validate_target_role(cls, value: str) -> str:
        normalized = _normalize_text(value)
        if len(normalized) < 2:
            raise ValueError('Target role wajib diisi.')
        if len(normalized) > 120:
            raise ValueError('Target role terlalu panjang.')
        return normalized


class ConsultationResponse(BaseModel):
    target_role: str
    profile: dict[str, Any]
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    market_summary: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    response_text: str | None = None


class RouteTaskInput(BaseModel):
    query: str


class RAGSearchInput(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)


class SQLQuestionInput(BaseModel):
    question: str


class CVTextInput(BaseModel):
    cv_text: str


class SkillGapInput(BaseModel):
    cv_text: str
    target_role: str
