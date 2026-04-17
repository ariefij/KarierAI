from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .agent import get_prompt, local_chat_response
from .config import get_settings
from .database import get_database_stats
from .ingestion import ingest_jobs
from .models import (
    CVAnalyzeRequest,
    CVAnalyzeResponse,
    ChatRequest,
    ChatResponse,
    ConsultationRequest,
    ConsultationResponse,
    IngestResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from .services import (
    build_career_consultation,
    build_recommendations,
    extract_cv_profile_data,
    extract_text_from_upload_bytes,
    is_ocr_ready,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ALLOWED_FILE_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
_ALLOWED_CONTENT_TYPES = {
    'application/pdf',
    'application/x-pdf',
    'image/png',
    'image/jpeg',
    'image/jpg',
    'image/webp',
    'image/bmp',
    'image/tiff',
}
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    logger.info('Starting KarierAI service')
    yield
    logger.info('Stopping KarierAI service')


app = FastAPI(
    title='KarierAI API',
    description='LLM-first career assistant API for Indonesian job dataset',
    lifespan=lifespan,
)


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/ready')
def ready() -> dict[str, object]:
    settings = get_settings()
    ocr_ready, ocr_reason = is_ocr_ready()
    return {
        'sqlite_path': str(settings.sqlite_file),
        'jobs_path_exists': settings.jobs_path.exists(),
        'qdrant_configured': bool(settings.qdrant_url),
        'openai_configured': bool(settings.openai_api_key),
        'langfuse_configured': bool(settings.langfuse_public_key and settings.langfuse_secret_key),
        'ocr_ready': ocr_ready,
        'ocr_reason': ocr_reason,
        'database_stats': get_database_stats(),
    }


@app.post('/ingest', response_model=IngestResponse)
def ingest(limit: int | None = None, replace_existing: bool = True) -> IngestResponse:
    try:
        return IngestResponse(**ingest_jobs(limit=limit, replace_existing=replace_existing))
    except Exception as exc:
        logger.exception('ingest failed')
        raise HTTPException(status_code=500, detail='Gagal menjalankan ingestion dataset.') from exc


@app.post('/chat', response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return ChatResponse(**local_chat_response(request.query, request.history))
    except Exception as exc:
        logger.exception('chat failed')
        raise HTTPException(status_code=500, detail='Gagal memproses chat.') from exc


def _validate_upload(file: UploadFile) -> None:
    filename = (file.filename or '').lower()
    content_type = (file.content_type or '').lower()
    has_allowed_extension = any(filename.endswith(ext) for ext in _ALLOWED_FILE_EXTENSIONS)
    has_allowed_content_type = content_type in _ALLOWED_CONTENT_TYPES
    if has_allowed_extension or has_allowed_content_type:
        return
    raise HTTPException(
        status_code=400,
        detail='Endpoint ini menerima CV dalam format PDF, PNG, JPG, JPEG, WEBP, BMP, atau TIFF.',
    )


async def _extract_cv_text_from_upload(file: UploadFile) -> str:
    _validate_upload(file)
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail='File CV kosong.')
    if len(raw_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='Ukuran file CV melebihi batas 8 MB.')
    try:
        return extract_text_from_upload_bytes(file.filename or '', file.content_type, raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('failed to extract cv text from upload')
        raise HTTPException(status_code=500, detail='Gagal memproses file CV yang diunggah.') from exc


@app.post('/cv/analyze', response_model=CVAnalyzeResponse)
def cv_analyze(request: CVAnalyzeRequest) -> CVAnalyzeResponse:
    return CVAnalyzeResponse(profile=extract_cv_profile_data(request.cv_text))


@app.post('/cv/analyze-file', response_model=CVAnalyzeResponse)
async def cv_analyze_file(file: UploadFile = File(...)) -> CVAnalyzeResponse:
    cv_text = await _extract_cv_text_from_upload(file)
    return CVAnalyzeResponse(profile=extract_cv_profile_data(cv_text))


@app.post('/recommend', response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    return RecommendationResponse(**build_recommendations(request.cv_text, top_k=request.top_k))


@app.post('/recommend-file', response_model=RecommendationResponse)
async def recommend_file(file: UploadFile = File(...), top_k: int = Form(5)) -> RecommendationResponse:
    cv_text = await _extract_cv_text_from_upload(file)
    return RecommendationResponse(**build_recommendations(cv_text, top_k=top_k))


@app.post('/consult', response_model=ConsultationResponse)
def consult(request: ConsultationRequest) -> ConsultationResponse:
    return ConsultationResponse(**build_career_consultation(request.cv_text, request.target_role))


@app.post('/consult-file', response_model=ConsultationResponse)
async def consult_file(file: UploadFile = File(...), target_role: str = Form(...)) -> ConsultationResponse:
    cv_text = await _extract_cv_text_from_upload(file)
    return ConsultationResponse(**build_career_consultation(cv_text, target_role))


@app.get('/prompts/{prompt_name}')
def prompt_preview(prompt_name: str) -> dict[str, str]:
    prompt = get_prompt(prompt_name)
    return {'name': prompt_name, 'prompt': prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)}
