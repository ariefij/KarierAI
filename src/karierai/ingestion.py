from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_settings
from .database import (
    clear_runtime_tables,
    fetch_staged_raw_jobs,
    get_vector_store,
    init_sqlite,
    insert_chunks,
    insert_jobs,
    mark_staged_jobs_loaded,
    reset_vector_store_collection,
    set_metadata,
    stage_raw_jobs,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == 'none':
        return None
    return ' '.join(text.split())


def _build_job_id(row: dict[str, Any]) -> str:
    seed = '|'.join(
        [
            _clean_text(row.get('job_title')) or '',
            _clean_text(row.get('company_name')) or '',
            _clean_text(row.get('location')) or '',
            _clean_text(row.get('_scrape_timestamp')) or '',
        ]
    )
    return hashlib.md5(seed.encode('utf-8')).hexdigest()


def _build_raw_id(row: dict[str, Any], *, index: int) -> str:
    source = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(f'{index}|{source}'.encode('utf-8')).hexdigest()


def normalize_job(row: dict[str, Any], source_file: str = 'jobs.jsonl') -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        'job_id': _build_job_id(row),
        'job_title': _clean_text(row.get('job_title')) or 'Untitled Job',
        'company_name': _clean_text(row.get('company_name')),
        'location': _clean_text(row.get('location')),
        'work_type': _clean_text(row.get('work_type')),
        'salary_raw': _clean_text(row.get('salary')),
        'job_description': _clean_text(row.get('job_description')) or '',
        'scrape_timestamp': _clean_text(row.get('_scrape_timestamp')),
        'source_file': source_file,
        'created_at': now,
    }


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = ' '.join(text.split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def build_chunk_rows(job: dict[str, Any]) -> list[dict[str, Any]]:
    settings = get_settings()
    header = (
        f"Job Title: {job.get('job_title') or '-'}\n"
        f"Company: {job.get('company_name') or '-'}\n"
        f"Location: {job.get('location') or '-'}\n"
        f"Work Type: {job.get('work_type') or '-'}\n"
        f"Salary: {job.get('salary_raw') or '-'}\n\nDescription:\n"
    )
    description_chunks = _chunk_text(job.get('job_description', ''), settings.chunk_size, settings.chunk_overlap)
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for idx, chunk in enumerate(description_chunks):
        chunk_text = header + chunk
        rows.append(
            {
                'chunk_id': f"{job['job_id']}-{idx}",
                'job_id': job['job_id'],
                'chunk_index': idx,
                'chunk_text': chunk_text,
                'char_count': len(chunk_text),
                'token_estimate': max(1, len(chunk_text) // 4),
                'created_at': now,
            }
        )
    return rows


def _extract_to_staging(raw_rows: list[dict[str, Any]], *, source_file: str, replace_existing: bool) -> int:
    staged_payload = [
        {
            'raw_id': _build_raw_id(row, index=index),
            'raw_json': row,
            'source_file': source_file,
            'status': 'staged',
        }
        for index, row in enumerate(raw_rows)
    ]
    return stage_raw_jobs(staged_payload, replace_existing=replace_existing)


def _transform_and_load_from_staging(*, source_file: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    staged_rows = fetch_staged_raw_jobs(status='staged') or fetch_staged_raw_jobs()
    jobs: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    loaded_raw_ids: list[str] = []
    seen_job_ids: set[str] = set()

    for staged in staged_rows:
        payload = json.loads(staged['raw_json'])
        job = normalize_job(payload, source_file=source_file)
        if job['job_id'] in seen_job_ids:
            loaded_raw_ids.append(staged['raw_id'])
            continue
        seen_job_ids.add(job['job_id'])
        jobs.append(job)
        chunk_rows.extend(build_chunk_rows(job))
        loaded_raw_ids.append(staged['raw_id'])

    return jobs, chunk_rows, loaded_raw_ids


def _ingest_vectors(jobs: list[dict[str, Any]], chunk_rows: list[dict[str, Any]], *, replace_existing: bool) -> tuple[bool, str, str]:
    settings = get_settings()
    vector_reset = False
    collection_name = settings.qdrant_collection_name
    qdrant_status = 'not_configured'
    try:
        from langchain_core.documents import Document
    except Exception as exc:
        if settings.qdrant_url and settings.openai_api_key:
            return vector_reset, collection_name, f'skipped_optional_vector_dependencies_missing:{exc}'
        return vector_reset, collection_name, 'skipped_local_only'

    try:
        if settings.qdrant_url and settings.openai_api_key:
            if replace_existing:
                reset_vector_store_collection()
                vector_reset = True
            documents = []
            ids = []
            job_map = {job['job_id']: job for job in jobs}
            for chunk in chunk_rows:
                job = job_map[chunk['job_id']]
                documents.append(
                    Document(
                        page_content=chunk['chunk_text'],
                        metadata={**job, 'chunk_id': chunk['chunk_id'], 'chunk_index': chunk['chunk_index']},
                    )
                )
                ids.append(chunk['chunk_id'])
            if documents:
                get_vector_store().add_documents(documents=documents, ids=ids)
            qdrant_status = 'ingested'
        else:
            qdrant_status = 'skipped_local_only'
    except Exception as exc:
        qdrant_status = f'failed:{exc}'
    return vector_reset, collection_name, qdrant_status


def ingest_jobs(limit: int | None = None, replace_existing: bool = True) -> dict[str, int | str | bool | None]:
    settings = get_settings()
    raw_rows = load_jsonl(settings.jobs_path)
    total_raw_rows = len(raw_rows)
    if limit is not None:
        raw_rows = raw_rows[:limit]

    init_sqlite()
    if replace_existing:
        clear_runtime_tables(clear_staging=True)

    staged_rows = _extract_to_staging(raw_rows, source_file=settings.jobs_path.name, replace_existing=replace_existing)
    jobs, chunk_rows, loaded_raw_ids = _transform_and_load_from_staging(source_file=settings.jobs_path.name)
    jobs_inserted = insert_jobs(jobs)
    chunks_inserted = insert_chunks(chunk_rows)
    transformed_rows = mark_staged_jobs_loaded(loaded_raw_ids)
    vector_reset, collection_name, qdrant_status = _ingest_vectors(jobs, chunk_rows, replace_existing=replace_existing)

    metadata = {
        'ingested_at': datetime.now(timezone.utc).isoformat(),
        'source_file': settings.jobs_path.name,
        'raw_rows_loaded': len(raw_rows),
        'raw_rows_available': total_raw_rows,
        'staged_rows': staged_rows,
        'transformed_rows': transformed_rows,
        'jobs_inserted': jobs_inserted,
        'chunks_inserted': chunks_inserted,
        'replace_existing': replace_existing,
        'limit': limit,
        'pipeline_mode': 'ELT',
        'qdrant_status': qdrant_status,
    }
    set_metadata('last_ingest', metadata)

    return {
        'jobs_inserted': jobs_inserted,
        'chunks_inserted': chunks_inserted,
        'collection_name': collection_name,
        'raw_rows_loaded': len(raw_rows),
        'replace_existing': replace_existing,
        'vector_reset': vector_reset,
    }
