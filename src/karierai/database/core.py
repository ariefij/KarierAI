from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import get_settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_jobs (
    raw_id TEXT PRIMARY KEY,
    raw_json TEXT NOT NULL,
    source_file TEXT,
    extracted_at TEXT,
    transformed_at TEXT,
    status TEXT DEFAULT 'staged'
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_title TEXT NOT NULL,
    company_name TEXT,
    location TEXT,
    work_type TEXT,
    salary_raw TEXT,
    job_description TEXT NOT NULL,
    scrape_timestamp TEXT,
    source_file TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS job_chunks (
    chunk_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    char_count INTEGER,
    token_estimate INTEGER,
    created_at TEXT,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS app_metadata (
    meta_key TEXT PRIMARY KEY,
    meta_value TEXT,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_raw_jobs_status ON raw_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(job_title);
CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs(location);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_name);
CREATE INDEX IF NOT EXISTS idx_jobs_work_type ON jobs(work_type);
CREATE INDEX IF NOT EXISTS idx_jobs_scrape_timestamp ON jobs(scrape_timestamp);
CREATE INDEX IF NOT EXISTS idx_job_chunks_job_id ON job_chunks(job_id);
CREATE INDEX IF NOT EXISTS idx_job_chunks_chunk_index ON job_chunks(chunk_index);

CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts USING fts5(
    job_id UNINDEXED,
    job_title,
    company_name,
    location,
    work_type,
    salary_raw,
    job_description,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""

SQLITE_WHITESPACE = re.compile(r"\s+")
_SAFE_FTS_TOKEN_RE = re.compile(r"[^0-9a-zA-Z+#./-]+")


def _normalize_salary_number(token: str, suffix: str) -> float | None:
    token = token.strip().replace(",", ".").replace(" ", "")
    if not token:
        return None
    if "." in token and token.count(".") > 1 and all(len(part) == 3 for part in token.split(".")[1:]):
        numeric = float(token.replace(".", ""))
    elif "," in token and token.count(",") > 1 and all(len(part) == 3 for part in token.split(",")[1:]):
        numeric = float(token.replace(",", ""))
    else:
        numeric = float(token)
    multiplier = {
        "m": 1_000_000,
        "jt": 1_000_000,
        "juta": 1_000_000,
        "k": 1_000,
        "rb": 1_000,
        "ribu": 1_000,
        "": 1,
    }.get(suffix.lower(), 1)
    return numeric * multiplier


def _extract_salary_numbers(value: str | None) -> list[float]:
    if not value:
        return []
    cleaned = str(value).lower().replace("\xa0", " ")
    cleaned = cleaned.replace("idr", " ").replace("rp", " ")
    matches = re.findall(r"(\d+(?:[\.,]\d+)*)(?:\s*)(m|jt|juta|k|rb|ribu)?", cleaned)
    numbers: list[float] = []
    for token, suffix in matches:
        try:
            parsed = _normalize_salary_number(token, suffix)
        except Exception:
            parsed = None
        if parsed is not None:
            numbers.append(parsed)
    return numbers


def _salary_min(value: str | None) -> float | None:
    numbers = _extract_salary_numbers(value)
    return min(numbers) if numbers else None


def _salary_max(value: str | None) -> float | None:
    numbers = _extract_salary_numbers(value)
    return max(numbers) if numbers else None


def _salary_mid(value: str | None) -> float | None:
    numbers = _extract_salary_numbers(value)
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _get_db_path() -> Path:
    settings = get_settings()
    path = settings.sqlite_file
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.create_function("salary_min", 1, _salary_min)
    conn.create_function("salary_max", 1, _salary_max)
    conn.create_function("salary_mid", 1, _salary_mid)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_sqlite() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)


def set_metadata(meta_key: str, meta_value: Any) -> None:
    init_sqlite()
    payload = json.dumps(meta_value, ensure_ascii=False) if not isinstance(meta_value, str) else meta_value
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO app_metadata(meta_key, meta_value, updated_at) VALUES (?, ?, ?)",
            (meta_key, payload, now),
        )


def get_metadata(meta_key: str, default: Any = None) -> Any:
    init_sqlite()
    with get_connection() as conn:
        row = conn.execute("SELECT meta_value FROM app_metadata WHERE meta_key = ?", (meta_key,)).fetchone()
    if not row:
        return default
    raw = row["meta_value"]
    try:
        return json.loads(raw)
    except Exception:
        return raw


def clear_runtime_tables(*, clear_staging: bool = True) -> None:
    init_sqlite()
    with get_connection() as conn:
        conn.execute("DELETE FROM job_chunks")
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM jobs_fts")
        if clear_staging:
            conn.execute("DELETE FROM raw_jobs")


def stage_raw_jobs(rows: Iterable[dict[str, Any]], *, replace_existing: bool = True) -> int:
    staged_rows = list(rows)
    if not staged_rows:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    payloads = []
    for idx, row in enumerate(staged_rows):
        payloads.append(
            {
                "raw_id": str(row.get("raw_id") or f"raw-{idx}-{now}"),
                "raw_json": json.dumps(row.get("raw_json") or row, ensure_ascii=False),
                "source_file": row.get("source_file"),
                "extracted_at": row.get("extracted_at") or now,
                "transformed_at": row.get("transformed_at"),
                "status": row.get("status") or "staged",
            }
        )
    with get_connection() as conn:
        if replace_existing:
            conn.execute("DELETE FROM raw_jobs")
        conn.executemany(
            """
            INSERT OR REPLACE INTO raw_jobs(raw_id, raw_json, source_file, extracted_at, transformed_at, status)
            VALUES (:raw_id, :raw_json, :source_file, :extracted_at, :transformed_at, :status)
            """,
            payloads,
        )
    return len(payloads)


def fetch_staged_raw_jobs(status: str | None = None) -> list[dict[str, Any]]:
    init_sqlite()
    sql = "SELECT raw_id, raw_json, source_file, extracted_at, transformed_at, status FROM raw_jobs"
    params: tuple[Any, ...] = ()
    if status:
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY extracted_at ASC, raw_id ASC"
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def mark_staged_jobs_loaded(raw_ids: Iterable[str]) -> int:
    raw_ids = [raw_id for raw_id in raw_ids if raw_id]
    if not raw_ids:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.executemany(
            "UPDATE raw_jobs SET transformed_at = ?, status = 'loaded' WHERE raw_id = ?",
            [(now, raw_id) for raw_id in raw_ids],
        )
    return len(raw_ids)


def _rebuild_fts_index(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]] | None = None) -> None:
    payload_rows = list(rows or [])
    if not payload_rows:
        conn.execute("DELETE FROM jobs_fts")
        conn.execute(
            """
            INSERT INTO jobs_fts(job_id, job_title, company_name, location, work_type, salary_raw, job_description)
            SELECT job_id, job_title, COALESCE(company_name, ''), COALESCE(location, ''), COALESCE(work_type, ''),
                   COALESCE(salary_raw, ''), COALESCE(job_description, '')
            FROM jobs
            """
        )
        return

    conn.executemany("DELETE FROM jobs_fts WHERE job_id = ?", [(row["job_id"],) for row in payload_rows])
    conn.executemany(
        """
        INSERT INTO jobs_fts(job_id, job_title, company_name, location, work_type, salary_raw, job_description)
        VALUES (:job_id, :job_title, COALESCE(:company_name, ''), COALESCE(:location, ''), COALESCE(:work_type, ''),
                COALESCE(:salary_raw, ''), COALESCE(:job_description, ''))
        """,
        payload_rows,
    )


def insert_jobs(rows: Iterable[dict[str, Any]]) -> int:
    payload_rows = list(rows)
    if not payload_rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO jobs (
                job_id, job_title, company_name, location, work_type,
                salary_raw, job_description, scrape_timestamp, source_file, created_at
            ) VALUES (
                :job_id, :job_title, :company_name, :location, :work_type,
                :salary_raw, :job_description, :scrape_timestamp, :source_file, :created_at
            )
            """,
            payload_rows,
        )
        _rebuild_fts_index(conn, payload_rows)
    return len(payload_rows)


def insert_chunks(rows: Iterable[dict[str, Any]]) -> int:
    payload_rows = list(rows)
    if not payload_rows:
        return 0
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO job_chunks (
                chunk_id, job_id, chunk_index, chunk_text, char_count, token_estimate, created_at
            ) VALUES (
                :chunk_id, :job_id, :chunk_index, :chunk_text, :char_count, :token_estimate, :created_at
            )
            """,
            payload_rows,
        )
    return len(payload_rows)


def fetch_job_by_id(job_id: str) -> dict[str, Any] | None:
    init_sqlite()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_available_filters() -> dict[str, list[str]]:
    init_sqlite()
    with get_connection() as conn:
        locations = [r[0] for r in conn.execute("SELECT DISTINCT location FROM jobs WHERE location IS NOT NULL ORDER BY location LIMIT 20")]
        work_types = [r[0] for r in conn.execute("SELECT DISTINCT work_type FROM jobs WHERE work_type IS NOT NULL ORDER BY work_type LIMIT 20")]
        companies = [r[0] for r in conn.execute("SELECT company_name FROM jobs WHERE company_name IS NOT NULL GROUP BY company_name ORDER BY COUNT(*) DESC LIMIT 20")]
    return {"locations": locations, "work_types": work_types, "companies": companies}


def _count_raw_dataset_rows() -> int:
    settings = get_settings()
    if not settings.jobs_path.exists():
        return 0
    with settings.jobs_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def get_database_stats() -> dict[str, Any]:
    init_sqlite()
    with get_connection() as conn:
        jobs_count = conn.execute("SELECT COUNT(*) AS total FROM jobs").fetchone()["total"]
        chunks_count = conn.execute("SELECT COUNT(*) AS total FROM job_chunks").fetchone()["total"]
        staged_count = conn.execute("SELECT COUNT(*) AS total FROM raw_jobs").fetchone()["total"]
        loaded_staging_count = conn.execute("SELECT COUNT(*) AS total FROM raw_jobs WHERE status = 'loaded'").fetchone()["total"]
    ingest_meta = get_metadata("last_ingest", default={}) or {}
    raw_dataset_count = _count_raw_dataset_rows()
    ingested_rows = int(ingest_meta.get("raw_rows_loaded", jobs_count) or 0)
    is_synced = bool(jobs_count) and jobs_count == ingested_rows == raw_dataset_count
    return {
        "jobs_count": jobs_count,
        "chunks_count": chunks_count,
        "staged_raw_jobs": staged_count,
        "loaded_staged_jobs": loaded_staging_count,
        "raw_dataset_count": raw_dataset_count,
        "last_ingest": ingest_meta,
        "database_file": str(_get_db_path()),
        "is_likely_synced": is_synced,
        "fts_enabled": True,
    }


def _build_fts_query(search_query: str) -> str | None:
    normalized = SQLITE_WHITESPACE.sub(" ", search_query.strip().lower())
    tokens = []
    for raw_token in re.split(r"[,\s]+", normalized):
        cleaned = _SAFE_FTS_TOKEN_RE.sub("", raw_token)
        if len(cleaned) >= 2:
            tokens.append(f'{cleaned}*')
    if not tokens:
        return None
    return " ".join(tokens[:10])


def _search_jobs_like(search_query: str, limit: int) -> list[dict[str, Any]]:
    terms = [term for term in re.split(r"[,\s]+", search_query.lower()) if len(term) >= 2][:8]
    if not terms:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT *, 0 AS relevance_score, 'latest' AS search_backend FROM jobs ORDER BY scrape_timestamp DESC, created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    scoring_fragments: list[str] = []
    where_fragments: list[str] = []
    sql_params: list[Any] = []
    for term in terms:
        like_term = f"%{term}%"
        scoring_fragments.append(
            "("
            "CASE WHEN LOWER(job_title) LIKE ? THEN 8 ELSE 0 END + "
            "CASE WHEN LOWER(company_name) LIKE ? THEN 5 ELSE 0 END + "
            "CASE WHEN LOWER(location) LIKE ? THEN 4 ELSE 0 END + "
            "CASE WHEN LOWER(work_type) LIKE ? THEN 3 ELSE 0 END + "
            "CASE WHEN LOWER(job_description) LIKE ? THEN 2 ELSE 0 END)"
        )
        sql_params.extend([like_term, like_term, like_term, like_term, like_term])
        where_fragments.append(
            "(LOWER(job_title) LIKE ? OR LOWER(company_name) LIKE ? OR LOWER(location) LIKE ? OR LOWER(work_type) LIKE ? OR LOWER(job_description) LIKE ?)"
        )
        sql_params.extend([like_term, like_term, like_term, like_term, like_term])
    relevance_sql = " + ".join(scoring_fragments)
    sql = (
        f"SELECT *, ({relevance_sql}) AS relevance_score, 'like' AS search_backend FROM jobs "
        f"WHERE {' OR '.join(where_fragments)} "
        "ORDER BY relevance_score DESC, scrape_timestamp DESC, created_at DESC LIMIT ?"
    )
    params = [*sql_params, limit]
    with get_connection() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def search_jobs(search_query: str = "", limit: int = 10) -> list[dict[str, Any]]:
    init_sqlite()
    normalized = SQLITE_WHITESPACE.sub(" ", search_query.strip())
    if not normalized:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT *, 0 AS relevance_score, 'latest' AS search_backend FROM jobs ORDER BY scrape_timestamp DESC, created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    fts_query = _build_fts_query(normalized)
    if not fts_query:
        return _search_jobs_like(normalized, limit)

    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT j.*, CAST(-bm25(jobs_fts) AS REAL) AS relevance_score, 'fts5' AS search_backend
                FROM jobs_fts
                JOIN jobs AS j ON j.job_id = jobs_fts.job_id
                WHERE jobs_fts MATCH ?
                ORDER BY bm25(jobs_fts), j.scrape_timestamp DESC, j.created_at DESC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        results = [dict(row) for row in rows]
        if results:
            return results
    except sqlite3.OperationalError:
        pass
    return _search_jobs_like(normalized, limit)
