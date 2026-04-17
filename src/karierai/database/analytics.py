from __future__ import annotations

import json
import re
from typing import Any

from ..llm import invoke_json, require_llm
from .core import get_connection, init_sqlite

SAFE_SQL_PREFIXES = ("select", "with")
FORBIDDEN_SQL_TOKENS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum|reindex)\b",
    flags=re.IGNORECASE,
)
ALLOWED_TABLES = {"jobs", "job_chunks"}


def _build_schema_for_llm() -> str:
    return (
        "Gunakan SQLite. Tabel jobs(job_id, job_title, company_name, location, work_type, salary_raw, "
        "job_description, scrape_timestamp, source_file, created_at). "
        "Tabel job_chunks(chunk_id, job_id, chunk_index, chunk_text, char_count, token_estimate, created_at). "
        "Fungsi SQLite yang boleh dipakai: salary_min(salary_raw), salary_max(salary_raw), salary_mid(salary_raw). "
        "Balas JSON dengan kunci sql dan explanation. Hanya SELECT/CTE read-only. "
        "Tambahkan LIMIT 50 bila query menampilkan daftar baris."
    )


def _generate_sql_with_llm(question: str) -> tuple[str, str]:
    require_llm()
    prompt = (
        f"{_build_schema_for_llm()}\n\n"
        "Gunakan hanya tabel dan kolom yang tersedia. Jangan gunakan DML atau DDL.\n"
        f"Pertanyaan user: {question}\n\n"
        'Contoh output: {"sql": "SELECT COUNT(*) AS total FROM jobs", "explanation": "count all jobs"}'
    )
    payload, _ = invoke_json(prompt, temperature=0)
    sql = str(payload.get("sql") or "").strip()
    explanation = str(payload.get("explanation") or "llm_text2sql").strip()
    if not sql:
        raise RuntimeError("LLM tidak mengembalikan SQL.")
    return sql, explanation


def _validate_sql(sql: str) -> str:
    normalized = " ".join(sql.strip().split())
    if not normalized:
        raise ValueError("SQL kosong.")
    if ";" in normalized.rstrip(";"):
        raise ValueError("Hanya satu statement SQL yang diizinkan.")
    normalized = normalized.rstrip(";")
    if not normalized.lower().startswith(SAFE_SQL_PREFIXES):
        raise ValueError("Hanya query SELECT/CTE yang diizinkan.")
    if FORBIDDEN_SQL_TOKENS.search(normalized):
        raise ValueError("Query mengandung operasi yang tidak diizinkan.")
    tables = [
        match.group(1).lower()
        for match in re.finditer(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", normalized, flags=re.IGNORECASE)
    ]
    if any(table not in ALLOWED_TABLES for table in tables):
        raise ValueError("Query mengakses tabel di luar whitelist.")
    if " limit " not in f" {normalized.lower()} ":
        normalized += " LIMIT 50"
    return normalized


def _execute_safe_sql(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def summarize_analytics_result(result: dict[str, Any]) -> str:
    rows = result.get("rows", []) or []
    if not rows:
        return "Tidak ada data yang cocok dengan pertanyaan tersebut."
    first = rows[0]
    if "total" in first and len(first) == 1:
        return f"Total hasil: {first['total']}."
    if "total_companies" in first:
        return f"Jumlah perusahaan unik: {first['total_companies']}."
    if "total_locations" in first:
        return f"Jumlah lokasi unik: {first['total_locations']}."
    if "avg_salary" in first and "location" not in first:
        return f"Rata-rata gaji terhitung: {round(float(first['avg_salary']), 2)}."
    preview = rows[:5]
    return "Preview hasil: " + json.dumps(preview, ensure_ascii=False)


def run_safe_analytics(question: str) -> dict[str, Any]:
    init_sqlite()
    sql, explanation = _generate_sql_with_llm(question)
    safe_sql = _validate_sql(sql)
    rows = _execute_safe_sql(safe_sql)
    result = {
        "mode": "llm_text2sql",
        "sql": safe_sql,
        "params": (),
        "rows": rows,
        "explanation": explanation,
    }
    result["summary"] = summarize_analytics_result(result)
    return result


def get_market_summary_for_role(target_role: str) -> dict[str, Any]:
    init_sqlite()
    like_role = f"%{target_role.lower()}%"
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS total FROM jobs WHERE LOWER(job_title) LIKE ?", (like_role,)).fetchone()["total"]
        locations = [
            row["location"]
            for row in conn.execute(
                "SELECT location, COUNT(*) AS total FROM jobs WHERE LOWER(job_title) LIKE ? AND location IS NOT NULL GROUP BY location ORDER BY total DESC LIMIT 5",
                (like_role,),
            )
        ]
        sample_titles = [
            row["job_title"]
            for row in conn.execute(
                "SELECT job_title FROM jobs WHERE LOWER(job_title) LIKE ? ORDER BY scrape_timestamp DESC LIMIT 5",
                (like_role,),
            )
        ]
    return {"matching_jobs": count, "top_locations": locations, "sample_titles": sample_titles}
