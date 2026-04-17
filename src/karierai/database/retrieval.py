from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from ..config import get_settings
from .core import get_connection, init_sqlite, search_jobs
from .vector import get_vector_store

TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9+#./-]+")
_LOCATION_HINTS = ["remote", "hybrid", "jakarta", "bandung", "surabaya"]



def _tokenize(text: str) -> list[str]:
    normalized = TOKEN_SPLIT_RE.sub(" ", (text or "").lower())
    return [token for token in normalized.split() if len(token) >= 2]



def _compose_job_text(job: dict[str, Any]) -> str:
    parts = [
        str(job.get("job_title") or ""),
        str(job.get("company_name") or ""),
        str(job.get("location") or ""),
        str(job.get("work_type") or ""),
        str(job.get("salary_raw") or ""),
        str(job.get("job_description") or ""),
    ]
    return " ".join(part for part in parts if part).strip()



def _fetch_jobs_for_retrieval() -> list[dict[str, Any]]:
    init_sqlite()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT job_id, job_title, company_name, location, work_type, salary_raw, job_description, scrape_timestamp, created_at FROM jobs"
        ).fetchall()
    return [dict(row) for row in rows]



def _corpus_cache_key() -> tuple[str, int, int]:
    sqlite_file = get_settings().sqlite_file
    try:
        stat = sqlite_file.stat()
        return str(sqlite_file), int(stat.st_mtime_ns), int(stat.st_size)
    except FileNotFoundError:
        return str(sqlite_file), 0, 0


@lru_cache(maxsize=4)
def _load_corpus(db_path: str, mtime_ns: int, file_size: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[Counter[str]], list[int], dict[str, int], float]:
    del db_path, mtime_ns, file_size
    jobs = _fetch_jobs_for_retrieval()
    term_counts_by_doc: list[Counter[str]] = []
    doc_lengths: list[int] = []
    document_frequencies: Counter[str] = Counter()
    for job in jobs:
        tokens = _tokenize(_compose_job_text(job))
        counts = Counter(tokens)
        term_counts_by_doc.append(counts)
        doc_lengths.append(len(tokens) or 1)
        for token in counts:
            document_frequencies[token] += 1
    avgdl = (sum(doc_lengths) / max(len(doc_lengths), 1)) or 1.0
    job_by_id = {str(job.get("job_id")): job for job in jobs}
    return jobs, job_by_id, term_counts_by_doc, doc_lengths, dict(document_frequencies), avgdl



def _build_bm25_scores(
    query_tokens: list[str],
    *,
    term_counts_by_doc: list[Counter[str]],
    doc_lengths: list[int],
    document_frequencies: dict[str, int],
    avgdl: float,
) -> list[float]:
    if not query_tokens or not term_counts_by_doc:
        return [0.0 for _ in term_counts_by_doc]

    total_docs = len(term_counts_by_doc)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for term_counts, doc_len in zip(term_counts_by_doc, doc_lengths):
        score = 0.0
        for token in query_tokens:
            if token not in term_counts:
                continue
            df = document_frequencies.get(token, 0)
            idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
            tf = term_counts[token]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            score += idf * (numerator / denominator)
        scores.append(round(score, 6))
    return scores



def _rank_positions(items: list[tuple[str, float]]) -> dict[str, int]:
    ranked = sorted(items, key=lambda item: item[1], reverse=True)
    return {job_id: idx + 1 for idx, (job_id, score) in enumerate(ranked) if score > 0}



def _vector_job_ranks(query: str, candidate_pool: int) -> dict[str, int]:
    try:
        docs = get_vector_store().similarity_search_with_score(query, k=candidate_pool)
    except Exception:
        return {}

    best_rank: dict[str, int] = {}
    for idx, (doc, _score) in enumerate(docs, start=1):
        job_id = str(doc.metadata.get("job_id") or "").strip()
        if not job_id:
            continue
        if job_id not in best_rank or idx < best_rank[job_id]:
            best_rank[job_id] = idx
    return best_rank



def _rrf_score(*ranks: int | None, k: int = 60) -> float:
    total = 0.0
    for rank in ranks:
        if rank is None:
            continue
        total += 1.0 / (k + rank)
    return total



def _rerank_bonus(query: str, job: dict[str, Any], *, target_role: str | None = None, skill_hints: list[str] | None = None) -> float:
    lower_query = query.lower()
    title = str(job.get("job_title") or "").lower()
    description = str(job.get("job_description") or "").lower()
    company = str(job.get("company_name") or "").lower()
    location = str(job.get("location") or "").lower()
    work_type = str(job.get("work_type") or "").lower()

    query_tokens = set(_tokenize(lower_query))
    title_tokens = set(_tokenize(title))
    body_tokens = set(_tokenize(f"{company} {location} {work_type} {description}"))

    score = 0.0
    score += len(query_tokens.intersection(title_tokens)) * 1.3
    score += len(query_tokens.intersection(body_tokens)) * 0.35

    if target_role and target_role.lower() in title:
        score += 2.8
    if target_role and target_role.lower() in description:
        score += 1.2

    for phrase in _LOCATION_HINTS:
        if phrase in lower_query and (phrase in location or phrase in work_type or phrase in description):
            score += 0.9

    for skill in skill_hints or []:
        skill_lower = skill.lower()
        if skill_lower in title:
            score += 0.9
        elif skill_lower in description:
            score += 0.45

    if lower_query and (lower_query in description or lower_query in title):
        score += 1.5
    return round(score, 6)



def hybrid_search_jobs(
    search_query: str,
    limit: int = 10,
    *,
    target_role: str | None = None,
    skill_hints: list[str] | None = None,
) -> list[dict[str, Any]]:
    init_sqlite()
    query_tokens = _tokenize(search_query)
    candidate_pool = max(limit * 8, 30)

    cache_key = _corpus_cache_key()
    jobs, job_by_id, term_counts_by_doc, doc_lengths, document_frequencies, avgdl = _load_corpus(*cache_key)
    if not jobs:
        return []

    bm25_scores = _build_bm25_scores(
        query_tokens,
        term_counts_by_doc=term_counts_by_doc,
        doc_lengths=doc_lengths,
        document_frequencies=document_frequencies,
        avgdl=avgdl,
    )
    bm25_by_id = {str(job.get("job_id")): score for job, score in zip(jobs, bm25_scores)}
    bm25_ranks = _rank_positions([(str(job.get("job_id")), score) for job, score in zip(jobs, bm25_scores)])

    lexical_rows = search_jobs(search_query=search_query, limit=candidate_pool)
    lexical_ranks = {str(row.get("job_id")): idx + 1 for idx, row in enumerate(lexical_rows)}
    vector_ranks = _vector_job_ranks(search_query, candidate_pool)

    candidate_ids = set(list(bm25_ranks)[:candidate_pool]) | set(lexical_ranks) | set(vector_ranks)
    if not candidate_ids:
        candidate_ids = {str(job.get("job_id")) for job in jobs[:candidate_pool]}

    scored: list[tuple[float, dict[str, Any]]] = []
    for job_id in candidate_ids:
        job = job_by_id.get(job_id)
        if not job:
            continue
        bm25_score = bm25_by_id.get(job_id, 0.0)
        fused = _rrf_score(bm25_ranks.get(job_id), lexical_ranks.get(job_id), vector_ranks.get(job_id))
        rerank_score = _rerank_bonus(search_query, job, target_role=target_role, skill_hints=skill_hints)
        final_score = round(fused * 100 + rerank_score + bm25_score, 6)
        payload = {
            **job,
            "bm25_score": bm25_score,
            "hybrid_score": final_score,
            "rerank_score": rerank_score,
            "retrieval_sources": [
                source
                for source, present in [
                    ("bm25", job_id in bm25_ranks),
                    ("sqlite_lexical", job_id in lexical_ranks),
                    ("vector", job_id in vector_ranks),
                ]
                if present
            ],
        }
        scored.append((final_score, payload))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in scored[:limit]]


__all__ = ["hybrid_search_jobs"]
