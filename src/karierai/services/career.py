from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..database import get_market_summary_for_role, hybrid_search_jobs
from .cv import ROLE_SKILLS, _normalize_role_name, extract_cv_profile_data

_INTENT_KEYWORDS = {
    "rag": {"cari", "carikan", "lowongan", "jobs", "job", "tampilkan", "show", "remote"},
    "sql": {"berapa", "jumlah", "statistik", "distribusi", "rata", "rata-rata", "avg", "average", "gaji", "tren", "trend"},
    "cv": {"cv", "resume", "profil", "analisis", "ringkas", "extract"},
    "consultation": {"gap", "skill", "karier", "career", "konsultasi", "target", "cocok", "roadmap", "kurang"},
}
_HYBRID_PATTERNS = [
    re.compile(r"\b(cari|carikan|tampilkan|show)\b.*\b(jumlah|berapa|statistik|rata-rata|avg|average|distribusi)\b", re.IGNORECASE),
    re.compile(r"\bsekaligus\b", re.IGNORECASE),
    re.compile(r"\bserta\b", re.IGNORECASE),
]
_ROLE_PATTERNS = [
    r"(?:target role|posisi target|menjadi|jadi|role)\s*[:\-]?\s*([a-zA-Z ]{3,50})",
    r"(?:untuk|sebagai)\s+(data analyst|data scientist|business analyst|hr manager|recruiter|business intelligence|analytics engineer)",
]


def _tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^a-zA-Z0-9+#./-]+", " ", (text or "").lower())
    return [token for token in normalized.split() if len(token) >= 2]



def classify_chat_intent(query: str) -> dict[str, Any]:
    normalized_query = " ".join((query or "").split())
    lowered = normalized_query.lower()
    tokens = set(_tokenize(normalized_query))

    listing_signal = bool(tokens.intersection(_INTENT_KEYWORDS["rag"]))
    analytics_signal = bool(tokens.intersection(_INTENT_KEYWORDS["sql"]))
    cv_signal = "cv" in tokens or "resume" in tokens
    consultation_signal = bool(tokens.intersection(_INTENT_KEYWORDS["consultation"]))

    if any(pattern.search(normalized_query) for pattern in _HYBRID_PATTERNS) or (listing_signal and analytics_signal):
        intent = "hybrid"
    elif analytics_signal:
        intent = "sql"
    elif cv_signal and consultation_signal:
        intent = "consultation"
    elif cv_signal:
        intent = "cv"
    else:
        intent = "rag"

    confidence = 92 if intent != "rag" else 84
    if not lowered:
        confidence = 55

    return {
        "intent": intent,
        "confidence": confidence,
        "reason": "keyword_intent_classifier",
        "candidates": [{"intent": intent, "score": 1.0, "reason": "top_match"}],
    }



def extract_target_role(text: str) -> str | None:
    lower = (text or "").lower()
    for pattern in _ROLE_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            return match.group(1).strip().title()
    return None



def _unique_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join((value or "").split()).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result



def summarize_skill_overlap(cv_skills: list[str], job_text: str, job_title: str = "") -> dict[str, object]:
    lower = f"{job_title} {job_text}".lower()
    matched = [skill for skill in cv_skills if skill in lower]
    keyword_hits = Counter(matched)
    return {"matched_skills": sorted(keyword_hits), "match_count": len(matched)}



def _score_job(profile: dict[str, object], job: dict[str, Any]) -> tuple[float, dict[str, object]]:
    cv_skills = profile.get("skills", []) if isinstance(profile.get("skills"), list) else []
    likely_roles = profile.get("likely_roles", []) if isinstance(profile.get("likely_roles"), list) else []
    primary_role = str(profile.get("primary_role_guess") or "").lower().strip()
    years = int(profile.get("estimated_years_experience", 0) or 0)
    overlap = summarize_skill_overlap(cv_skills, str(job.get("job_description", "")), str(job.get("job_title", "")))
    matched_skills = overlap["matched_skills"]
    match_count = int(overlap["match_count"])

    title = str(job.get("job_title", "")).lower()
    description = str(job.get("job_description", "")).lower()
    role_bonus = 0.0
    matched_roles: list[str] = []
    for role in likely_roles:
        if role in title:
            role_bonus += 3.0
            matched_roles.append(role)
        elif role in description:
            role_bonus += 1.25
    if primary_role and primary_role in title and primary_role not in matched_roles:
        role_bonus += 1.8

    required_skills = ROLE_SKILLS.get(primary_role, [])
    must_have_matches = [skill for skill in required_skills if skill in matched_skills]
    missing_core_skills = [skill for skill in required_skills if skill not in matched_skills]
    core_skill_bonus = len(must_have_matches) * 1.2

    years_bonus = 0.0
    if years > 0 and any(token in description for token in [f"{years} tahun", f"{years} years", f"{years}+", "senior", "lead"]):
        years_bonus = 0.75

    retrieval_bonus = float(job.get("hybrid_score", 0.0)) * 0.08 + float(job.get("bm25_score", 0.0)) * 0.35
    score = match_count * 1.5 + role_bonus + core_skill_bonus + years_bonus + retrieval_bonus

    explanation: list[str] = []
    if matched_roles:
        explanation.append(f"Role CV selaras dengan judul lowongan: {', '.join(matched_roles)}")
    if matched_skills:
        explanation.append(f"Skill yang cocok: {', '.join(matched_skills[:6])}")
    if must_have_matches:
        explanation.append(f"Skill inti role yang cocok: {', '.join(must_have_matches[:4])}")
    if missing_core_skills and required_skills:
        explanation.append(f"Beberapa skill inti masih belum tampak: {', '.join(missing_core_skills[:3])}")
    if job.get("retrieval_sources"):
        explanation.append(f"Hybrid retrieval menemukan lowongan ini lewat: {', '.join(job['retrieval_sources'])}")
    if not explanation:
        explanation.append("Kecocokan dihitung dari retrieval dan overlap skill antara CV dan deskripsi lowongan.")

    return score, {
        "job_id": job.get("job_id"),
        "job_title": job.get("job_title"),
        "company_name": job.get("company_name"),
        "location": job.get("location"),
        "work_type": job.get("work_type"),
        "salary_raw": job.get("salary_raw"),
        "score": round(score, 2),
        "matched_skills": matched_skills,
        "explanation": explanation,
        "job_excerpt": str(job.get("job_description", ""))[:350],
    }



def _build_search_query(profile: dict[str, object], cv_text: str) -> str:
    primary_role = str(profile.get("primary_role_guess") or "").strip()
    likely_roles = [str(role) for role in (profile.get("likely_roles") or [])[:4] if str(role).strip()]
    skills = [str(skill) for skill in (profile.get("skills") or [])[:8] if str(skill).strip()]
    search_terms = _unique_terms([primary_role, *likely_roles, *skills])
    if search_terms:
        return " ".join(search_terms[:12])
    return " ".join((cv_text or "").split())[:160]



def _has_minimum_profile_signal(profile: dict[str, object]) -> bool:
    validation = profile.get("validation") if isinstance(profile.get("validation"), dict) else {}
    completeness = float(validation.get("completeness_score", 0) or 0)
    skills = profile.get("skills") if isinstance(profile.get("skills"), list) else []
    likely_roles = profile.get("likely_roles") if isinstance(profile.get("likely_roles"), list) else []
    primary_role = str(profile.get("primary_role_guess") or "").strip()
    return completeness >= 35 or len(skills) >= 2 or bool(primary_role) or bool(likely_roles)



def build_recommendations(cv_text: str, top_k: int = 5) -> dict[str, object]:
    profile = extract_cv_profile_data(cv_text)
    if not _has_minimum_profile_signal(profile):
        return {"profile": profile, "search_query": "", "matches": []}

    primary_role = str(profile.get("primary_role_guess") or "").strip()
    skill_hints = [str(skill) for skill in (profile.get("skills") or [])[:10]]
    search_query = _build_search_query(profile, cv_text)
    jobs = hybrid_search_jobs(
        search_query=search_query,
        limit=max(top_k * 8, 20),
        target_role=primary_role or None,
        skill_hints=skill_hints,
    )
    if not jobs and primary_role:
        jobs = hybrid_search_jobs(search_query=primary_role, limit=max(top_k * 8, 20), target_role=primary_role)

    scored: list[tuple[float, dict[str, object]]] = []
    for job in jobs:
        score, payload = _score_job(profile, job)
        if score > 0:
            scored.append((score, payload))
    scored.sort(key=lambda item: item[0], reverse=True)
    return {"profile": profile, "search_query": search_query, "matches": [payload for _, payload in scored[:top_k]]}



def build_career_consultation(cv_text: str, target_role: str) -> dict[str, Any]:
    profile = extract_cv_profile_data(cv_text)
    role_key = _normalize_role_name(target_role.lower().strip())
    required_skills = ROLE_SKILLS.get(role_key, [])
    if not required_skills and profile.get("primary_role_guess"):
        role_key = str(profile["primary_role_guess"]).lower()
        required_skills = ROLE_SKILLS.get(role_key, [])
    cv_skills = set(profile.get("skills", []))
    matched = [skill for skill in required_skills if skill in cv_skills]
    missing = [skill for skill in required_skills if skill not in cv_skills]
    market = get_market_summary_for_role(role_key or target_role)

    recommendations: list[str] = []
    if missing:
        recommendations.append(f"Prioritaskan penguatan skill berikut: {', '.join(missing[:5])}.")
    if market["sample_titles"]:
        recommendations.append(f"Pantau lowongan seperti: {', '.join(market['sample_titles'][:3])}.")
    if market["top_locations"]:
        recommendations.append(f"Lokasi pasar kerja teratas untuk role ini: {', '.join(market['top_locations'][:3])}.")
    if not recommendations:
        recommendations.append("Profil sudah cukup dekat. Fokus pada portfolio, hasil kerja, dan kesiapan interview.")

    return {
        "target_role": target_role,
        "profile": profile,
        "matched_skills": matched,
        "missing_skills": missing,
        "market_summary": market,
        "recommendations": recommendations,
    }
