from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

try:
    import spacy  # type: ignore
except Exception:
    spacy = None

from ..llm import extract_json_object, invoke_text, llm_is_available

SKILL_KEYWORDS = [
    'python', 'sql', 'excel', 'power bi', 'tableau', 'communication',
    'leadership', 'recruitment', 'payroll', 'analysis', 'machine learning',
    'statistics', 'dashboard', 'etl', 'data visualization', 'forecasting',
    'reporting', 'hris', 'talent acquisition', 'business intelligence',
    'r', 'spark', 'tensorflow', 'pytorch', 'project management',
    'data cleaning', 'data modelling', 'data modeling', 'presentation', 'problem solving',
    'numpy', 'pandas', 'scikit-learn', 'bigquery', 'looker', 'metabase',
    'a/b testing', 'experimentation', 'stakeholder management', 'data warehousing',
    'dbt', 'airflow', 'nlp', 'deep learning', 'data engineering', 'llm', 'generative ai',
]

SKILL_ALIASES = {
    'powerbi': 'power bi',
    'bi': 'business intelligence',
    'ml': 'machine learning',
    'nlp engineer': 'nlp',
    'genai': 'generative ai',
    'llms': 'llm',
    'data visualisation': 'data visualization',
    'visualization': 'data visualization',
    'visualisation': 'data visualization',
    'communication skills': 'communication',
    'excel advanced': 'excel',
    'data modelling': 'data modeling',
    'google sheets': 'excel',
    'spreadsheet': 'excel',
    'spreadsheet analysis': 'excel',
    'ms excel': 'excel',
    'power query': 'etl',
    'pyspark': 'spark',
    'sklearn': 'scikit-learn',
    'scikit learn': 'scikit-learn',
    'stakeholder communication': 'stakeholder management',
}

ROLE_KEYWORDS = [
    'data analyst', 'business analyst', 'data scientist', 'machine learning engineer',
    'hr manager', 'recruiter', 'talent acquisition', 'payroll specialist',
    'business intelligence', 'product analyst', 'finance analyst', 'bi analyst',
    'analytics engineer', 'people analyst', 'hr analyst', 'data engineer',
    'nlp engineer', 'ai engineer',
]

ROLE_ALIASES = {
    'bi analyst': 'business intelligence',
    'business intelligence analyst': 'business intelligence',
    'ta': 'talent acquisition',
    'ml engineer': 'machine learning engineer',
    'people analytics': 'people analyst',
    'ai engineer': 'machine learning engineer',
}

EDUCATION_KEYWORDS = ['s1', 's2', 'sarjana', 'bachelor', 'master', 'phd', 'universitas', 'university']

ROLE_SKILLS = {
    'data analyst': ['sql', 'excel', 'tableau', 'power bi', 'analysis', 'statistics', 'dashboard'],
    'data scientist': ['python', 'sql', 'machine learning', 'statistics', 'pandas', 'scikit-learn'],
    'hr manager': ['leadership', 'recruitment', 'communication', 'payroll', 'hris'],
    'business analyst': ['sql', 'excel', 'dashboard', 'analysis', 'communication', 'stakeholder management'],
    'business intelligence': ['sql', 'power bi', 'tableau', 'dashboard', 'business intelligence', 'etl'],
    'machine learning engineer': ['python', 'sql', 'machine learning', 'tensorflow', 'pytorch', 'spark'],
    'analytics engineer': ['sql', 'python', 'etl', 'data modeling', 'bigquery', 'dbt'],
    'data engineer': ['python', 'sql', 'etl', 'airflow', 'spark', 'data warehousing'],
}

SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    ('summary', ['summary', 'profil', 'profile', 'tentang saya', 'about me']),
    ('experience', ['experience', 'pengalaman', 'work experience', 'professional experience', 'riwayat kerja']),
    ('skills', ['skills', 'keahlian', 'kompetensi', 'toolbox', 'stack']),
    ('education', ['education', 'pendidikan', 'academic background']),
    ('certifications', ['certification', 'certifications', 'sertifikasi', 'lisensi']),
    ('projects', ['projects', 'proyek', 'portfolio', 'portofolio']),
    ('languages', ['languages', 'bahasa']),
    ('contact', ['contact', 'kontak']),
]

LANGUAGE_HINTS = [
    'indonesian', 'english', 'mandarin', 'japanese', 'korean', 'german', 'french',
    'bahasa indonesia', 'inggris', 'jepang', 'mandarin',
]

CERTIFICATION_HINTS = [
    'certified', 'certificate', 'certification', 'sertifikat', 'sertifikasi', 'google data analytics',
    'aws', 'scrum', 'oracle', 'microsoft', 'coursera',
]

EXPERIENCE_SENIORITY_RULES = [
    (8, 'senior'),
    (5, 'mid-senior'),
    (2, 'mid-level'),
    (0, 'entry-level'),
]

MONTH_NAMES = (
    'jan|feb|mar|apr|mei|may|jun|jul|aug|agt|sep|oct|okt|nov|dec|des|january|february|march|april|june|july|august|'
    'september|october|november|december'
)

EMAIL_RE = re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', re.IGNORECASE)
PHONE_RE = re.compile(r'(?:\+62|62|0)\d{8,13}')
URL_RE = re.compile(r'(?:https?://|www\.)\S+', re.IGNORECASE)
LINKEDIN_RE = re.compile(r'(?:linkedin\.com/in/|linkedin:?[\s/]+)([a-z0-9\-_/]+)', re.IGNORECASE)
GITHUB_RE = re.compile(r'(?:github\.com/|github:?[\s/]+)([a-z0-9\-_/]+)', re.IGNORECASE)
LOCATION_RE = re.compile(r'\b(jakarta|bandung|surabaya|yogyakarta|bogor|depok|tangerang|bekasi|semarang|medan|denpasar|remote|hybrid|indonesia|singapore)\b', re.IGNORECASE)
YEAR_RANGE_RE = re.compile(rf'(?:{MONTH_NAMES})?\s*(20\d{{2}}|19\d{{2}})\s*(?:-|to|sampai|–|—)\s*(present|current|sekarang|20\d{{2}}|19\d{{2}})', re.IGNORECASE)
YEARS_OF_EXPERIENCE_RE = re.compile(r'(\d+)\+?\s*(?:tahun|years?)', re.IGNORECASE)
DEGREE_RE = re.compile(r"(?:s1|s2|s3|d3|bachelor(?:'s)?|master(?:'s)?|phd|sarjana)[^\n,;]{0,80}", re.IGNORECASE)
BULLET_SPLIT_RE = re.compile(r'\r?\n+')


def _normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def _split_lines(text: str) -> list[str]:
    return [line.strip(' -•\t') for line in BULLET_SPLIT_RE.split(text) if line.strip()]


def _normalize_skill_name(value: str) -> str:
    lowered = ' '.join(value.lower().split())
    return SKILL_ALIASES.get(lowered, lowered)


def _normalize_role_name(value: str) -> str:
    lowered = ' '.join(value.lower().split())
    return ROLE_ALIASES.get(lowered, lowered)


def _tokenize(text: str) -> list[str]:
    normalized = re.sub(r'[^a-zA-Z0-9+#./-]+', ' ', text.lower())
    return [token for token in normalized.split() if len(token) >= 2]


def _unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = _normalize_whitespace(item)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _find_keywords(text: str, keywords: list[str], *, normalize: str) -> list[str]:
    lower = text.lower()
    found: list[str] = []
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = rf'(?<!\w){re.escape(keyword.lower())}(?!\w)'
        if re.search(pattern, lower):
            found.append(keyword)
    if normalize == 'skill':
        normalized = [_normalize_skill_name(item) for item in found]
    elif normalize == 'role':
        normalized = [_normalize_role_name(item) for item in found]
    else:
        normalized = found
    return _unique_preserve(normalized)


def _extract_years(text: str) -> dict[str, int | list[int]]:
    lower = text.lower()
    mentions = [int(match) for match in YEARS_OF_EXPERIENCE_RE.findall(lower)]
    current_year = datetime.now(timezone.utc).year
    durations: list[int] = []
    for start_year, end_year in YEAR_RANGE_RE.findall(lower):
        try:
            start = int(start_year)
            end = current_year if str(end_year).lower() in {'present', 'current', 'sekarang'} else int(end_year)
        except Exception:
            continue
        if start <= current_year and end >= start:
            durations.append(max(0, min(end, current_year) - start))
    all_values = sorted({*mentions, *durations})
    return {'mentions': all_values, 'max_years': max(all_values) if all_values else 0}


def _extract_sentences(text: str, keywords: list[str], limit: int = 5) -> list[str]:
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    selected: list[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword in lower for keyword in keywords):
            cleaned = ' '.join(sentence.split())
            if cleaned and cleaned not in selected:
                selected.append(cleaned)
        if len(selected) >= limit:
            break
    return selected


def _normalize_section_name(line: str) -> str | None:
    lowered = _normalize_whitespace(line.lower()).strip(':')
    for name, hints in SECTION_PATTERNS:
        if lowered in hints:
            return name
    return None


def _split_sections(text: str) -> list[tuple[str, str]]:
    lines = _split_lines(text)
    if not lines:
        return []
    sections: list[tuple[str, str]] = []
    current_name = 'general'
    buffer: list[str] = []
    for line in lines:
        section_name = _normalize_section_name(line)
        is_heading = section_name is not None and len(line.split()) <= 4
        if is_heading:
            if buffer:
                sections.append((current_name, '\n'.join(buffer).strip()))
            current_name = section_name
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_name, '\n'.join(buffer).strip()))
    return sections or [('general', _normalize_whitespace(text))]


def _build_section_map(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, content in _split_sections(text):
        if content:
            mapping[name] = content
    if 'general' not in mapping:
        mapping['general'] = _normalize_whitespace(text)
    return mapping


def _extract_contact_info(text: str) -> dict[str, Any]:
    compact = _normalize_whitespace(text)
    email = EMAIL_RE.search(compact)
    phone = PHONE_RE.search(compact)
    urls = URL_RE.findall(compact)
    linkedin_match = LINKEDIN_RE.search(compact)
    github_match = GITHUB_RE.search(compact)
    location_match = LOCATION_RE.search(compact)
    return {
        'email': email.group(0) if email else None,
        'phone': phone.group(0) if phone else None,
        'linkedin': linkedin_match.group(0) if linkedin_match else None,
        'github': github_match.group(0) if github_match else None,
        'location_hint': location_match.group(0).title() if location_match else None,
        'urls': urls[:5],
    }


def _extract_languages(text: str) -> list[str]:
    languages = _find_keywords(text, LANGUAGE_HINTS, normalize='language')
    normalized: list[str] = []
    for item in languages:
        fixed = item.lower().replace('bahasa indonesia', 'indonesian').replace('inggris', 'english').replace('jepang', 'japanese')
        if fixed not in normalized:
            normalized.append(fixed)
    return normalized


def _extract_certifications(text: str) -> list[str]:
    lines = _split_lines(text)
    matches: list[str] = []
    for line in lines:
        lower = line.lower()
        if any(hint in lower for hint in CERTIFICATION_HINTS):
            cleaned = _normalize_whitespace(line)
            if cleaned not in matches:
                matches.append(cleaned)
    return matches[:8]


def _extract_education_entries(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in _split_lines(text):
        lower = line.lower()
        if not (DEGREE_RE.search(lower) or 'universitas' in lower or 'university' in lower):
            continue
        degree_match = DEGREE_RE.search(line)
        institution = None
        inst_match = re.search(r'(?:universitas|university|institut|institute|politeknik)\s+[A-Za-z0-9 .&-]+', line, re.IGNORECASE)
        if inst_match:
            institution = _normalize_whitespace(inst_match.group(0))
        year_match = re.search(r'(19\d{2}|20\d{2})', line)
        entries.append(
            {
                'raw': _normalize_whitespace(line),
                'degree': _normalize_whitespace(degree_match.group(0)) if degree_match else None,
                'institution': institution,
                'year': int(year_match.group(1)) if year_match else None,
            }
        )
    return entries[:6]


def _extract_experience_entries(text: str) -> list[dict[str, Any]]:
    lines = _split_lines(text)
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in lines:
        lower = line.lower()
        is_date_line = bool(YEAR_RANGE_RE.search(lower) or YEARS_OF_EXPERIENCE_RE.search(lower))
        looks_like_job_line = any(token in lower for token in ROLE_KEYWORDS) or ' at ' in lower or ' - ' in line or ' | ' in line
        if looks_like_job_line and len(line.split()) <= 18:
            if current:
                entries.append(current)
            role = None
            company = None
            role_match = None
            for keyword in sorted(ROLE_KEYWORDS, key=len, reverse=True):
                if keyword in lower:
                    role_match = keyword
                    break
            if role_match:
                role = _normalize_role_name(role_match)
            company_match = re.search(r'(?:at|@|\||-|, )\s*([A-Z][A-Za-z0-9&.,\- ]{2,60})', line)
            if company_match:
                company = _normalize_whitespace(company_match.group(1))
            current = {'raw': _normalize_whitespace(line), 'title': role, 'company': company, 'date_range': None, 'highlights': []}
            continue
        if current and is_date_line and current.get('date_range') is None:
            current['date_range'] = _normalize_whitespace(line)
            continue
        if current and len(current['highlights']) < 4:
            current['highlights'].append(_normalize_whitespace(line))

    if current:
        entries.append(current)
    return entries[:8]


def _score_roles(skills: list[str], likely_roles: list[str], text: str) -> dict[str, float]:
    lower = text.lower()
    skill_set = set(skills)
    scores: dict[str, float] = {}
    for role, required in ROLE_SKILLS.items():
        score = 0.0
        score += len(skill_set.intersection(required)) * 1.6
        if role in likely_roles:
            score += 3.0
        if role in lower:
            score += 2.0
        if role == 'business intelligence' and any(alias in lower for alias in ['bi analyst', 'business intelligence analyst']):
            score += 1.0
        if score > 0:
            scores[role] = round(score, 2)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _infer_primary_role(skills: list[str], likely_roles: list[str], text: str = '') -> str | None:
    if likely_roles:
        return likely_roles[0]
    scores = _score_roles(skills, likely_roles, text)
    return next(iter(scores), None)


def _infer_seniority(years: int) -> str:
    for threshold, label in EXPERIENCE_SENIORITY_RULES:
        if years >= threshold:
            return label
    return 'entry-level'


def _extract_name(text: str) -> str | None:
    lines = _split_lines(text)
    if not lines:
        return None
    for candidate in lines[:5]:
        if len(candidate.split()) <= 5 and not EMAIL_RE.search(candidate) and not PHONE_RE.search(candidate):
            if re.fullmatch(r'[A-Za-z][A-Za-z .\'-]{1,80}', candidate):
                return ' '.join(part.capitalize() for part in candidate.split())
    return None


def _build_summary_evidence(text: str, skills: list[str]) -> list[str]:
    keywords = skills[:8] if skills else SKILL_KEYWORDS[:8]
    evidence = _extract_sentences(text, keywords + ['impact', 'improved', 'meningkatkan', 'membangun', 'built'], limit=6)
    if evidence:
        return evidence
    return _extract_sentences(text, ROLE_KEYWORDS + ['experience', 'berpengalaman'], limit=4)


def _extract_heuristic_profile(text: str) -> dict[str, Any]:
    sections = _build_section_map(text)
    searchable_text = '\n'.join(sections.values())
    compact_text = _normalize_whitespace(text)

    skills = _find_keywords(searchable_text, SKILL_KEYWORDS, normalize='skill')
    likely_roles = _find_keywords(searchable_text, ROLE_KEYWORDS, normalize='role')
    years = _extract_years(searchable_text)
    role_scores = _score_roles(skills, likely_roles, searchable_text)
    primary_role = _infer_primary_role(skills, likely_roles, searchable_text)

    profile = {
        'name': _extract_name(text),
        'contact': _extract_contact_info(text),
        'skills': skills,
        'likely_roles': likely_roles,
        'primary_role_guess': primary_role,
        'role_scores': role_scores,
        'education_mentions': _find_keywords(searchable_text, EDUCATION_KEYWORDS, normalize='education'),
        'education_entries': _extract_education_entries(sections.get('education', searchable_text)),
        'experience_entries': _extract_experience_entries(sections.get('experience', searchable_text)),
        'certifications': _extract_certifications(sections.get('certifications', searchable_text)),
        'languages': _extract_languages(sections.get('languages', searchable_text)),
        'years_of_experience_mentions': years['mentions'],
        'estimated_years_experience': years['max_years'],
        'seniority_guess': _infer_seniority(int(years['max_years'])),
        'headline': (_extract_sentences(sections.get('summary', searchable_text), ROLE_KEYWORDS + ['experience', 'berpengalaman'], limit=1) or [compact_text[:180]])[0],
        'strength_evidence': _build_summary_evidence(searchable_text, skills),
        'sections_found': sorted([key for key, value in sections.items() if value.strip()]),
        'section_presence': {name: bool(content.strip()) for name, content in sections.items() if content.strip()},
        'skill_count': len(skills),
        'text_excerpt': compact_text[:1200],
    }
    return profile


@lru_cache(maxsize=1)
def _get_spacy_nlp() -> Any:
    model_name = 'xx_ent_wiki_sm'
    if spacy is None:
        return None
    for candidate in [model_name, 'en_core_web_sm']:
        try:
            return spacy.load(candidate)
        except Exception:
            continue
    return None


def _extract_ner_profile(text: str) -> dict[str, Any]:
    nlp = _get_spacy_nlp()
    entities: dict[str, list[str]] = {'PERSON': [], 'ORG': [], 'GPE': [], 'LOC': []}
    if nlp is not None:
        try:
            doc = nlp(text[:20000])
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(_normalize_whitespace(ent.text))
        except Exception:
            pass

    if not any(entities.values()):
        fallback_name = _extract_name(text)
        if fallback_name:
            entities['PERSON'].append(fallback_name)
        entities['GPE'].extend([match.group(0).title() for match in LOCATION_RE.finditer(text)])
        for line in _split_lines(text):
            org_match = re.search(r'(?:at|@|company|perusahaan)\s+([A-Z][A-Za-z0-9&.,\- ]{2,60})', line)
            if org_match:
                entities['ORG'].append(_normalize_whitespace(org_match.group(1)))

    organizations = _unique_preserve(entities['ORG'])[:10]
    locations = _unique_preserve(entities['GPE'] + entities['LOC'])[:10]
    return {
        'name': _unique_preserve(entities['PERSON'])[:1][0] if _unique_preserve(entities['PERSON']) else None,
        'entity_summary': {
            'people': _unique_preserve(entities['PERSON'])[:5],
            'organizations': organizations,
            'locations': locations,
        },
        'organizations': organizations,
        'locations': locations,
        'parser_source': 'spacy_ner' if nlp is not None else 'heuristic_ner',
    }


def _extract_llm_profile(text: str) -> dict[str, Any]:
    if not llm_is_available():
        return {'enabled': False, 'reason': 'LLM not configured'}

    prompt = (
        'Ekstrak CV menjadi JSON valid. Gunakan hanya informasi yang eksplisit pada CV. '
        'Kembalikan object JSON dengan kunci: name, skills, likely_roles, primary_role_guess, '
        'education_entries, experience_entries, certifications, languages, estimated_years_experience, headline. '
        'Jangan beri markdown.\n\nCV:\n'
        + text[:12000]
    )
    try:
        result = invoke_text(prompt, temperature=0)
        if result is None:
            return {'enabled': False, 'reason': 'LLM unavailable'}
        payload = extract_json_object(result.content)
        if not payload:
            return {'enabled': True, 'parsed': False, 'reason': 'LLM did not return JSON'}
        return {'enabled': True, 'parsed': True, 'payload': payload}
    except Exception as exc:
        return {'enabled': True, 'parsed': False, 'reason': str(exc)}


def _merge_profiles(heuristic: dict[str, Any], ner: dict[str, Any], llm: dict[str, Any], raw_text: str) -> dict[str, Any]:
    profile = {**heuristic}
    if not profile.get('name') and ner.get('name'):
        profile['name'] = ner['name']

    contact = dict(profile.get('contact') or {})
    if not contact.get('location_hint') and ner.get('locations'):
        contact['location_hint'] = ner['locations'][0]
    profile['contact'] = contact

    if llm.get('parsed') and isinstance(llm.get('payload'), dict):
        payload = dict(llm['payload'])
        llm_skills = [_normalize_skill_name(str(item)) for item in payload.get('skills', []) if str(item).strip()]
        llm_roles = [_normalize_role_name(str(item)) for item in payload.get('likely_roles', []) if str(item).strip()]
        profile['skills'] = _unique_preserve([*(profile.get('skills') or []), *llm_skills])
        profile['likely_roles'] = _unique_preserve([*(profile.get('likely_roles') or []), *llm_roles])
        if payload.get('primary_role_guess') and not profile.get('primary_role_guess'):
            profile['primary_role_guess'] = _normalize_role_name(str(payload['primary_role_guess']))
        if payload.get('headline') and (not profile.get('headline') or len(str(profile['headline'])) < 30):
            profile['headline'] = str(payload['headline'])
        if payload.get('languages'):
            profile['languages'] = _unique_preserve([*(profile.get('languages') or []), *[str(item).lower() for item in payload['languages']]])
        if payload.get('certifications'):
            profile['certifications'] = _unique_preserve([*(profile.get('certifications') or []), *[str(item) for item in payload['certifications']]])
        if payload.get('experience_entries') and not profile.get('experience_entries'):
            profile['experience_entries'] = payload['experience_entries']
        if payload.get('education_entries') and not profile.get('education_entries'):
            profile['education_entries'] = payload['education_entries']
        years = payload.get('estimated_years_experience')
        if isinstance(years, int) and years > int(profile.get('estimated_years_experience', 0) or 0):
            profile['estimated_years_experience'] = years
            profile['seniority_guess'] = _infer_seniority(years)

    role_scores = _score_roles(profile.get('skills') or [], profile.get('likely_roles') or [], raw_text)
    profile['role_scores'] = role_scores
    profile['primary_role_guess'] = profile.get('primary_role_guess') or _infer_primary_role(
        profile.get('skills') or [], profile.get('likely_roles') or [], raw_text
    )
    profile['skill_count'] = len(profile.get('skills') or [])
    profile['entity_summary'] = ner.get('entity_summary', {'people': [], 'organizations': [], 'locations': []})
    profile['parser_pipeline'] = {
        'heuristic_parser': {'enabled': True},
        'ner_parser': {'enabled': True, 'source': ner.get('parser_source', 'heuristic_ner')},
        'llm_parser': {'enabled': bool(llm.get('enabled')), 'parsed': bool(llm.get('parsed', False)), 'reason': llm.get('reason')},
    }
    return profile


def _validate_cv_profile(profile: dict[str, Any], raw_text: str) -> dict[str, Any]:
    warnings: list[str] = []
    normalized_text = _normalize_whitespace(raw_text)
    completeness_points = 0

    contact = profile.get('contact') or {}
    if profile.get('name'):
        completeness_points += 1
    if contact.get('email') or contact.get('phone'):
        completeness_points += 1
    if profile.get('skills'):
        completeness_points += 1
    if profile.get('experience_entries'):
        completeness_points += 1
    if profile.get('education_entries'):
        completeness_points += 1
    if profile.get('primary_role_guess'):
        completeness_points += 1

    if len(normalized_text) < 80:
        warnings.append('Teks CV sangat pendek; sebagian informasi mungkin tidak terbaca lengkap.')
    if not (contact.get('email') or contact.get('phone')):
        warnings.append('Kontak utama belum terdeteksi dengan yakin.')
    if not profile.get('skills'):
        warnings.append('Skill eksplisit belum banyak terdeteksi.')
    if not profile.get('experience_entries') and not profile.get('estimated_years_experience'):
        warnings.append('Pengalaman kerja belum terstruktur dengan baik pada CV.')
    if contact.get('email') and not EMAIL_RE.fullmatch(str(contact['email'])):
        warnings.append('Format email terdeteksi tetapi tidak lolos validasi penuh.')

    years = int(profile.get('estimated_years_experience', 0) or 0)
    if years > 45:
        warnings.append('Estimasi tahun pengalaman tampak terlalu tinggi dan telah dibatasi.')
        years = 45
        profile['estimated_years_experience'] = years
        profile['seniority_guess'] = _infer_seniority(years)

    skills = [_normalize_skill_name(str(item)) for item in (profile.get('skills') or []) if str(item).strip()]
    profile['skills'] = _unique_preserve(skills)
    profile['likely_roles'] = _unique_preserve([_normalize_role_name(str(item)) for item in (profile.get('likely_roles') or []) if str(item).strip()])
    profile['certifications'] = _unique_preserve([str(item) for item in (profile.get('certifications') or [])])
    profile['languages'] = _unique_preserve([str(item).lower() for item in (profile.get('languages') or [])])
    profile['skill_count'] = len(profile['skills'])
    profile['primary_role_guess'] = profile.get('primary_role_guess') or _infer_primary_role(profile['skills'], profile['likely_roles'], raw_text)
    profile['role_scores'] = _score_roles(profile['skills'], profile['likely_roles'], raw_text)

    confidence = round(min(0.99, 0.2 + (completeness_points / 6) * 0.65 - len(warnings) * 0.04), 2)
    completeness_score = round((completeness_points / 6) * 100, 1)
    return {
        'is_valid': bool(normalized_text),
        'completeness_score': completeness_score,
        'parser_confidence': max(0.05, confidence),
        'warnings': warnings,
    }


def extract_cv_profile_data(cv_text: str) -> dict[str, Any]:
    raw_text = cv_text or ''
    text = raw_text.replace('\r', '\n')
    heuristic = _extract_heuristic_profile(text)
    ner = _extract_ner_profile(text)
    llm = _extract_llm_profile(text)
    profile = _merge_profiles(heuristic, ner, llm, text)
    profile['validation'] = _validate_cv_profile(profile, text)
    return profile
