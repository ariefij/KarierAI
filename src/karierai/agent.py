from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .database import hybrid_search_jobs, run_safe_analytics
from .llm import invoke_json, invoke_text, require_llm
from .services import build_career_consultation, build_recommendations, extract_cv_profile_data, extract_target_role

PROMPTS = {
    "chat_query_router": (
        "Anda adalah router intent untuk KarierAI. Pahami maksud user lalu pilih satu intent terbaik: rag, sql, cv, consultation, atau hybrid. "
        "Utamakan intent yang benar-benar paling membantu, jangan pilih hybrid kalau user hanya butuh satu jalur. "
        "Kalau user menanyakan lowongan, buat search_query yang lebih ringkas dan siap dipakai retrieval. "
        "Balas hanya JSON valid."
    ),
    "chat_response_writer": (
        "Anda adalah KarierAI, asisten karier berbahasa Indonesia yang terdengar hangat, natural, dan profesional. "
        "Tulis seperti percakapan manusia yang enak dibaca, bukan seperti dump database, laporan robotik, atau hasil tool. "
        "Selalu mulai dari inti jawaban lebih dulu, lalu beri detail penting seperlunya. "
        "Gunakan kalimat yang rapi dengan ritme natural. Hindari label teknis seperti JSON, field internal, evidence, tool, hybrid_score, atau job_id kecuali benar-benar perlu. "
        "Kalau membahas lowongan, sebutkan nama posisi, perusahaan, lokasi, dan alasan singkat kenapa relevan. "
        "Kalau membahas analisis CV atau konsultasi, jelaskan insight utama, kekuatan, gap, dan langkah lanjut dengan bahasa yang membumi. "
        "Kalau data belum cukup, sampaikan dengan natural dan beri arahan berikutnya."
    ),
    "natural_response_rewriter": (
        "Anda bertugas memoles jawaban agar terdengar seperti asisten karier Indonesia yang natural. "
        "Pertahankan fakta inti, tetapi perbaiki diksi, alur, dan transisi supaya lebih cair, tidak kaku, dan tidak terasa seperti hasil template. "
        "Jangan menambah fakta baru."
    ),
    "endpoint_response_writer": (
        "Anda adalah KarierAI. Ubah data terstruktur menjadi jawaban bahasa Indonesia yang natural, ringkas, dan jelas. "
        "Jangan menyalin nama field mentah. Fokus pada insight yang paling berguna bagi user."
    ),
}

_ALLOWED_INTENTS = {"rag", "sql", "cv", "consultation", "hybrid"}
_CV_HINT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        r"\b(linkedin|github)\b",
        r"\b(summary|experience|education|skills|certifications?|projects?)\b",
        r"\b(python|sql|tableau|power\s*bi|excel|machine learning|statistics?)\b",
        r"\b(data analyst|data scientist|business analyst|product manager|software engineer)\b",
        r"\b\d+\s*(years?|tahun)\b",
    ]
]


@dataclass(frozen=True)
class ChatRoutePlan:
    intent: str
    search_query: str
    target_role: str | None = None
    confidence: int = 0
    reason: str = ""


@dataclass(frozen=True)
class ChatExecutionResult:
    evidence_blocks: list[str]
    used_tools: list[str]
    tool_messages: list[str]


@dataclass(frozen=True)
class ChatSynthesisResult:
    response: str
    input_tokens: int = 0
    output_tokens: int = 0



def get_prompt(prompt_name: str) -> str:
    return PROMPTS.get(prompt_name, "Anda adalah asisten yang membantu user dengan jelas dan natural.")



def _looks_like_cv_text(text: str) -> bool:
    normalized = " ".join((text or "").split())
    if len(normalized) < 80:
        return False
    return sum(1 for pattern in _CV_HINT_PATTERNS if pattern.search(normalized)) >= 2



def _resolve_cv_text(query: str, history: str = "") -> str:
    history_text = (history or "").strip()
    query_text = (query or "").strip()
    if _looks_like_cv_text(history_text):
        return history_text
    if _looks_like_cv_text(query_text):
        return query_text
    return ""



def _to_text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items) if items else "-"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    return text or "-"



def _format_search_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Tidak ada lowongan relevan ditemukan."
    blocks: list[str] = ["Ringkasan lowongan yang ditemukan:"]
    for index, row in enumerate(rows, start=1):
        reasons: list[str] = []
        work_type = str(row.get("work_type") or "").strip()
        if work_type:
            reasons.append(work_type)
        location = str(row.get("location") or "").strip()
        if location:
            reasons.append(location)
        if row.get("retrieval_sources"):
            reasons.append(f"sumber: {_to_text(row.get('retrieval_sources'))}")
        snippet = " ".join(str(row.get("job_description") or "").split())[:280]
        blocks.append(
            "\n".join(
                [
                    f"{index}. {row.get('job_title', 'Lowongan tanpa judul')} di {row.get('company_name', 'Perusahaan tidak diketahui')}",
                    f"   Lokasi/detail: {' | '.join(reasons) if reasons else '-'}",
                    f"   Alasan relevan: {snippet or 'Deskripsi singkat belum tersedia.'}",
                ]
            )
        )
    return "\n".join(blocks)



def _format_analytics_result(result: dict[str, Any]) -> str:
    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    lines = [
        "Ringkasan hasil analisis data lowongan:",
        f"- Mode: {_to_text(result.get('mode'))}",
        f"- SQL tervalidasi: {_to_text(result.get('sql'))}",
        f"- Penjelasan: {_to_text(result.get('explanation'))}",
        f"- Jumlah baris hasil: {len(rows)}",
    ]
    if rows:
        lines.append("- Contoh hasil teratas:")
        for idx, row in enumerate(rows[:5], start=1):
            if isinstance(row, dict):
                summary = "; ".join(f"{key}={_to_text(value)}" for key, value in row.items())
            else:
                summary = _to_text(row)
            lines.append(f"  {idx}. {summary}")
    else:
        lines.append("- Tidak ada baris hasil yang bisa diringkas.")
    return "\n".join(lines)



def _format_cv_profile(profile: dict[str, Any]) -> str:
    contact = profile.get("contact") if isinstance(profile.get("contact"), dict) else {}
    validation = profile.get("validation") if isinstance(profile.get("validation"), dict) else {}
    lines = [
        "Ringkasan profil CV:",
        f"- Role utama yang terdeteksi: {_to_text(profile.get('primary_role_guess'))}",
        f"- Role yang mungkin cocok: {_to_text(profile.get('likely_roles'))}",
        f"- Estimasi pengalaman: {_to_text(profile.get('estimated_years_experience'))} tahun",
        f"- Skill utama: {_to_text((profile.get('skills') or [])[:10])}",
        f"- Email: {_to_text(contact.get('email'))}",
        f"- Kelengkapan profil: {_to_text(validation.get('completeness_score'))}",
    ]
    strengths = profile.get("strengths") if isinstance(profile.get("strengths"), list) else []
    if strengths:
        lines.append(f"- Kekuatan yang tampak: {_to_text(strengths[:5])}")
    return "\n".join(lines)



def _format_consultation_payload(payload: dict[str, Any]) -> str:
    market = payload.get("market_summary") if isinstance(payload.get("market_summary"), dict) else {}
    lines = [
        "Ringkasan konsultasi karier:",
        f"- Target role: {_to_text(payload.get('target_role'))}",
        f"- Skill yang sudah cocok: {_to_text(payload.get('matched_skills'))}",
        f"- Skill yang masih kurang: {_to_text(payload.get('missing_skills'))}",
        f"- Judul lowongan yang sering muncul: {_to_text(market.get('sample_titles'))}",
        f"- Lokasi pasar teratas: {_to_text(market.get('top_locations'))}",
        f"- Saran aksi: {_to_text(payload.get('recommendations'))}",
    ]
    return "\n".join(lines)



def _format_recommendation_payload(payload: dict[str, Any]) -> str:
    profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
    matches = payload.get("matches") if isinstance(payload.get("matches"), list) else []
    lines = [
        "Ringkasan rekomendasi lowongan dari CV:",
        f"- Query pencarian: {_to_text(payload.get('search_query'))}",
        f"- Role utama CV: {_to_text(profile.get('primary_role_guess'))}",
        f"- Skill CV yang paling menonjol: {_to_text((profile.get('skills') or [])[:8])}",
        f"- Jumlah match yang diringkas: {len(matches)}",
    ]
    if matches:
        lines.append("- Match teratas:")
        for idx, item in enumerate(matches[:5], start=1):
            if not isinstance(item, dict):
                continue
            lines.append(
                f"  {idx}. {item.get('job_title', 'Lowongan')} di {item.get('company_name', 'Perusahaan')} | "
                f"lokasi: {_to_text(item.get('location'))} | alasan: {_to_text(item.get('explanation'))}"
            )
    return "\n".join(lines)



def _search_jobs(query: str, k: int = 5) -> str:
    return _format_search_rows(hybrid_search_jobs(search_query=query, limit=k))



def _route_chat(query: str, history: str = "") -> tuple[ChatRoutePlan, int, int, str]:
    require_llm()
    router_prompt = get_prompt("chat_query_router") + (
        f"\n\nQuery user: {query}\n"
        f"History ringkas: {history or '-'}\n\n"
        "Balas JSON valid saja dengan kunci: intent, confidence, search_query, target_role, reason.\n"
        "intent harus salah satu dari: rag, sql, cv, consultation, hybrid.\n"
        "search_query dipakai untuk retrieval lowongan dan boleh lebih ringkas dari pertanyaan user.\n"
        "target_role diisi bila memang jelas, selain itu null.\n"
        "confidence berupa integer 0-100."
    )
    payload, result = invoke_json(router_prompt, temperature=0)
    intent = str(payload.get("intent", "")).strip().lower()
    if intent not in _ALLOWED_INTENTS:
        raise RuntimeError(f"Intent hasil router tidak valid: {intent!r}")

    target_role_raw = payload.get("target_role")
    plan = ChatRoutePlan(
        intent=intent,
        search_query=str(payload.get("search_query") or query).strip() or query,
        target_role=None if target_role_raw in (None, "", "null") else str(target_role_raw).strip(),
        confidence=max(0, min(100, int(payload.get("confidence", 0) or 0))),
        reason=str(payload.get("reason") or "llm_router").strip(),
    )
    return plan, result.input_tokens, result.output_tokens, json.dumps(payload, ensure_ascii=False)



def _collect_evidence(plan: ChatRoutePlan, query: str, history: str = "") -> ChatExecutionResult:
    if plan.intent == "hybrid":
        rag_output = _search_jobs(plan.search_query, k=3)
        sql_output = _format_analytics_result(run_safe_analytics(query))
        return ChatExecutionResult([rag_output, sql_output], ["rag_search_jobs", "sql_query_jobs"], [rag_output, sql_output])

    if plan.intent == "sql":
        sql_output = _format_analytics_result(run_safe_analytics(query))
        return ChatExecutionResult([sql_output], ["sql_query_jobs"], [sql_output])

    if plan.intent == "rag":
        rag_output = _search_jobs(plan.search_query, k=5)
        return ChatExecutionResult([rag_output], ["rag_search_jobs"], [rag_output])

    cv_text = _resolve_cv_text(query, history)
    if not cv_text:
        missing_cv_message = (
            "Data CV belum tersedia. User belum memberikan teks CV yang cukup untuk dianalisis. "
            "Minta user mengirim teks CV lengkap atau upload file CV. "
            f"Kalau target role sudah terlihat, role yang diduga: {plan.target_role or extract_target_role(query) or '-'}"
        )
        return ChatExecutionResult([missing_cv_message], [], [missing_cv_message])

    if plan.intent == "cv":
        profile_payload = _format_cv_profile(extract_cv_profile_data(cv_text))
        return ChatExecutionResult([profile_payload], ["extract_cv_profile"], [profile_payload])

    consultation_payload = _format_consultation_payload(
        build_career_consultation(cv_text, plan.target_role or extract_target_role(query) or "Data Analyst")
    )
    return ChatExecutionResult([consultation_payload], ["analyze_skill_gap"], [consultation_payload])



def _compose_response(query: str, history: str, plan: ChatRoutePlan, evidence_blocks: list[str]) -> ChatSynthesisResult:
    require_llm()
    evidence = "\n\n".join(f"[Konteks {index}]\n{block}" for index, block in enumerate(evidence_blocks, start=1))
    writer_prompt = get_prompt("chat_response_writer") + (
        f"\n\nIntent terpilih: {plan.intent}\n"
        f"Query asli user: {query}\n"
        f"Query kerja yang dipakai: {plan.search_query}\n"
        f"History ringkas: {history or '-'}\n"
        f"Confidence router: {plan.confidence}\n"
        f"Alasan router: {plan.reason or '-'}\n\n"
        "Tulis jawaban akhir dalam bahasa Indonesia natural dengan aturan berikut:\n"
        "1. Paragraf pertama harus langsung menjawab inti pertanyaan user.\n"
        "2. Bila relevan, lanjutkan dengan poin atau paragraf pendek yang mudah dipindai.\n"
        "3. Jangan menyalin label teknis atau struktur mentah dari konteks.\n"
        "4. Kalau konteks berisi beberapa lowongan, pilih yang paling relevan dan jelaskan singkat.\n"
        "5. Tutup dengan langkah lanjut yang membantu bila diperlukan.\n\n"
        f"Konteks pendukung:\n{evidence or '-'}"
    )
    draft = invoke_text(writer_prompt, temperature=0.55)
    draft_response = draft.content.strip()
    if not draft_response:
        raise RuntimeError("LLM writer tidak mengembalikan jawaban.")

    rewrite_prompt = get_prompt("natural_response_rewriter") + (
        f"\n\nPertanyaan user: {query}\n"
        f"Draft jawaban:\n{draft_response}\n\n"
        "Rapikan supaya lebih cair dan natural, tetap ringkas, dan pertahankan makna inti."
    )
    rewritten = invoke_text(rewrite_prompt, temperature=0.45)
    final_response = rewritten.content.strip() or draft_response
    return ChatSynthesisResult(
        response=final_response,
        input_tokens=draft.input_tokens + rewritten.input_tokens,
        output_tokens=draft.output_tokens + rewritten.output_tokens,
    )



def write_natural_endpoint_response(task: str, payload: dict[str, Any]) -> tuple[str, int, int]:
    require_llm()
    if task == "cv_analysis":
        context_block = _format_cv_profile(payload.get("profile") if isinstance(payload.get("profile"), dict) else payload)
    elif task == "recommendation":
        context_block = _format_recommendation_payload(payload)
    elif task == "consultation":
        context_block = _format_consultation_payload(payload)
    else:
        context_block = json.dumps(payload, ensure_ascii=False)

    prompt = get_prompt("endpoint_response_writer") + (
        f"\n\nTugas narasi: {task}\n"
        "Tulis jawaban natural dalam bahasa Indonesia. Boleh 1-3 paragraf pendek atau poin singkat bila memang membantu.\n"
        "Mulai dari insight yang paling penting, lalu jelaskan detail penting seperlunya.\n"
        f"\nKonteks:\n{context_block}"
    )
    result = invoke_text(prompt, temperature=0.6)
    text = result.content.strip()
    if not text:
        raise RuntimeError("LLM endpoint writer tidak mengembalikan jawaban.")
    return text, result.input_tokens, result.output_tokens



def local_chat_response(query: str, history: str = "") -> dict[str, Any]:
    plan, input_tokens, output_tokens, route_message = _route_chat(query, history)
    execution = _collect_evidence(plan, query, history)
    synthesis = _compose_response(query, history, plan, execution.evidence_blocks)
    return {
        "response": synthesis.response,
        "input_tokens": input_tokens + synthesis.input_tokens,
        "output_tokens": output_tokens + synthesis.output_tokens,
        "tool_messages": [route_message, *execution.tool_messages],
        "used_tools": ["llm_query_router", *execution.used_tools, "llm_response_writer", "llm_response_rewriter"],
    }
