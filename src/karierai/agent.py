from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .database import hybrid_search_jobs, run_safe_analytics
from .llm import invoke_json, invoke_text, require_llm
from .services import build_career_consultation, extract_cv_profile_data, extract_target_role

PROMPTS = {
    "chat_query_router": (
        "Anda adalah router intent untuk KarierAI. Pahami maksud user lalu pilih satu intent terbaik: rag, sql, cv, consultation, atau hybrid. "
        "Utamakan intent yang benar-benar paling membantu, jangan pilih hybrid kalau user hanya butuh satu jalur. "
        "Kalau user menanyakan lowongan, buat search_query yang lebih ringkas dan siap dipakai retrieval. "
        "Balas hanya JSON valid."
    ),
    "chat_response_writer": (
        "Anda adalah KarierAI, asisten karier berbahasa Indonesia yang terdengar natural. "
        "Tulis jawaban yang hangat, jelas, langsung ke inti, dan tidak kaku. "
        "Gunakan hanya evidence yang diberikan. Jangan tampilkan JSON mentah, jangan menyebut tool internal, "
        "dan jangan mengarang fakta baru. Bila data kurang, katakan secara natural apa yang belum tersedia."
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



def _format_search_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Tidak ada lowongan relevan ditemukan."
    blocks: list[str] = []
    for index, row in enumerate(rows, start=1):
        retrieval_sources = row.get("retrieval_sources", [])
        source_label = ", ".join(retrieval_sources) if isinstance(retrieval_sources, list) else "hybrid"
        blocks.append(
            "\n".join(
                [
                    f"Result {index}",
                    f"source: {source_label or 'hybrid'}",
                    f"job_id: {row.get('job_id', 'N/A')}",
                    f"title: {row.get('job_title', 'N/A')}",
                    f"company: {row.get('company_name', 'N/A')}",
                    f"location: {row.get('location', 'N/A')}",
                    f"work_type: {row.get('work_type', 'N/A')}",
                    f"hybrid_score: {row.get('hybrid_score', row.get('relevance_score', 0))}",
                    f"snippet: {str(row.get('job_description', ''))[:500]}",
                ]
            )
        )
    return "\n\n".join(blocks)



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
        sql_output = json.dumps(run_safe_analytics(query), ensure_ascii=False)
        return ChatExecutionResult([rag_output, sql_output], ["rag_search_jobs", "sql_query_jobs"], [rag_output, sql_output])

    if plan.intent == "sql":
        sql_output = json.dumps(run_safe_analytics(query), ensure_ascii=False)
        return ChatExecutionResult([sql_output], ["sql_query_jobs"], [sql_output])

    if plan.intent == "rag":
        rag_output = _search_jobs(plan.search_query, k=5)
        return ChatExecutionResult([rag_output], ["rag_search_jobs"], [rag_output])

    cv_text = _resolve_cv_text(query, history)
    if not cv_text:
        missing_cv_payload = json.dumps(
            {
                "status": "needs_cv_text",
                "message": "User belum memberikan teks CV yang cukup untuk dianalisis.",
                "guidance": "Minta user mengirim teks CV lengkap atau upload file CV lewat endpoint file.",
                "target_role": plan.target_role or extract_target_role(query),
            },
            ensure_ascii=False,
        )
        return ChatExecutionResult([missing_cv_payload], [], [missing_cv_payload])

    if plan.intent == "cv":
        profile_payload = json.dumps(extract_cv_profile_data(cv_text), ensure_ascii=False)
        return ChatExecutionResult([profile_payload], ["extract_cv_profile"], [profile_payload])

    consultation_payload = json.dumps(
        build_career_consultation(cv_text, plan.target_role or extract_target_role(query) or "Data Analyst"),
        ensure_ascii=False,
    )
    return ChatExecutionResult([consultation_payload], ["analyze_skill_gap"], [consultation_payload])



def _compose_response(query: str, history: str, plan: ChatRoutePlan, evidence_blocks: list[str]) -> ChatSynthesisResult:
    require_llm()
    evidence = "\n\n".join(f"[Evidence {index}]\n{block}" for index, block in enumerate(evidence_blocks, start=1))
    writer_prompt = get_prompt("chat_response_writer") + (
        f"\n\nIntent terpilih: {plan.intent}\n"
        f"Query asli user: {query}\n"
        f"Query kerja yang dipakai: {plan.search_query}\n"
        f"History ringkas: {history or '-'}\n"
        f"Confidence router: {plan.confidence}\n"
        f"Alasan router: {plan.reason or '-'}\n\n"
        f"Evidence:\n{evidence or '-'}"
    )
    result = invoke_text(writer_prompt, temperature=0.35)
    response = result.content.strip()
    if not response:
        raise RuntimeError("LLM writer tidak mengembalikan jawaban.")
    return ChatSynthesisResult(response=response, input_tokens=result.input_tokens, output_tokens=result.output_tokens)



def local_chat_response(query: str, history: str = "") -> dict[str, Any]:
    plan, input_tokens, output_tokens, route_message = _route_chat(query, history)
    execution = _collect_evidence(plan, query, history)
    synthesis = _compose_response(query, history, plan, execution.evidence_blocks)
    return {
        "response": synthesis.response,
        "input_tokens": input_tokens + synthesis.input_tokens,
        "output_tokens": output_tokens + synthesis.output_tokens,
        "tool_messages": [route_message, *execution.tool_messages],
        "used_tools": ["llm_query_router", *execution.used_tools, "llm_response_writer"],
    }
