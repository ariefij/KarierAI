from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _extract(prompt: str, label: str) -> str:
    match = re.search(rf"{label}:\s*(.+)", prompt)
    return match.group(1).strip() if match else ""


@pytest.fixture(autouse=True)
def patch_llm_only_flow(monkeypatch):
    from karierai.agent import ChatSynthesisResult
    from karierai.llm import LLMResult
    import karierai.agent as agent
    import karierai.database.analytics as analytics
    import karierai.llm as llm

    def fake_route_payload(query: str) -> dict:
        lowered = query.lower()
        if "gaji" in lowered or "jumlah" in lowered or "rata-rata" in lowered or "rata rata" in lowered:
            intent = "sql"
        elif "cv" in lowered and ("gap" in lowered or "cocok" in lowered or "karier" in lowered):
            intent = "consultation"
        elif "cv" in lowered or "resume" in lowered:
            intent = "cv"
        elif any(token in lowered for token in ["sekaligus", "serta"]):
            intent = "hybrid"
        else:
            intent = "rag"
        target_role = "Data Analyst" if "data analyst" in lowered else None
        return {
            "intent": intent,
            "confidence": 92,
            "search_query": query.strip(),
            "target_role": target_role,
            "reason": "mock_llm_router",
        }

    def fake_sql_payload(question: str) -> dict:
        lowered = question.lower()
        if "rata-rata gaji" in lowered or "rata rata gaji" in lowered:
            return {
                "sql": "SELECT location, AVG(salary_mid(salary_raw)) AS avg_salary FROM jobs WHERE LOWER(job_title) LIKE '%data analyst%' AND salary_mid(salary_raw) IS NOT NULL GROUP BY location ORDER BY avg_salary DESC LIMIT 50",
                "explanation": "average salary grouped by location",
            }
        if "perusahaan unik" in lowered or "jumlah perusahaan unik" in lowered:
            return {
                "sql": "SELECT COUNT(DISTINCT company_name) AS total_companies FROM jobs WHERE LOWER(job_title) LIKE '%data analyst%' LIMIT 50",
                "explanation": "distinct company count",
            }
        if "jumlah" in lowered or "berapa" in lowered:
            return {
                "sql": "SELECT COUNT(*) AS total FROM jobs WHERE LOWER(job_title) LIKE '%data analyst%' LIMIT 50",
                "explanation": "count matching jobs",
            }
        return {
            "sql": "SELECT job_id, job_title, company_name, location FROM jobs ORDER BY created_at DESC LIMIT 50",
            "explanation": "default listing",
        }

    def fake_invoke_json(prompt: str, temperature: float = 0.0):
        if "intent harus salah satu" in prompt:
            payload = fake_route_payload(_extract(prompt, "Query user"))
            return payload, LLMResult(content=str(payload), input_tokens=11, output_tokens=7)
        if "Pertanyaan user:" in prompt and '"sql"' in prompt:
            payload = fake_sql_payload(_extract(prompt, "Pertanyaan user"))
            return payload, LLMResult(content=str(payload), input_tokens=19, output_tokens=14)
        raise AssertionError(f"Prompt invoke_json tidak dikenali: {prompt[:120]}")

    def fake_invoke_text(prompt: str, temperature: float = 0.2):
        intent = _extract(prompt, "Intent terpilih")
        task = _extract(prompt, "Tugas narasi")
        if '"status": "needs_cv_text"' in prompt or "Data CV belum tersedia" in prompt or "needs_cv_text" in prompt:
            content = "Aku belum bisa kasih evaluasi yang spesifik karena teks CV-mu belum ada. Kirim teks CV lengkap atau upload file CV dulu, lalu aku bantu analisis dengan lebih akurat."
        elif task == "cv_analysis":
            content = "CV ini sudah memberi sinyal yang cukup jelas soal arah kariermu. Kekuatan utamanya ada di role yang terdeteksi, skill inti, dan pengalaman yang terlihat dari profil."
        elif task == "recommendation":
            content = "Dari CV yang kamu kirim, ada beberapa lowongan yang paling nyambung dengan profilmu. Prioritas utamanya biasanya terlihat dari kecocokan role, skill, dan konteks lokasi atau tipe kerja."
        elif task == "consultation":
            content = "Kalau targetmu adalah role ini, fokus terbaiknya adalah memperkuat gap skill yang paling penting dulu sambil menjaga kekuatan yang sudah kamu punya."
        elif "Draft jawaban:" in prompt:
            draft = prompt.split("Draft jawaban:", 1)[1].strip()
            content = draft.split("Rapikan supaya", 1)[0].strip() or "Saya sudah rapikan jawabannya supaya lebih natural, tetap jelas, dan langsung ke inti tanpa terasa kaku."
        elif intent == "sql":
            content = "Dari data lowongan yang ada, pertanyaanmu sudah saya olah dan pola utamanya sudah terlihat jelas. Hasil ini bisa dipakai untuk membandingkan lokasi, jumlah lowongan, atau tren gaji secara lebih praktis."
        elif intent == "rag":
            content = "Saya sudah cari lowongan yang paling relevan dan merangkumnya dengan bahasa yang lebih natural. Kalau mau, pencarian ini bisa dipersempit lagi berdasarkan lokasi, skill, atau tipe kerja."
        elif intent == "consultation":
            content = "Dari konteks CV dan target role, saya sudah susun masukan karier yang paling relevan. Fokus utamanya adalah memperkuat gap skill yang paling berpengaruh dulu."
        else:
            content = "Saya sudah menyusun jawaban akhir dari konteks yang tersedia dengan gaya yang lebih natural."
        return LLMResult(content=content, input_tokens=23, output_tokens=17)

    monkeypatch.setattr(llm, "require_llm", lambda: None)
    monkeypatch.setattr(llm, "invoke_json", fake_invoke_json)
    monkeypatch.setattr(llm, "invoke_text", fake_invoke_text)

    monkeypatch.setattr(agent, "require_llm", lambda: None)
    monkeypatch.setattr(agent, "invoke_json", fake_invoke_json)
    monkeypatch.setattr(agent, "invoke_text", fake_invoke_text)

    monkeypatch.setattr(analytics, "require_llm", lambda: None)
    monkeypatch.setattr(analytics, "invoke_json", fake_invoke_json)

    yield
