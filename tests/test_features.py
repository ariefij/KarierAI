from io import BytesIO

import fitz
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from karierai.database import hybrid_search_jobs, init_sqlite, run_safe_analytics
from karierai.ingestion import ingest_jobs
from karierai.server import app
from karierai.services import classify_chat_intent, extract_cv_profile_data, extract_text_from_image_bytes, extract_text_from_pdf_bytes


def setup_module() -> None:
    init_sqlite()
    ingest_jobs(limit=50)


def _make_cv_image_bytes(text: str) -> bytes:
    image = Image.new('RGB', (1400, 700), 'white')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 64)
    except Exception:
        font = ImageFont.load_default()
    draw.text((60, 220), text, fill='black', font=font)
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def _make_scanned_pdf_bytes(text: str) -> bytes:
    png_bytes = _make_cv_image_bytes(text)
    image = Image.open(BytesIO(png_bytes))
    document = fitz.open()
    page = document.new_page(width=image.width, height=image.height)
    page.insert_image(page.rect, stream=png_bytes)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_cv_analyze_and_recommend() -> None:
    client = TestClient(app)
    cv_text = 'Data analyst with 3 years experience using SQL, Python, Tableau, Power BI.'
    analyze = client.post('/cv/analyze', json={'cv_text': cv_text})
    assert analyze.status_code == 200
    analyze_payload = analyze.json()
    assert 'skills' in analyze_payload['profile']
    assert analyze_payload['response_text']

    rec = client.post('/recommend', json={'cv_text': cv_text, 'top_k': 3})
    assert rec.status_code == 200
    rec_payload = rec.json()
    assert 'matches' in rec_payload
    assert rec_payload['response_text']


def test_consult() -> None:
    client = TestClient(app)
    payload = {'cv_text': 'I use SQL, Python, dashboarding and statistics.', 'target_role': 'Data Scientist'}
    response = client.post('/consult', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data['target_role'] == 'Data Scientist'
    assert 'recommendations' in data
    assert data['response_text']


def test_extract_text_from_image_bytes_ocr() -> None:
    text = extract_text_from_image_bytes(_make_cv_image_bytes('Data Analyst SQL Python 5 years'))
    lowered = text.lower()
    assert 'data analyst' in lowered
    assert 'sql' in lowered
    assert 'python' in lowered


def test_extract_text_from_scanned_pdf_bytes_ocr() -> None:
    text = extract_text_from_pdf_bytes(_make_scanned_pdf_bytes('Data Analyst SQL Python 5 years'))
    lowered = text.lower()
    assert 'data analyst' in lowered
    assert 'sql' in lowered
    assert 'python' in lowered


def test_cv_file_upload_endpoints_support_pdf_and_image() -> None:
    client = TestClient(app)
    image_bytes = _make_cv_image_bytes('Data Analyst SQL Python 2 years')
    pdf_bytes = _make_scanned_pdf_bytes('Data Analyst SQL Python 2 years')

    image_analyze = client.post('/cv/analyze-file', files={'file': ('cv.png', image_bytes, 'image/png')})
    assert image_analyze.status_code == 200
    image_analyze_payload = image_analyze.json()
    assert 'sql' in image_analyze_payload['profile']['skills']
    assert image_analyze_payload['response_text']

    pdf_recommend = client.post('/recommend-file', files={'file': ('cv.pdf', pdf_bytes, 'application/pdf')}, data={'top_k': '3'})
    assert pdf_recommend.status_code == 200
    pdf_recommend_payload = pdf_recommend.json()
    assert 'matches' in pdf_recommend_payload
    assert pdf_recommend_payload['response_text']

    image_consult = client.post('/consult-file', files={'file': ('cv.jpg', image_bytes, 'image/jpeg')}, data={'target_role': 'Data Analyst'})
    assert image_consult.status_code == 200
    image_consult_payload = image_consult.json()
    assert image_consult_payload['target_role'] == 'Data Analyst'
    assert image_consult_payload['response_text']


def test_flexible_text2sql_salary_grouping() -> None:
    result = run_safe_analytics('Berapa rata-rata gaji data analyst per lokasi?')
    assert result['mode'] in {'heuristic_text2sql', 'llm_text2sql'}
    assert 'avg_salary' in result['sql'].lower()
    assert 'group by location' in result['sql'].lower()
    assert isinstance(result['rows'], list)


def test_flexible_text2sql_distinct_companies() -> None:
    result = run_safe_analytics('Berapa jumlah perusahaan unik untuk lowongan data analyst?')
    assert 'count(distinct company_name)' in result['sql'].lower()
    assert result['rows']


def test_structured_cv_profile_and_classifier() -> None:
    cv_text = """
    Jane Doe
    jane.doe@example.com | +6281234567890 | linkedin.com/in/janedoe

    Summary
    Data Analyst with 4 years experience using SQL, Python, Tableau, and Power BI.

    Experience
    Data Analyst at PT Maju Jaya
    Jan 2021 - Present
    Built dashboards and automated reporting using SQL and Python.

    Education
    Bachelor of Statistics, University of Indonesia, 2020

    Certifications
    Google Data Analytics Professional Certificate
    """
    profile = extract_cv_profile_data(cv_text)
    assert profile['contact']['email'] == 'jane.doe@example.com'
    assert profile['primary_role_guess'] in {'data analyst', 'business intelligence'}
    assert profile['experience_entries']
    assert profile['education_entries']
    assert 'certifications' in profile

