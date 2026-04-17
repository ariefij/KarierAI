from fastapi.testclient import TestClient

from karierai.server import app


def test_chat_fallback() -> None:
    client = TestClient(app)
    response = client.post('/chat', json={'query': 'Cari lowongan data analyst di Jakarta', 'history': ''})
    assert response.status_code == 200
    payload = response.json()
    assert 'response' in payload
    assert 'used_tools' in payload


def test_chat_sql_path() -> None:
    client = TestClient(app)
    response = client.post('/chat', json={'query': 'Berapa rata-rata gaji data analyst per lokasi?', 'history': ''})
    assert response.status_code == 200
    payload = response.json()
    assert any(tool in payload['used_tools'] for tool in ['sql_query_jobs', 'route_task'])


def test_chat_validation_rejects_blank_query() -> None:
    client = TestClient(app)
    response = client.post('/chat', json={'query': '   ', 'history': ''})
    assert response.status_code == 422



def test_chat_cv_mention_without_cv_text_does_not_trigger_recommendation() -> None:
    client = TestClient(app)
    response = client.post('/chat', json={'query': 'CV saya cocok untuk lowongan apa?', 'history': ''})
    assert response.status_code == 200
    payload = response.json()
    assert 'build_recommendations' not in payload['used_tools']
    assert 'upload file cv' in payload['response'].lower() or 'kirim teks cv lengkap' in payload['response'].lower()
