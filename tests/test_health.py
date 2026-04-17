from fastapi.testclient import TestClient

from karierai.server import app


client = TestClient(app)


def test_health() -> None:
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_ready_exposes_runtime_status() -> None:
    response = client.get('/ready')
    assert response.status_code == 200
    payload = response.json()
    assert 'ocr_ready' in payload
    assert 'database_stats' in payload
    assert 'jobs_count' in payload['database_stats']
