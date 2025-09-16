from fastapi.testclient import TestClient

from ai_research_agent.src.api.main import app


def test_health():
	client = TestClient(app)
	r = client.get('/health')
	assert r.status_code == 200
	assert r.json().get('status') == 'ok' 