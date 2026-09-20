"""Browser chat page stays separate from the Ollama root probe."""

from fastapi.testclient import TestClient

from vmlx_engine import server
from vmlx_engine.web_ui import CHAT_HTML_PATH, wants_browser_page


class _FakeRequest:
    def __init__(self, accept: str):
        self.headers = {"accept": accept}


def test_browser_accept_gets_html_and_clients_still_see_ollama(monkeypatch):
    monkeypatch.setattr(server, "_api_key", "secret-token", raising=False)
    monkeypatch.setattr(server, "_model_name", "mlx-community/Qwen3.8-27B-4bit", raising=False)
    client = TestClient(server.app)

    probe = client.get("/", headers={"Accept": "*/*"})
    assert probe.status_code == 200
    assert "Ollama" in probe.text

    page = client.get("/", headers={"Accept": "text/html"})
    assert page.status_code == 200
    assert "Local Qwen" in page.text
    assert "text/html" in page.headers.get("content-type", "")

    chat = client.get("/chat")
    assert chat.status_code == 200
    assert "Local Qwen" in chat.text


def test_bootstrap_gives_loopback_key_only(monkeypatch):
    monkeypatch.setattr(server, "_api_key", "vmlx_test_key", raising=False)
    monkeypatch.setattr(server, "_model_name", "demo-model", raising=False)
    monkeypatch.setattr("vmlx_engine.web_ui.is_loopback_bind", lambda host: True)
    client = TestClient(server.app)

    data = client.get("/chat/bootstrap").json()
    assert data["model"] == "demo-model"
    assert data["api_key"] == "vmlx_test_key"


def test_html_file_exists():
    assert CHAT_HTML_PATH.is_file()


def test_wants_browser_page():
    assert wants_browser_page(_FakeRequest("text/html,application/xhtml+xml"))
    assert not wants_browser_page(_FakeRequest("*/*"))
