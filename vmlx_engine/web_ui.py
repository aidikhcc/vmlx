# SPDX-License-Identifier: Apache-2.0
"""Small browser chat page for the local vMLX server.

The API at port 8000 is not a website by itself. These routes give Safari
and Chrome a real chat window, while Ollama-style clients still see the
plain-text root probe.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from .http_security import is_loopback_bind

CHAT_HTML_PATH = Path(__file__).resolve().parent / "web_ui_chat.html"


def wants_browser_page(request: Request) -> bool:
    """True when a web browser asked for a page, not an API probe."""
    accept = (request.headers.get("accept") or "").lower()
    return "text/html" in accept


def chat_html_response() -> FileResponse:
    """Return the local chat page."""
    if not CHAT_HTML_PATH.is_file():
        raise FileNotFoundError(f"Chat page missing: {CHAT_HTML_PATH}")
    return FileResponse(CHAT_HTML_PATH, media_type="text/html; charset=utf-8")


def register_web_chat(app: FastAPI) -> None:
    """Attach /chat and a same-Mac bootstrap helper to the FastAPI app."""

    @app.get("/chat", include_in_schema=False)
    async def chat_page() -> FileResponse:
        return chat_html_response()

    @app.get("/chat/bootstrap", include_in_schema=False)
    async def chat_bootstrap(request: Request) -> JSONResponse:
        # Imported here to avoid a circular import at module load time.
        from vmlx_engine import server as srv

        client_host = request.client.host if request.client else ""
        api_key = None
        if is_loopback_bind(client_host):
            api_key = srv._api_key
        return JSONResponse(
            {
                "model": srv._resolve_model_name(),
                "ready": bool(getattr(srv, "_engine", None)),
                "api_key": api_key,
            }
        )
