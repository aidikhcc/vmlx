# SPDX-License-Identifier: Apache-2.0
"""HTTP and bind-time security helpers for the vMLX serve path.

These controls are the AIDI-aligned defaults: require an API key, keep the
browser origin list tight, refuse private-network media fetches, and rate-limit
inference by default.
"""

from __future__ import annotations

import ipaddress
import os
import secrets
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Default CORS policy: only pages on this Mac (any localhost port).
DEFAULT_ALLOWED_ORIGINS = "loopback"
LOOPBACK_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"

# Healthcare-friendly default: 60 requests per minute per client.
DEFAULT_RATE_LIMIT = 60

# Audio uploads for /v1/audio/transcriptions.
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024

# Hostnames that must never be fetched as image/video URLs.
_BLOCKED_MEDIA_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.google.com",
}


class MediaUrlError(ValueError):
    """Raised when an image/video URL is not safe to fetch."""


class ServeSecurityError(ValueError):
    """Raised when serve flags would start the API in an unsafe bind mode."""


@dataclass(frozen=True)
class ServeSecurityPlan:
    """Resolved security settings after CLI/env defaults are applied."""

    api_key: str | None
    generated_api_key: bool
    allowed_origins: str
    rate_limit: int
    trust_forwarded_for: bool
    trust_remote_code: bool
    ssl_certfile: str | None
    ssl_keyfile: str | None


def is_loopback_bind(host: str | None) -> bool:
    """Return True when the listen address is this computer only."""
    if not host:
        return False
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1"}


def generate_api_key() -> str:
    """Create a random local API key. This is not an OpenAI or HF token."""
    return "vmlx_" + secrets.token_urlsafe(24)


def is_local_model_path(model_name: str | None) -> bool:
    """Return True when the model argument is an existing local directory or file."""
    if not model_name:
        return False
    try:
        path = Path(model_name).expanduser()
    except (OSError, RuntimeError):
        return False
    return path.exists()


def resolve_trust_remote_code(
    *,
    trust_remote_code: bool | None,
    no_trust_remote_code: bool,
    model_name: str | None,
) -> bool:
    """Decide whether Hugging Face remote Python in a model may run.

    Explicit flags win. Otherwise local folders stay allowed (the files are
    already on disk) and Hugging Face hub IDs stay blocked until the operator
    passes --trust-remote-code.
    """
    if no_trust_remote_code:
        return False
    if trust_remote_code:
        return True
    return is_local_model_path(model_name)


def apply_trust_remote_code_env(enabled: bool) -> None:
    """Publish the serve-time choice so tokenizer/loader helpers can read it."""
    os.environ["VMLX_TRUST_REMOTE_CODE"] = "1" if enabled else "0"


def env_trust_remote_code(default: bool = False) -> bool:
    """Read VMLX_TRUST_REMOTE_CODE. Missing env keeps *default*."""
    raw = os.environ.get("VMLX_TRUST_REMOTE_CODE")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _has_tls(ssl_certfile: str | None, ssl_keyfile: str | None) -> bool:
    return bool(ssl_certfile and ssl_keyfile)


def resolve_serve_security(
    *,
    host: str,
    api_key: str | None,
    env_api_key: str | None,
    allow_unauthenticated: bool,
    allow_insecure_lan: bool,
    allowed_origins: str | None,
    rate_limit: int,
    trust_forwarded_for: bool,
    trust_remote_code: bool | None,
    no_trust_remote_code: bool,
    model_name: str | None,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
    uds: str | None = None,
) -> ServeSecurityPlan:
    """Validate bind/auth flags and fill in secure defaults."""
    resolved_key = (api_key or env_api_key or "").strip() or None
    generated = False

    if allow_unauthenticated:
        if not uds and not is_loopback_bind(host):
            raise ServeSecurityError(
                "--allow-unauthenticated is only permitted with --host 127.0.0.1 "
                "(or --uds). Binding to the LAN without an API key is blocked."
            )
        resolved_key = None
    elif resolved_key is None:
        resolved_key = generate_api_key()
        generated = True

    if (
        not uds
        and not is_loopback_bind(host)
        and not _has_tls(ssl_certfile, ssl_keyfile)
        and not allow_insecure_lan
    ):
        raise ServeSecurityError(
            f"Refusing to bind {host} without TLS. Pass --ssl-certfile and "
            "--ssl-keyfile, or --allow-insecure-lan for a trusted private network."
        )

    origins = (allowed_origins or "").strip() or DEFAULT_ALLOWED_ORIGINS
    if rate_limit < 0:
        raise ServeSecurityError("--rate-limit must be >= 0")

    return ServeSecurityPlan(
        api_key=resolved_key,
        generated_api_key=generated,
        allowed_origins=origins,
        rate_limit=rate_limit,
        trust_forwarded_for=trust_forwarded_for,
        trust_remote_code=resolve_trust_remote_code(
            trust_remote_code=trust_remote_code,
            no_trust_remote_code=no_trust_remote_code,
            model_name=model_name,
        ),
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )


def configure_cors(app: Any, allowed_origins: str) -> None:
    """Attach FastAPI CORS rules. Default is localhost pages only."""
    from fastapi.middleware.cors import CORSMiddleware

    raw = [part.strip() for part in (allowed_origins or "").split(",") if part.strip()]
    if not raw or raw == ["loopback"]:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[],
            allow_origin_regex=LOOPBACK_ORIGIN_REGEX,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        return

    has_wildcard = "*" in raw
    explicit = [origin for origin in raw if origin != "*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if has_wildcard else explicit,
        allow_origin_regex=None if has_wildcard else LOOPBACK_ORIGIN_REGEX,
        allow_credentials=not has_wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def rate_limit_client_id(request: Any, trust_forwarded_for: bool) -> str:
    """Identify a client for rate limiting. Do not trust X-Forwarded-For by default."""
    if trust_forwarded_for:
        forwarded = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        if forwarded:
            return forwarded
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_outbound_media_url(url: str) -> None:
    """Block image/video fetches that can reach this Mac or an internal network."""
    if not url or not isinstance(url, str):
        raise MediaUrlError("Media URL is empty")

    parsed = urlparse(url.strip())
    if parsed.scheme.lower() != "https":
        raise MediaUrlError("Remote image/video URLs must use https://")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise MediaUrlError("Media URL is missing a hostname")
    if host in _BLOCKED_MEDIA_HOSTS or host.endswith(".localhost"):
        raise MediaUrlError("Media URL hostname is not allowed")
    if parsed.username or parsed.password:
        raise MediaUrlError("Media URL must not include credentials")

    try:
        ipaddress.ip_address(host)
        numeric_hosts = [host]
    except ValueError:
        numeric_hosts = []

    if not numeric_hosts:
        try:
            infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise MediaUrlError("Media URL hostname could not be resolved") from exc
        numeric_hosts = []
        for info in infos:
            address = info[4][0]
            if address:
                numeric_hosts.append(address)

    if not numeric_hosts:
        raise MediaUrlError("Media URL hostname resolved to no addresses")

    for raw_ip in numeric_hosts:
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise MediaUrlError("Media URL resolved to an invalid address") from exc
        if _ip_is_blocked(ip):
            raise MediaUrlError("Media URL points at a private or local address")


def validate_audio_upload(filename: str | None, size_bytes: int) -> None:
    """Reject unexpected audio types and oversized uploads."""
    name = filename or "audio.wav"
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"File type {ext or '(none)'} is not allowed. "
            f"Use one of: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )
    if size_bytes > MAX_AUDIO_UPLOAD_BYTES:
        raise ValueError(
            f"Audio file exceeds {MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)}MB limit"
        )


def uvicorn_tls_kwargs(plan: ServeSecurityPlan) -> dict[str, str]:
    """Extra uvicorn.run kwargs when the operator supplied certificates."""
    if not _has_tls(plan.ssl_certfile, plan.ssl_keyfile):
        return {}
    return {
        "ssl_certfile": plan.ssl_certfile or "",
        "ssl_keyfile": plan.ssl_keyfile or "",
    }
