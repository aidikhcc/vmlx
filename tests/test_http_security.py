# SPDX-License-Identifier: Apache-2.0
"""Tests for AIDI-aligned serve/auth/CORS/SSRF defaults."""

from types import SimpleNamespace

import pytest

from vmlx_engine.http_security import (
    DEFAULT_ALLOWED_ORIGINS,
    DEFAULT_RATE_LIMIT,
    MediaUrlError,
    ServeSecurityError,
    generate_api_key,
    is_loopback_bind,
    rate_limit_client_id,
    resolve_serve_security,
    resolve_trust_remote_code,
    validate_audio_upload,
    validate_outbound_media_url,
)


def test_loopback_bind_names():
    assert is_loopback_bind("127.0.0.1")
    assert is_loopback_bind("localhost")
    assert is_loopback_bind("::1")
    assert not is_loopback_bind("0.0.0.0")
    assert not is_loopback_bind("192.168.1.10")


def test_generate_api_key_is_local_prefix():
    key = generate_api_key()
    assert key.startswith("vmlx_")
    assert len(key) > 20


def test_resolve_serve_security_generates_key_on_loopback():
    plan = resolve_serve_security(
        host="127.0.0.1",
        api_key=None,
        env_api_key=None,
        allow_unauthenticated=False,
        allow_insecure_lan=False,
        allowed_origins=None,
        rate_limit=DEFAULT_RATE_LIMIT,
        trust_forwarded_for=False,
        trust_remote_code=None,
        no_trust_remote_code=False,
        model_name="mlx-community/Qwen3-8B-4bit",
    )
    assert plan.generated_api_key is True
    assert plan.api_key and plan.api_key.startswith("vmlx_")
    assert plan.allowed_origins == DEFAULT_ALLOWED_ORIGINS
    assert plan.trust_remote_code is False


def test_allow_unauthenticated_blocked_on_lan():
    with pytest.raises(ServeSecurityError, match="allow-unauthenticated"):
        resolve_serve_security(
            host="0.0.0.0",
            api_key=None,
            env_api_key=None,
            allow_unauthenticated=True,
            allow_insecure_lan=True,
            allowed_origins="*",
            rate_limit=0,
            trust_forwarded_for=False,
            trust_remote_code=None,
            no_trust_remote_code=False,
            model_name="local",
        )


def test_lan_bind_requires_tls_or_override():
    with pytest.raises(ServeSecurityError, match="without TLS"):
        resolve_serve_security(
            host="0.0.0.0",
            api_key="secret",
            env_api_key=None,
            allow_unauthenticated=False,
            allow_insecure_lan=False,
            allowed_origins="loopback",
            rate_limit=60,
            trust_forwarded_for=False,
            trust_remote_code=None,
            no_trust_remote_code=False,
            model_name="local",
        )

    plan = resolve_serve_security(
        host="0.0.0.0",
        api_key="secret",
        env_api_key=None,
        allow_unauthenticated=False,
        allow_insecure_lan=True,
        allowed_origins="*",
        rate_limit=60,
        trust_forwarded_for=False,
        trust_remote_code=None,
        no_trust_remote_code=False,
        model_name="local",
    )
    assert plan.api_key == "secret"
    assert plan.generated_api_key is False


def test_media_url_requires_https_and_public_host(monkeypatch):
    with pytest.raises(MediaUrlError, match="https"):
        validate_outbound_media_url("http://example.com/a.png")
    with pytest.raises(MediaUrlError):
        validate_outbound_media_url("https://127.0.0.1/secret.png")
    with pytest.raises(MediaUrlError):
        validate_outbound_media_url("https://169.254.169.254/latest/meta-data")
    with pytest.raises(MediaUrlError):
        validate_outbound_media_url("https://localhost/pic.png")

    def fake_getaddrinfo(host, port, type=0):
        assert host == "example.com"
        return [(None, None, None, None, ("93.184.216.34", port))]

    monkeypatch.setattr("vmlx_engine.http_security.socket.getaddrinfo", fake_getaddrinfo)
    validate_outbound_media_url("https://example.com/ok.png")


def test_rate_limit_ignores_forwarded_header_by_default():
    request = SimpleNamespace(
        headers={"X-Forwarded-For": "8.8.8.8"},
        client=SimpleNamespace(host="10.0.0.2"),
    )
    assert rate_limit_client_id(request, trust_forwarded_for=False) == "10.0.0.2"
    assert rate_limit_client_id(request, trust_forwarded_for=True) == "8.8.8.8"


def test_audio_upload_rejects_bad_type_and_size():
    validate_audio_upload("note.wav", 1024)
    with pytest.raises(ValueError, match="not allowed"):
        validate_audio_upload("note.exe", 1024)
    with pytest.raises(ValueError, match="exceeds"):
        validate_audio_upload("note.wav", 40 * 1024 * 1024)


def test_trust_remote_code_explicit_and_local(tmp_path):
    assert (
        resolve_trust_remote_code(
            trust_remote_code=None,
            no_trust_remote_code=False,
            model_name="mlx-community/Qwen3-8B-4bit",
        )
        is False
    )
    local = tmp_path / "bundle"
    local.mkdir()
    assert (
        resolve_trust_remote_code(
            trust_remote_code=None,
            no_trust_remote_code=False,
            model_name=str(local),
        )
        is True
    )
    assert (
        resolve_trust_remote_code(
            trust_remote_code=True,
            no_trust_remote_code=True,
            model_name=str(local),
        )
        is False
    )
