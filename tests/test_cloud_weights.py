"""Focused Cloud weight tests; all downloads are mocked and contain no secrets."""
from __future__ import annotations

import gzip
import hashlib
import io
from dataclasses import replace
from urllib.error import URLError

import pytest
import requests
from urllib3 import HTTPResponse

from demo_core.registry import get_model, load_model_registry
from demo_core.results import available_samples, load_cached_result
from demo_core.weights import ensure_unet_checkpoint

SECRET_MARKER = "private-test-token"


def _remote_model(model_id, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROCYTE_DEPLOYMENT", "streamlit_cloud")
    monkeypatch.setenv("ASTROCYTE_WEIGHT_CACHE", str(tmp_path / "cache"))
    model = replace(
        get_model(model_id),
        checkpoint=tmp_path / model_id / "checkpoints" / "final.pt",
    )
    payloads = {
        "checkpoint": f"verified {model_id} checkpoint".encode(),
        "config": f"verified {model_id} config".encode(),
    }
    urls = {
        key: f"https://storage.example/{model_id}/{key}?signature={SECRET_MARKER}"
        for key in payloads
    }
    deployment = dict(model.deployment)
    for key in ("checkpoint", "config"):
        deployment[f"{key}_sha256"] = hashlib.sha256(payloads[key]).hexdigest()
        deployment[f"{key}_size_bytes"] = len(payloads[key])
        monkeypatch.setenv(deployment[f"{key}_url_env"], urls[key])
    model = replace(model, deployment=deployment)
    return model, urls, payloads


def _mock_downloads(monkeypatch, urls, payloads, calls):
    def fake_urlopen(url, timeout):
        assert timeout == 60
        calls.append(url)
        for key, candidate in urls.items():
            if url == candidate:
                return io.BytesIO(payloads[key])
        raise URLError("unexpected URL")

    monkeypatch.setattr("demo_core.weights.urlopen", fake_urlopen)


def test_cloud_gating_existing_results_and_startup_do_not_download(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROCYTE_DEPLOYMENT", "streamlit_cloud")
    for name in (
        "ASTROCYTE_UNET_V1_CHECKPOINT_URL", "ASTROCYTE_UNET_V1_CONFIG_URL",
        "ASTROCYTE_UNET_V2_CHECKPOINT_URL", "ASTROCYTE_UNET_V2_CONFIG_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("Network access during discovery/Existing Results is forbidden")

    monkeypatch.setattr("demo_core.weights.urlopen", no_network)
    models = {model.id: model for model in load_model_registry()}
    for model_id in ("cellpose", "v33b2"):
        assert not models[model_id].live_available
        assert "Existing Results remain available" in models[model_id].unavailable_reason
    for model_id in ("unet_v1", "unet_v2"):
        models[model_id] = replace(
            models[model_id], checkpoint=tmp_path / model_id / "checkpoints" / "final.pt",
        )
        assert not models[model_id].live_available
        assert "Streamlit secrets" in models[model_id].unavailable_reason
        for key in ("checkpoint_url_env", "config_url_env"):
            monkeypatch.setenv(models[model_id].deployment[key], f"https://storage.example/{model_id}/{key}")

    assert models["unet_v1"].live_available and models["unet_v2"].live_available
    result = load_cached_result(available_samples()[0], models["unet_v1"])
    assert result.mode == "cached"


def test_verified_download_is_lazy_cached_and_atomic(tmp_path, monkeypatch):
    model, urls, payloads = _remote_model("unet_v1", tmp_path, monkeypatch)
    calls = []
    _mock_downloads(monkeypatch, urls, payloads, calls)
    assert model.live_available and calls == []

    checkpoint = ensure_unet_checkpoint(model)
    assert checkpoint.read_bytes() == payloads["checkpoint"]
    assert (checkpoint.parent.parent / "config_snapshot.yaml").read_bytes() == payloads["config"]
    assert len(calls) == 2
    assert ensure_unet_checkpoint(model) == checkpoint
    assert len(calls) == 2
    assert not list((tmp_path / "cache").rglob("*.part"))


def test_corrupt_cached_checkpoint_is_replaced(tmp_path, monkeypatch):
    model, urls, payloads = _remote_model("unet_v2", tmp_path, monkeypatch)
    calls = []
    _mock_downloads(monkeypatch, urls, payloads, calls)
    checkpoint = ensure_unet_checkpoint(model)
    checkpoint.write_bytes(b"corrupt")
    calls.clear()

    assert ensure_unet_checkpoint(model) == checkpoint
    assert checkpoint.read_bytes() == payloads["checkpoint"]
    assert calls == [urls["checkpoint"]]


def test_hash_mismatch_fails_closed_without_leaking_signed_url(tmp_path, monkeypatch):
    model, urls, payloads = _remote_model("unet_v1", tmp_path, monkeypatch)
    payloads["checkpoint"] = b"X" * len(payloads["checkpoint"])
    calls = []
    _mock_downloads(monkeypatch, urls, payloads, calls)

    with pytest.raises(RuntimeError, match="integrity") as caught:
        ensure_unet_checkpoint(model)
    assert SECRET_MARKER not in str(caught.value)
    checkpoint = tmp_path / "cache" / "unet_v1" / model.deployment["checkpoint_sha256"][:16] / "checkpoints" / "final.pt"
    assert not checkpoint.exists()
    assert not list((tmp_path / "cache").rglob("*.part"))


def test_download_error_does_not_leak_signed_url(tmp_path, monkeypatch):
    model, _, _ = _remote_model("unet_v1", tmp_path, monkeypatch)

    def fail(url, timeout):
        raise URLError(f"request failed: {url}")

    monkeypatch.setattr("demo_core.weights.urlopen", fail)
    with pytest.raises(RuntimeError, match="Could not download") as caught:
        ensure_unet_checkpoint(model)
    assert SECRET_MARKER not in str(caught.value)


def test_non_https_remote_weight_is_rejected(tmp_path, monkeypatch):
    model, _, _ = _remote_model("unet_v1", tmp_path, monkeypatch)
    monkeypatch.setenv(model.deployment["config_url_env"], "http://storage.example/config")
    with pytest.raises(ValueError, match="HTTPS"):
        ensure_unet_checkpoint(model)


class _FakeHfResponse:
    def __init__(self, payload=b"", error=None):
        self.raw = io.BytesIO(payload)
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def raise_for_status(self):
        if self.error:
            raise self.error


def _use_private_hf_urls(model, monkeypatch):
    urls = {
        "checkpoint": "https://huggingface.co/xinyu-liu2026/astrocyte-unet-demo/resolve/main/v1/final.pt",
        "config": "https://huggingface.co/xinyu-liu2026/astrocyte-unet-demo/resolve/main/v1/config_snapshot.yaml",
    }
    for key, url in urls.items():
        monkeypatch.setenv(model.deployment[f"{key}_url_env"], url)
    return urls


def test_private_hf_download_uses_header_and_preserves_integrity(tmp_path, monkeypatch):
    model, _, payloads = _remote_model("unet_v1", tmp_path, monkeypatch)
    urls = _use_private_hf_urls(model, monkeypatch)
    token = "hf_read_only_test_token"
    monkeypatch.setenv("HF_TOKEN", token)
    calls = []

    def fake_get(url, headers, stream, timeout):
        calls.append({"url": url, "headers": headers, "stream": stream, "timeout": timeout})
        key = "checkpoint" if url == urls["checkpoint"] else "config"
        return _FakeHfResponse(payloads[key])

    monkeypatch.setattr("demo_core.weights.requests.get", fake_get)
    checkpoint = ensure_unet_checkpoint(model)
    assert checkpoint.read_bytes() == payloads["checkpoint"]
    assert len(calls) == 2
    for call in calls:
        assert call["headers"] == {"Authorization": f"Bearer {token}"}
        assert call["stream"] is True and call["timeout"] == (10, 60)
        assert token not in call["url"]


def test_private_hf_download_requires_token_without_network(tmp_path, monkeypatch):
    model, _, _ = _remote_model("unet_v1", tmp_path, monkeypatch)
    _use_private_hf_urls(model, monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("Network request attempted without HF_TOKEN")

    monkeypatch.setattr("demo_core.weights.requests.get", no_network)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        ensure_unet_checkpoint(model)


def test_private_hf_http_failure_is_safe_and_leaves_no_partial_file(tmp_path, monkeypatch):
    model, _, _ = _remote_model("unet_v1", tmp_path, monkeypatch)
    _use_private_hf_urls(model, monkeypatch)
    token = "hf_read_only_test_token"
    monkeypatch.setenv("HF_TOKEN", token)

    def denied(*args, **kwargs):
        return _FakeHfResponse(error=requests.HTTPError("401 Unauthorized"))

    monkeypatch.setattr("demo_core.weights.requests.get", denied)
    with pytest.raises(RuntimeError, match="Could not download") as caught:
        ensure_unet_checkpoint(model)
    assert token not in str(caught.value)
    assert not list((tmp_path / "cache").rglob("*.part"))


def test_cross_host_redirect_strips_authorization():
    session = requests.Session()
    token = "hf_read_only_test_token"
    original = requests.Request(
        "GET", "https://huggingface.co/repo/resolve/main/model.pt",
        headers={"Authorization": f"Bearer {token}"},
    ).prepare()
    redirected = requests.Request("GET", "https://cdn.example/model.pt").prepare()
    response = requests.Response()
    response.request = original

    session.rebuild_auth(redirected, response)
    assert "Authorization" not in redirected.headers


def _gzip_encoded_raw(payload: bytes) -> HTTPResponse:
    """Faithful stand-in for HF's gzipped text responses."""
    return HTTPResponse(
        body=io.BytesIO(gzip.compress(payload)),
        headers={"Content-Encoding": "gzip"},
        preload_content=False,
        decode_content=False,
    )


def test_gzip_encoded_hf_response_is_decoded_before_verification(tmp_path, monkeypatch):
    """HF gzips small text configs; verification must cover the real bytes."""
    model, _, payloads = _remote_model("unet_v2", tmp_path, monkeypatch)
    urls = _use_private_hf_urls(model, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_read_only_test_token")

    def fake_get(url, headers, stream, timeout):
        key = "checkpoint" if url == urls["checkpoint"] else "config"
        response = _FakeHfResponse()
        response.raw = _gzip_encoded_raw(payloads[key])
        return response

    monkeypatch.setattr("demo_core.weights.requests.get", fake_get)
    checkpoint = ensure_unet_checkpoint(model)
    assert checkpoint.read_bytes() == payloads["checkpoint"]
    assert (checkpoint.parent.parent / "config_snapshot.yaml").read_bytes() == payloads["config"]
