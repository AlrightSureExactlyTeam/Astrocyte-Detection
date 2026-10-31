"""Cloud-only, verified runtime weights for the existing U-Net Demo adapters."""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import requests

_DOWNLOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_CHUNK_SIZE = 1024 * 1024


def is_streamlit_cloud() -> bool:
    """Cloud behavior is explicit so local inference remains unchanged."""
    return os.environ.get("ASTROCYTE_DEPLOYMENT", "").strip().lower() == "streamlit_cloud"


def local_unet_ready(model: Any) -> bool:
    config = model.checkpoint.parent.parent / "config_snapshot.yaml"
    return model.checkpoint.is_file() and config.is_file()


def remote_unet_configured(model: Any) -> bool:
    deployment = model.deployment
    required = (
        "checkpoint_url_env", "config_url_env", "checkpoint_sha256",
        "checkpoint_size_bytes", "config_sha256", "config_size_bytes",
    )
    return (
        model.inference_adapter == "unet"
        and all(deployment.get(key) for key in required)
        and all(os.environ.get(deployment[key], "").strip() for key in ("checkpoint_url_env", "config_url_env"))
    )


@contextmanager
def cloud_inference_guard() -> Iterator[None]:
    """Prevent concurrent Cloud requests from multiplying model memory use."""
    if not is_streamlit_cloud():
        yield
        return
    with _INFERENCE_LOCK:
        yield


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_file(path: Path, expected_size: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and _sha256(path) == expected_sha256
    )


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    local_test_host = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local_test_host):
        raise ValueError("Weight URLs must use HTTPS; plain HTTP is allowed only for loopback tests.")


@contextmanager
def _open_download(url: str) -> Iterator[Any]:
    """Authenticate private hosts; never place tokens in URLs."""
    host = urlparse(url).hostname or ""
    if host == "huggingface.co":
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("Private Hugging Face downloads require HF_TOKEN in Streamlit secrets.")
        headers = {"Authorization": f"Bearer {token}"}
    elif host.endswith("github.com") or host.endswith("githubusercontent.com"):
        # Private GitHub Release assets answer 404 anonymously; the redirect
        # target is pre-signed, so requests stripping the header is correct.
        token = os.environ.get("ASTROCYTE_GITHUB_TOKEN", "").strip()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
    else:
        with urlopen(url, timeout=60) as response:
            yield response
        return
    # Requests strips Authorization when a redirect changes the destination host.
    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(10, 60),
    ) as response:
        response.raise_for_status()
        # Hugging Face serves small text files (e.g. YAML configs) with
        # Content-Encoding: gzip. Without this, raw.read() returns the
        # compressed bytes and integrity verification fails on valid files.
        response.raw.decode_content = True
        yield response.raw


def _download(url: str, destination: Path, expected_size: int, expected_sha256: str, label: str) -> None:
    """Download without including signed URLs or credentials in errors."""
    _validate_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            with _open_download(url) as response:
                size = 0
                digest = hashlib.sha256()
                while True:
                    block = response.read(_CHUNK_SIZE)
                    if not block:
                        break
                    size += len(block)
                    if size > expected_size:
                        raise RuntimeError(f"Downloaded {label} exceeds its declared size; inference was not started.")
                    digest.update(block)
                    handle.write(block)
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"Downloaded {label} failed integrity verification; inference was not started.")
        os.replace(temporary, destination)
    except (URLError, TimeoutError, OSError, requests.RequestException):
        raise RuntimeError(
            f"Could not download {label}. Check the configured storage and access permissions."
        ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_unet_checkpoint(model: Any) -> Path:
    """Return a local U-Net checkpoint or fetch its verified Cloud run layout."""
    if local_unet_ready(model):
        return model.checkpoint
    if not is_streamlit_cloud() or not model.deployment.get("cloud_enabled", False):
        raise RuntimeError("This U-Net checkpoint is not available locally.")
    if not remote_unet_configured(model):
        raise RuntimeError("U-Net Live inference is not configured; checkpoint and config URLs are required.")

    deployment = model.deployment
    checkpoint_sha256 = str(deployment["checkpoint_sha256"]).lower()
    config_sha256 = str(deployment["config_sha256"]).lower()
    checkpoint_size = int(deployment["checkpoint_size_bytes"])
    config_size = int(deployment["config_size_bytes"])
    cache_root = Path(os.environ.get(
        "ASTROCYTE_WEIGHT_CACHE",
        Path(__file__).resolve().parents[1] / ".cache" / "astrocyte_weights",
    ))
    run_dir = cache_root / model.id / checkpoint_sha256[:16]
    checkpoint = run_dir / "checkpoints" / "final.pt"
    config = run_dir / "config_snapshot.yaml"

    with _DOWNLOAD_LOCK:
        if not _valid_file(config, config_size, config_sha256):
            _download(
                os.environ[deployment["config_url_env"]], config, config_size, config_sha256,
                "the U-Net configuration",
            )
        if not _valid_file(checkpoint, checkpoint_size, checkpoint_sha256):
            _download(
                os.environ[deployment["checkpoint_url_env"]], checkpoint, checkpoint_size, checkpoint_sha256,
                "the U-Net checkpoint",
            )
    return checkpoint


_V33B2_REMOTE_KEYS = (
    "v3_checkpoint_url_env", "v3_checkpoint_size_bytes",
    "v3_config_url_env", "v3_config_sha256", "v3_config_size_bytes",
    "cpsam_onnx_part_url_template_env", "cpsam_onnx_parts",
    "cpsam_onnx_sha256", "cpsam_onnx_size_bytes",
)
_V33B2_REMOTE_ENVS = ("v3_checkpoint_url_env", "v3_config_url_env", "cpsam_onnx_part_url_template_env")


def remote_v33b2_configured(model: Any) -> bool:
    deployment = model.deployment
    return (
        model.inference_adapter == "v33b2"
        and all(deployment.get(key) for key in _V33B2_REMOTE_KEYS)
        and all(os.environ.get(deployment[key], "").strip() for key in _V33B2_REMOTE_ENVS)
    )


def ensure_v33b2_remote(model: Any) -> dict[str, Path]:
    """Fetch the verified Cloud layout for V3.3-B.2 (V3 checkpoint/config + CPSAM ONNX)."""
    if not remote_v33b2_configured(model):
        raise RuntimeError(
            "V3.3-B.2 Live inference is not configured for Cloud; CPSAM ONNX and V3 artifact URLs are required."
        )
    deployment = model.deployment
    onnx_sha = str(deployment["cpsam_onnx_sha256"]).lower()
    onnx_size = int(deployment["cpsam_onnx_size_bytes"])
    config_sha = str(deployment["v3_config_sha256"]).lower()
    config_size = int(deployment["v3_config_size_bytes"])
    v3_sha = str(model.expected_input.get("v3_checkpoint_sha256", "")).lower()
    v3_size = int(deployment["v3_checkpoint_size_bytes"])
    cache_root = Path(os.environ.get(
        "ASTROCYTE_WEIGHT_CACHE",
        Path(__file__).resolve().parents[1] / ".cache" / "astrocyte_weights",
    ))
    run_dir = cache_root / model.id / onnx_sha[:16]
    v3_checkpoint = run_dir / "v3" / "checkpoints" / "final.pt"
    v3_config = run_dir / "v3" / "config_snapshot.yaml"
    cpsam_onnx = run_dir / "cpsam_fp16store.onnx"

    with _DOWNLOAD_LOCK:
        if not _valid_file(v3_config, config_size, config_sha):
            _download(
                os.environ[deployment["v3_config_url_env"]], v3_config, config_size, config_sha,
                "the V3 configuration",
            )
        if not _valid_file(v3_checkpoint, v3_size, v3_sha):
            _download(
                os.environ[deployment["v3_checkpoint_url_env"]], v3_checkpoint, v3_size, v3_sha,
                "the V3 checkpoint",
            )
        if not _valid_file(cpsam_onnx, onnx_size, onnx_sha):
            _download_onnx_parts(model, cpsam_onnx, onnx_size, onnx_sha)
    return {
        "v3_checkpoint": v3_checkpoint,
        "cellpose_checkpoint": None,
        "v3_config": v3_config,
        "cpsam_onnx": cpsam_onnx,
    }


def _fetch_raw(url: str, destination: Path) -> None:
    """Download without integrity checks (used for shards; the reassembled file is verified)."""
    _validate_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as handle, _open_download(url) as response:
            while True:
                block = response.read(_CHUNK_SIZE)
                if not block:
                    break
                handle.write(block)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _download_onnx_parts(model: Any, destination: Path, expected_size: int, expected_sha256: str) -> None:
    """Fetch the sharded CPSAM ONNX (fp16-stored) and reassemble it with one final checksum."""
    deployment = model.deployment
    template = os.environ[deployment["cpsam_onnx_part_url_template_env"]]
    parts = int(deployment["cpsam_onnx_parts"])
    parts_dir = destination.parent / "onnx_parts"
    temporary = destination.with_name(destination.name + ".assemble")
    try:
        with temporary.open("wb") as handle:
            for index in range(1, parts + 1):
                url = template.replace("{nn}", f"{index:02d}")
                part_path = parts_dir / f"part{index:02d}"
                if not part_path.is_file():
                    _fetch_raw(url, part_path)
                with part_path.open("rb") as part_handle:
                    while True:
                        block = part_handle.read(_CHUNK_SIZE)
                        if not block:
                            break
                        handle.write(block)
        if temporary.stat().st_size != expected_size or _sha256(temporary) != expected_sha256:
            temporary.unlink(missing_ok=True)
            for leftover in parts_dir.glob("part*"):
                leftover.unlink(missing_ok=True)
            raise RuntimeError(
                "Downloaded the CPSAM ONNX failed integrity verification; inference was not started."
            )
        os.replace(temporary, destination)
        for leftover in parts_dir.glob("part*"):
            leftover.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
