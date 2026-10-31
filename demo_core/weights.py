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
            with urlopen(url, timeout=60) as response:
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
    except (URLError, TimeoutError, OSError):
        raise RuntimeError(f"Could not download {label}. Check the configured weight storage and try again.") from None
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
