"""Configuration-driven model discovery; no model is loaded during discovery."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .weights import is_streamlit_cloud, local_unet_ready, remote_unet_configured

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = {
    "id", "name", "model_type", "checkpoint", "inference_adapter",
    "expected_input", "output_label_mapping", "version",
}
SUPPORTED_ADAPTERS = {"cellpose", "unet", "v33b2"}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    model_type: str
    checkpoint: Path
    inference_adapter: str
    expected_input: dict[str, Any]
    output_label_mapping: dict[str, Any]
    version: str
    cached: dict[str, Any]
    deployment: dict[str, Any]
    enabled: bool = True

    @property
    def live_available(self) -> bool:
        if not self.enabled or self.inference_adapter not in SUPPORTED_ADAPTERS:
            return False
        if is_streamlit_cloud() and not self.deployment.get("cloud_enabled", False):
            return False
        if self.inference_adapter == "unet":
            return local_unet_ready(self) or (is_streamlit_cloud() and remote_unet_configured(self))
        if self.inference_adapter == "v33b2":
            from .weights import remote_v33b2_configured

            return self.checkpoint.is_file() or (
                is_streamlit_cloud() and remote_v33b2_configured(self)
            )
        return self.checkpoint.is_file()

    @property
    def unavailable_reason(self) -> str | None:
        if not self.enabled:
            return "Disabled in the model registry."
        if self.inference_adapter not in SUPPORTED_ADAPTERS:
            return f"No installed live adapter named {self.inference_adapter!r}."
        if is_streamlit_cloud() and not self.deployment.get("cloud_enabled", False):
            return self.deployment.get(
                "cloud_disabled_reason",
                "Live inference is disabled on Streamlit Community Cloud. Existing Results remain available.",
            )
        if self.inference_adapter == "unet" and not local_unet_ready(self):
            if is_streamlit_cloud():
                if not remote_unet_configured(self):
                    names = [
                        self.deployment.get("checkpoint_url_env", "checkpoint URL"),
                        self.deployment.get("config_url_env", "config URL"),
                    ]
                    return f"U-Net Live inference is not configured; set {', '.join(names)} in Streamlit secrets."
                return None
            return f"U-Net checkpoint/config not found beside: {self.checkpoint.parent.parent}"
        if self.inference_adapter == "v33b2" and not self.checkpoint.is_file():
            from .weights import remote_v33b2_configured

            if is_streamlit_cloud():
                if not remote_v33b2_configured(self):
                    return (
                        "V3.3-B.2 Cloud artifacts are not configured; set the CPSAM ONNX "
                        "and V3 URLs in Streamlit secrets."
                    )
                return None
        if not self.checkpoint.is_file():
            return f"Checkpoint/model file not found: {self.checkpoint}"
        return None


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_model_registry(path: str | Path = PROJECT_ROOT / "configs/demo_models.yaml") -> list[ModelSpec]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    entries = payload.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Model registry must contain a nonempty 'models' list")
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every model registry entry must be a mapping")
        missing = sorted(REQUIRED_FIELDS - entry.keys())
        if missing:
            raise ValueError(f"Model entry is missing required fields: {missing}")
        model_id = str(entry["id"])
        if not model_id or model_id in seen:
            raise ValueError(f"Model IDs must be nonempty and unique: {model_id!r}")
        seen.add(model_id)
        specs.append(ModelSpec(
            id=model_id,
            name=str(entry["name"]),
            model_type=str(entry["model_type"]),
            checkpoint=resolve_path(entry["checkpoint"]),
            inference_adapter=str(entry["inference_adapter"]),
            expected_input=dict(entry["expected_input"]),
            output_label_mapping=dict(entry["output_label_mapping"]),
            version=str(entry["version"]),
            cached=dict(entry.get("cached") or {}),
            deployment=dict(entry.get("deployment") or {}),
            enabled=bool(entry.get("enabled", True)),
        ))
    return specs


def get_model(model_id: str, registry_path: str | Path = PROJECT_ROOT / "configs/demo_models.yaml") -> ModelSpec:
    for spec in load_model_registry(registry_path):
        if spec.id == model_id:
            return spec
    raise KeyError(f"Unknown model ID: {model_id}")
