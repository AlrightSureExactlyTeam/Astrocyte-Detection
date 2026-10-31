"""Small, non-training dataset registration interface."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .registry import PROJECT_ROOT, resolve_path

ANNOTATION_TYPES = {"manual_partial", "imagej_weak", "fully_verified"}
REQUIRED_FIELDS = {
    "name", "image_location", "annotation_location", "annotation_type",
    "label_mapping", "source_description", "version",
}
MANUAL_MAPPING = {"soma": 1, "process": 2, "ignore": 255}


def _integer_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError("label_mapping must be a nonempty mapping")
    try:
        return {str(key).lower(): int(label) for key, label in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Every label mapping value must be an integer") from exc


def validate_dataset_config(config: dict[str, Any], require_files: bool = True) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Dataset configuration must be a mapping")
    missing = sorted(REQUIRED_FIELDS - config.keys())
    if missing:
        raise ValueError(f"Dataset configuration is missing fields: {missing}")
    annotation_type = str(config["annotation_type"])
    if annotation_type not in ANNOTATION_TYPES:
        raise ValueError(f"Unsupported annotation type {annotation_type!r}; choose {sorted(ANNOTATION_TYPES)}")
    name = str(config["name"]).strip()
    if not name:
        raise ValueError("Dataset name must not be empty")
    image_location = resolve_path(config["image_location"])
    annotation_location = resolve_path(config["annotation_location"])
    if require_files:
        for kind, path in (("Image", image_location), ("Annotation", annotation_location)):
            if not path.exists():
                raise FileNotFoundError(f"{kind} location not found: {path}")
    mapping = _integer_mapping(config["label_mapping"])
    if annotation_type == "manual_partial" and mapping != MANUAL_MAPPING:
        raise ValueError(f"Manual partial labels must use exactly {MANUAL_MAPPING}")
    if annotation_type == "imagej_weak" and {"soma", "process"} & mapping.keys():
        raise ValueError("ImageJ weak labels must not be interpreted as Soma or Process")
    normalized = dict(config)
    normalized.update({
        "name": name,
        "image_location": str(image_location),
        "annotation_location": str(annotation_location),
        "annotation_type": annotation_type,
        "label_mapping": mapping,
        "source_description": str(config["source_description"]),
        "version": str(config["version"]),
        "registration_only": True,
        "training_started": False,
    })
    return normalized


def register_dataset(
    dataset_config: dict[str, Any] | str | Path,
    registry_path: str | Path = PROJECT_ROOT / "configs/demo_dataset_registry.json",
    require_files: bool = True,
) -> dict[str, Any]:
    """Validate and register metadata only. This function never invokes training."""
    if isinstance(dataset_config, (str, Path)):
        with Path(dataset_config).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    else:
        config = dataset_config
    record = validate_dataset_config(config, require_files=require_files)
    registry_path = Path(registry_path)
    records = []
    if registry_path.exists():
        with registry_path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError("Dataset registry JSON must contain a list")
    if any(item.get("name") == record["name"] and item.get("version") == record["version"] for item in records):
        raise ValueError(f"Dataset {record['name']!r} version {record['version']!r} is already registered")
    records.append(record)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry_path.with_suffix(registry_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary registry already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(registry_path)
    return record


def load_dataset_registry(path: str | Path = PROJECT_ROOT / "configs/demo_datasets.yaml") -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    entries = payload.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Dataset configuration must contain a 'datasets' list")
    return [validate_dataset_config(entry, require_files=False) for entry in entries]
