"""Load real cached predictions and authoritative comparison statistics."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .images import read_mask
from .registry import ModelSpec, PROJECT_ROOT, resolve_path


@dataclass
class DemoResult:
    sample_id: str
    model_id: str
    model_name: str
    mode: str
    image_path: Path | None
    mask_path: Path
    overlay_path: Path
    statistics: dict[str, Any]
    artifacts: dict[str, Path]
    provenance: dict[str, Any]
    review_status: str = "unverified"
    ground_truth: bool = False


def load_demo_config(path: str | Path = PROJECT_ROOT / "configs/demo_datasets.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "comparison" not in config:
        raise ValueError("Demo configuration lacks comparison settings")
    return config


def comparison_manifest(config_path: str | Path = PROJECT_ROOT / "configs/demo_datasets.yaml") -> tuple[dict, list[dict], Path]:
    config = load_demo_config(config_path)
    comparison = config["comparison"]
    run_dir = resolve_path(comparison["run_dir"])
    verification_path = run_dir / comparison["source_verification"]
    with verification_path.open(encoding="utf-8") as handle:
        verification = json.load(handle)
    for item in verification:
        if not item.get("spatial_and_content_match"):
            raise ValueError(f"Source/content alignment is not verified for {item.get('sample_id')}")
    return comparison, verification, run_dir


def available_samples(config_path: str | Path = PROJECT_ROOT / "configs/demo_datasets.yaml") -> list[str]:
    _, verification, _ = comparison_manifest(config_path)
    return [str(item["sample_id"]) for item in verification]


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _metrics_from_csv(path: Path, sample_id: str, variants: list[str]) -> dict[str, dict[str, Any]]:
    if not variants:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, Any]] = {}
    for variant in variants:
        matches = [row for row in rows if row["sample_id"] == sample_id and row["prediction"] == variant]
        if len(matches) != 1:
            raise ValueError(f"Expected one comparison row for {sample_id} / {variant}, found {len(matches)}")
        metrics: dict[str, Any] = {}
        for key, value in matches[0].items():
            if key in {"sample_id", "prediction"}:
                continue
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                metrics[key] = value
        output[variant] = metrics
    return output


def load_cached_result(
    sample_id: str,
    model: ModelSpec,
    config_path: str | Path = PROJECT_ROOT / "configs/demo_datasets.yaml",
    verify_source: bool = True,
) -> DemoResult:
    comparison, verification, comparison_run = comparison_manifest(config_path)
    entries = {item["sample_id"]: item for item in verification}
    if sample_id not in entries:
        raise KeyError(f"No verified cached comparison sample: {sample_id}")
    cached = model.cached
    if not cached:
        raise ValueError(f"Model {model.id} has no cached-results configuration")
    model_run = resolve_path(cached["run_dir"])
    source_pattern = cached.get("source_pattern", comparison["source_pattern"])
    source = resolve_path(source_pattern.format(
        source_stem=sample_id.removesuffix("_RGB"), sample_id=sample_id,
    ))
    expected_source_sha = entries[sample_id]["quantitative_sha256"]
    source_evidence = entries[sample_id]
    if "source_manifest" in cached:
        with (model_run / cached["source_manifest"]).open(newline="", encoding="utf-8-sig") as handle:
            matches = [row for row in csv.DictReader(handle) if row["sample_id"] == sample_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one accepted-pilot source row for {sample_id}, found {len(matches)}")
        expected_source_sha = matches[0]["source_sha256"]
        source_evidence = matches[0]
    mask_path = model_run / cached["mask_pattern"].format(sample_id=sample_id)
    overlay_path = model_run / cached["overlay_pattern"].format(sample_id=sample_id)
    stats_path = model_run / cached["statistics_pattern"].format(sample_id=sample_id)
    for path in (source, mask_path, overlay_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required cached artifact not found: {path}")
    mask = read_mask(mask_path)
    if verify_source:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected_source_sha:
            raise ValueError(f"Source TIFF checksum mismatch for {sample_id}")
        statistics_on_disk = _json(stats_path)
        if statistics_on_disk.get("sample_id") != sample_id:
            raise ValueError(f"Prediction/image identifier mismatch for {sample_id}")
        if model.id == "cellpose" and statistics_on_disk.get("source_sha256") != digest:
            raise ValueError(f"Cellpose prediction/source checksum mismatch for {sample_id}")
    statistics = _json(stats_path)
    variants = [cached[key] for key in ("comparison_variant", "comparison_variant_foreground", "comparison_variant_process") if key in cached]
    statistics["comparison_metrics"] = _metrics_from_csv(
        comparison_run / comparison["summary_csv"], sample_id, variants,
    )
    statistics["review_status"] = "unverified"
    statistics["ground_truth"] = False
    statistics["evaluation_scope"] = "preliminary_in_sample_same_tissue_section"
    statistics["partial_annotation_warning"] = "Unlabeled pixels are not confirmed background."
    artifacts = {"mask": mask_path, "overlay": overlay_path, "statistics_source": stats_path}
    nucleus = cached.get("nucleus_pattern")
    if nucleus:
        artifacts["nucleus_mask"] = model_run / nucleus.format(sample_id=sample_id)
    process_probability = cached.get("process_probability_pattern")
    if process_probability:
        artifacts["process_probability"] = model_run / process_probability.format(sample_id=sample_id)
    for name, pattern in cached.get("artifact_patterns", {}).items():
        artifacts[name] = model_run / pattern.format(sample_id=sample_id)
    for name, path in list(artifacts.items()):
        if not path.is_file():
            raise FileNotFoundError(f"Cached {name} artifact not found: {path}")
    return DemoResult(
        sample_id=sample_id,
        model_id=model.id,
        model_name=model.name,
        mode="cached",
        image_path=source,
        mask_path=mask_path,
        overlay_path=overlay_path,
        statistics=statistics,
        artifacts=artifacts,
        provenance={
            "mode": "previously_generated_cached_result",
            "model_id": model.id,
            "model_version": model.version,
            "checkpoint_or_model": str(model.checkpoint),
            "model_run_dir": str(model_run),
            "comparison_run_dir": str(comparison_run),
            "source_verification": source_evidence,
        },
    )


def comparison_visualization(sample_id: str, config_path: str | Path = PROJECT_ROOT / "configs/demo_datasets.yaml") -> Path:
    comparison, _, run_dir = comparison_manifest(config_path)
    if sample_id not in available_samples(config_path):
        raise KeyError(sample_id)
    path = run_dir / comparison["visualization_pattern"].format(sample_id=sample_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def aggregate_metrics(config_path: str | Path = PROJECT_ROOT / "configs/demo_datasets.yaml") -> list[dict[str, Any]]:
    comparison, _, run_dir = comparison_manifest(config_path)
    with (run_dir / comparison["aggregate_csv"]).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            try:
                converted[key] = float(value)
            except (TypeError, ValueError):
                converted[key] = value
        output.append(converted)
    return output
