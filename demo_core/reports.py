"""Unique, non-destructive demo outputs and human-readable report exports."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .images import display_image, encode_png
from .registry import PROJECT_ROOT
from .results import DemoResult

OUTPUT_ROOT = PROJECT_ROOT / "results/sciencepal_demo"


def new_run_directory(output_root: str | Path = OUTPUT_ROOT) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / (datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%f_") + uuid.uuid4().hex[:6])
    path.mkdir(parents=False, exist_ok=False)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def report_markdown(result: DemoResult, analysis: str | None = None) -> str:
    stats = result.statistics
    lines = [
        "# Astrocyte Detection preliminary report", "",
        f"- **Image identifier:** {result.sample_id}",
        f"- **Model:** {result.model_name} (`{result.model_id}`)",
        f"- **Mode:** {'Previously generated cached prediction' if result.mode == 'cached' else 'New live inference'}",
        f"- **Review status:** {result.review_status}",
        f"- **Ground truth:** {result.ground_truth}", "",
        "## Measured statistics", "",
        "```json", json.dumps(_json_safe(stats), indent=2, ensure_ascii=False), "```", "",
        "## Scientific limitations", "",
        "- Predictions are unverified candidates, not confirmed cell counts.",
        "- Manual labels are partial: 1 = Soma, 2 = Process, 255 = Ignore; unlabeled pixels are not confirmed background.",
        "- ImageJ masks are weak GFAP foreground labels, not Soma/Process ground truth.",
        "- Comparison samples are preliminary, in-sample, and from the same tissue section.",
        "- Full-image Precision, F1, Dice, and IoU are not valid with these incomplete labels.",
        "- Higher coverage does not establish segmentation superiority; U-Net foreground overprediction must be considered.",
    ]
    if analysis:
        lines.extend(["", "## SciencePal preliminary quality analysis", "", analysis])
    return "\n".join(lines) + "\n"


def save_demo_result(
    result: DemoResult,
    image: np.ndarray,
    input_metadata: dict[str, Any],
    output_root: str | Path = OUTPUT_ROOT,
) -> Path:
    """Copy cached artifacts with provenance; live artifacts are copied within a new export run."""
    run_dir = new_run_directory(output_root)
    (run_dir / "original_gfap.png").write_bytes(encode_png(display_image(image)))
    for name, source in {"prediction_mask.tif": result.mask_path, "overlay.png": result.overlay_path, **{
        f"{key}{path.suffix}": path for key, path in result.artifacts.items() if key not in {"mask", "overlay"}
    }}.items():
        if source.is_file():
            shutil.copyfile(source, run_dir / name)
    metadata = dict(input_metadata)
    metadata.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": result.sample_id,
        "model_id": result.model_id,
        "mode": result.mode,
        "review_status": "unverified",
        "ground_truth": False,
        "provenance": result.provenance,
    })
    write_json(run_dir / "input_metadata.json", metadata)
    write_json(run_dir / "statistics.json", result.statistics)
    (run_dir / "report.md").write_text(report_markdown(result), encoding="utf-8")
    return run_dir


def save_analysis(run_dir: str | Path, analysis: str, result: DemoResult) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    response_path = run_dir / "sciencepal_analysis.md"
    report_path = run_dir / "report_with_analysis.md"
    if response_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite an existing SciencePal analysis export")
    response_path.write_text(analysis + "\n", encoding="utf-8")
    report_path.write_text(report_markdown(result, analysis), encoding="utf-8")
    return response_path, report_path
