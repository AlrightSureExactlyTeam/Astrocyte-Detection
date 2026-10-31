"""Safeguarded experimental fine-tuning using unverified center pseudo-labels."""
from __future__ import annotations

import copy
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Callable

import yaml

from .workflow import PROJECT_ROOT, read_manifest, relative_path, resolve_path, sha256, utc_now, write_json_new

EXPECTED_HEADS = {"semantic_classes": 3, "center_channels": 1}


def inspect_checkpoint(path: str | Path, verify_strict: bool = True) -> dict:
    """Inspect a U-Net checkpoint without executing pickled Python objects."""
    import torch

    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Checkpoint could not be loaded safely: {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a mapping")
    if checkpoint.get("output_heads") != EXPECTED_HEADS:
        raise ValueError(
            f"Incompatible output_heads {checkpoint.get('output_heads')!r}; expected {EXPECTED_HEADS!r}"
        )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint is missing a nonempty model_state_dict")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint is missing its model/training configuration")
    architecture = config["model"].get("architecture", "unet")
    encoder = config["model"].get("encoder", "resnet18")
    input_channels = config.get("data", {}).get("input_channels", 1)
    strict_compatible = None
    strict_error = None
    if verify_strict:
        try:
            from src.train import build_model

            model = build_model(config)
            model.load_state_dict(state, strict=True)
            strict_compatible = True
        except Exception as exc:
            strict_compatible = False
            strict_error = str(exc)
    optimizer = checkpoint.get("optimizer_state_dict")
    scheduler_keys = sorted(key for key in checkpoint if "scheduler" in key.lower())
    info = {
        "path": relative_path(path),
        "sha256": sha256(path),
        "epoch": checkpoint.get("epoch"),
        "architecture": architecture,
        "encoder": encoder,
        "input_channels": input_channels,
        "output_heads": checkpoint["output_heads"],
        "strict_model_compatibility": strict_compatible,
        "strict_compatibility_error": strict_error,
        "optimizer_state_available": isinstance(optimizer, dict) and bool(optimizer),
        "scheduler_state_available": bool(scheduler_keys),
        "scheduler_state_keys": scheduler_keys,
        "fine_tune_compatible": strict_compatible is not False,
        "interrupted_run_resume_supported": False,
        "resume_note": (
            "The existing trainer initializes a fresh optimizer and does not restore scheduler state; "
            "loading these weights starts a new fine-tuning round, not an interrupted-run resume."
        ),
    }
    if strict_compatible is False:
        raise ValueError(f"Checkpoint is not strictly compatible with the existing model: {strict_error}")
    return info


def _load_config(path: str | Path) -> dict:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    from src.train import validate_config

    return copy.deepcopy(validate_config(config))


def _validate_pseudo_dataset(config: dict) -> tuple[list[dict[str, str]], dict]:
    data = config["data"]
    images_dir = resolve_path(data.get("images_dir", ""))
    annotations_dir = resolve_path(data.get("annotations_dir", ""))
    dataset_root = images_dir.parent
    if dataset_root.name != "GFAP_pseudo_labeled" or images_dir.name != "images":
        raise ValueError("Experimental pseudo-label training must use datasets/GFAP_pseudo_labeled/images")
    if annotations_dir != dataset_root / "annotations":
        raise ValueError("Pseudo-label annotations must be under GFAP_pseudo_labeled/annotations")
    if data.get("semantic_mask_dir") or data.get("instance_mask_dir"):
        raise ValueError("Cellpose cell regions must not be supplied as semantic/instance training truth")
    if data.get("confirmed_only", False):
        raise ValueError("confirmed_only=true would discard candidate_unverified center annotations")
    manifest_path = dataset_root / "manifest.csv"
    rows = read_manifest(manifest_path)
    if not rows:
        raise ValueError("Pseudo-label dataset is empty")
    for row in rows:
        if (
            row.get("annotation_status") != "pseudo_label_unverified"
            or row.get("review_status") != "unverified"
            or row.get("training_target") != "center_points_only"
            or row.get("semantic_labels_inferred") != "false"
        ):
            raise ValueError(f"Pseudo-label provenance is invalid for {row.get('image_id')}")
    groups = data.get("groups")
    if not isinstance(groups, dict) or len(set(groups.values())) < 2:
        raise ValueError("Explicit data.groups with at least two source-slide groups is required")
    sample_ids = {Path(row["original_filename"]).stem for row in rows}
    missing = sorted(sample_ids - set(groups))
    extra = sorted(set(groups) - sample_ids)
    if missing or extra:
        raise ValueError(f"data.groups must exactly cover dataset samples; missing={missing}, extra={extra}")
    if any(not str(groups[sample_id]).strip() for sample_id in sample_ids):
        raise ValueError("Every source-slide group must be nonempty")
    channel = data.get("channel")
    channels = {row.get("cell_channel") for row in rows}
    if not isinstance(channel, int) or isinstance(channel, bool) or channels != {str(channel)}:
        raise ValueError(f"data.channel must match every manifest cell_channel; observed={sorted(channels)}")
    versions = sorted({row.get("dataset_version", "") for row in rows})
    if not all(versions):
        raise ValueError("Every pseudo-label row needs a dataset_version")
    info = {
        "dataset_root": relative_path(dataset_root),
        "dataset_versions": versions,
        "sample_count": len(rows),
        "source_group_count": len(set(groups.values())),
        "training_target": "center_points_only",
        "review_status": "unverified",
        "independent_verified_validation": False,
    }
    return rows, info


def _next_round(checkpoint_root: Path) -> int:
    rounds = []
    if checkpoint_root.is_dir():
        for path in checkpoint_root.iterdir():
            match = re.fullmatch(r"round_(\d{3,})", path.name)
            if path.is_dir() and match:
                rounds.append(int(match.group(1)))
    return max(rounds, default=0) + 1


def _validate_replay(config: dict, round_number: int, rows: list[dict[str, str]]) -> dict:
    continual = config.get("continual", {})
    replay = continual.get("replay", {})
    if round_number == 1:
        return {"required": False, "retained_previous_examples": False, "previous_rounds": []}
    if replay.get("retained_previous_examples") is not True:
        raise ValueError("Rounds after the first require retained previous examples/replay")
    previous = replay.get("previous_rounds")
    if not isinstance(previous, list) or not previous:
        raise ValueError("Rounds after the first require continual.replay.previous_rounds manifests")
    current_ids = {row["image_id"] for row in rows}
    references = []
    for item in previous:
        manifest_path = resolve_path(item.get("data_manifest", "")) if isinstance(item, dict) else resolve_path(item)
        previous_rows = read_manifest(manifest_path)
        previous_ids = {row["image_id"] for row in previous_rows}
        missing = sorted(previous_ids - current_ids)
        if not previous_ids or missing:
            raise ValueError(f"Replay examples missing from current dataset: {manifest_path}: {missing[:10]}")
        references.append(relative_path(manifest_path))
    return {"required": True, "retained_previous_examples": True, "previous_rounds": references}


def _new_csv(path: Path, history: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in history for key in row))
    if not columns:
        raise ValueError("Training returned no history")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def run_approved_finetuning(
    config_path: str | Path,
    approve_pseudo_labels: bool = False,
    operation: str = "fine_tune",
    device: str = "auto",
    trainer: Callable | None = None,
    progress: Callable = print,
) -> dict:
    """Run one explicitly approved experimental round; never resume in-place."""
    if operation != "fine_tune":
        raise ValueError(
            "Only operation='fine_tune' is supported. Interrupted-run resume is unavailable because "
            "the existing trainer does not restore optimizer/scheduler state."
        )
    if not approve_pseudo_labels:
        raise ValueError("Pseudo-label training requires the explicit CLI approval flag")
    config = _load_config(config_path)
    continual = config.get("continual")
    if not isinstance(continual, dict):
        raise ValueError("A continual experiment configuration section is required")
    if continual.get("mode") != "fine_tune":
        raise ValueError("continual.mode must be fine_tune; in-place resume is forbidden")
    if (
        continual.get("pseudo_label_training_approved") is not True
        or continual.get("unverified_labels_acknowledged") is not True
        or not str(continual.get("approval_reference", "")).strip()
    ):
        raise ValueError(
            "Config must explicitly approve unverified pseudo-label training and name an approval_reference"
        )
    training = config["training"]
    if training.get("semantic_weight") != 0 or training.get("center_weight", 0) <= 0:
        raise ValueError("Pseudo-label experiments require semantic_weight: 0 and positive center_weight")
    rows, dataset_info = _validate_pseudo_dataset(config)
    checkpoint_root = resolve_path(training.get("checkpoint_dir", "checkpoints/gfap_continual"))
    if checkpoint_root.name != "gfap_continual" or checkpoint_root == PROJECT_ROOT / "checkpoints":
        raise ValueError("Experimental checkpoints must use a separate gfap_continual directory")
    round_number = _next_round(checkpoint_root)
    round_name = f"round_{round_number:03d}"
    round_dir = checkpoint_root / round_name
    result_dir = PROJECT_ROOT / "results" / "gfap_experiments" / round_name
    if round_dir.exists() or result_dir.exists():
        raise FileExistsError(f"Training output conflict: {round_dir} or {result_dir}")
    replay_info = _validate_replay(config, round_number, rows)

    parent_info = None
    init_weights = training.get("init_weights")
    if init_weights:
        parent_info = inspect_checkpoint(init_weights)
    elif round_number > 1 or continual.get("allow_random_initialization") is not True:
        raise ValueError("Fine-tuning requires compatible training.init_weights")

    # All validation happens before allocating a round or invoking the trainer.
    from src import train as train_module

    actual_trainer = trainer or train_module.train
    train_module.set_seed(config.get("project", {}).get("seed", 42))
    torch_device = train_module.get_device(device)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    round_dir.mkdir(exist_ok=False)
    config["training"]["checkpoint_dir"] = str(round_dir)
    write_json_new(round_dir / "training_config.json", config)
    dataset_manifest = resolve_path(config["data"]["images_dir"]).parent / "manifest.csv"
    with dataset_manifest.open("rb") as source, (round_dir / "data_manifest.csv").open("xb") as target:
        shutil.copyfileobj(source, target)
    try:
        outcome = actual_trainer(config, torch_device, progress=progress)
        nested_dir = resolve_path(outcome["checkpoint_dir"])
        nested_dir.relative_to(round_dir)
        last_path = nested_dir / "last.pt"
        best_path = nested_dir / "best.pt"
        last_info = inspect_checkpoint(last_path)
        best_info = inspect_checkpoint(best_path)
        history = outcome["history"]
        _new_csv(round_dir / "training_history.csv", history)
        metadata = {
            "created_utc": utc_now(),
            "round": round_name,
            "experiment_type": "experimental_self_training_on_unverified_center_pseudo_labels",
            "operation": "fine_tune_fresh_optimizer",
            "interrupted_run_resume": False,
            "parent_checkpoint": parent_info,
            "last_checkpoint": last_info,
            "best_checkpoint": best_info,
            "checkpoint_directory": relative_path(nested_dir),
            "dataset": dataset_info,
            "replay": replay_info,
            "random_seed": config.get("project", {}).get("seed", 42),
            "architecture": best_info["architecture"],
            "encoder": best_info["encoder"],
            "last_epoch": last_info["epoch"],
            "optimizer_state_saved": last_info["optimizer_state_available"],
            "scheduler_state_saved": last_info["scheduler_state_available"],
            "checkpoint_selection_note": (
                "best.pt means lowest loss on held-out unverified pseudo-label tiles only; "
                "it is not best-performing on independent verified data."
            ),
            "accuracy_improvement_claimed": False,
        }
        write_json_new(round_dir / "checkpoint_metadata.json", metadata)
        result_dir.mkdir(parents=True, exist_ok=False)
        report = {
            "created_utc": utc_now(),
            "round": round_name,
            "status": "completed",
            "training_started": True,
            "experimental_unverified_pseudo_labels": True,
            "checkpoint_metadata": relative_path(round_dir / "checkpoint_metadata.json"),
            "history": history,
            "evaluation_metrics": {},
            "evaluation_note": "No independent verified evaluation or accuracy-improvement claim is available.",
        }
        write_json_new(result_dir / "report.json", report)
        (result_dir / "report.md").write_text(
            f"# GFAP continual experiment {round_name}\n\n"
            "Experimental center-only fine-tuning on unverified pseudo-labels completed.\n\n"
            f"- Checkpoint metadata: `{metadata['checkpoint_directory']}`\n"
            f"- Last epoch: {metadata['last_epoch']}\n"
            "- No independent verified evaluation or accuracy-improvement claim is available.\n",
            encoding="utf-8",
        )
        return {"round": round_name, "checkpoint_dir": str(nested_dir), "metadata": metadata, "report": str(result_dir)}
    except Exception as exc:
        write_json_new(round_dir / "failure.json", {"created_utc": utc_now(), "error": str(exc)})
        raise
