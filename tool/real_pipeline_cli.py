"""Shared CLI implementation for real-ROI pipeline entry points."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_pipeline import (  # noqa: E402
    compare_metrics, create_unique_dir, evaluate_experiment, generate_report,
    generate_summary, generate_visualizations, predict_cellpose_image,
    predict_unet_image, validate_real_data, write_json,
)
from train import get_device, load_config, set_seed, train  # noqa: E402


def parser_for(stage):
    parser = argparse.ArgumentParser(description=f"Real ROI pipeline: {stage}")
    parser.add_argument("--data-root", default="datasets/real_slide_test")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--results-root", default="results/experiments")
    parser.add_argument("--experiment-name", default="pilot")
    parser.add_argument("--checkpoint")
    parser.add_argument("--match-distance", type=float, default=8.0)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--model", choices=["unet", "cellpose"])
    parser.add_argument("--stage", choices=["validate", "train", "predict", "evaluate", "compare", "report", "all"])
    return parser


def experiment_directory(args, create=False):
    if args.experiment_dir is not None:
        path = args.experiment_dir
        if create:
            path.mkdir(parents=True, exist_ok=False)
        return path
    if create:
        return create_unique_dir(Path(args.results_root), args.experiment_name)
    raise ValueError("Pass --experiment-dir for an existing experiment, or omit it to create a new one")


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def snapshot(experiment, args):
    config = load_yaml(args.config)
    config["real_experiment"] = {
        "data_root": str(Path(args.data_root)), "match_distance_pixels": args.match_distance,
        "checkpoint": args.checkpoint, "device": "cpu",
    }
    with (experiment / "experiment_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return config


def require_images(data_root):
    images, errors = __import__("real_pipeline")._image_index(Path(data_root) / "images")
    if errors or not images:
        raise ValueError("; ".join(errors) if errors else f"No ROI images in {Path(data_root) / 'images'}")
    return images


def predict(args, experiment, config, model):
    images = require_images(args.data_root)
    if model == "unet":
        if not args.checkpoint:
            raise ValueError("U-Net prediction requires --checkpoint")
        if not Path(args.checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    for image_path in images.values():
        if model == "unet":
            predict_unet_image(image_path, args.checkpoint, config, experiment / "unet", "cpu")
        else:
            predict_cellpose_image(image_path, config, experiment / "cellpose", "cpu")
    print(f"{model} predictions saved under {experiment / model}")


def read_metric_rows(experiment, models=("unet", "cellpose")):
    rows = []
    for model in models:
        path = experiment / "metrics" / f"{model}_metrics.csv"
        if not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for key in ("tp", "fp", "fn", "predicted_count", "confirmed_count", "uncertain_count", "ignored_count", "count_error"):
                    row[key] = int(row[key])
                for key in ("precision", "recall", "f1", "match_distance_pixels"):
                    row[key] = float(row[key])
                rows.append(row)
    return rows


def finalize(experiment, rows, validation=None, visualize=False):
    if validation is None:
        validation_path = experiment / "validation" / "validation_report.json"
        if validation_path.is_file():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
    comparisons = compare_metrics(rows)
    summary = generate_summary(experiment, rows, comparisons, validation)
    if visualize and rows:
        generate_visualizations(args_data_root.get(), experiment, rows)
    generate_report(experiment, rows, comparisons, summary, validation)
    return summary


class args_data_root:
    value = None

    @classmethod
    def get(cls):
        if cls.value is None:
            raise ValueError("data root not initialized")
        return cls.value


def run_stage(stage, args):
    args_data_root.value = args.data_root
    if stage == "train":
        config = load_config(args.config)
        set_seed(config.get("project", {}).get("seed", 42))
        result = train(config, get_device("cpu"))
        print(json.dumps(result, indent=2))
        return 0

    create = stage in {"predict", "all"} and args.experiment_dir is None
    experiment = experiment_directory(args, create=create)
    experiment.mkdir(parents=True, exist_ok=True)
    config = snapshot(experiment, args) if create or not (experiment / "experiment_config.yaml").exists() else load_yaml(experiment / "experiment_config.yaml")

    if stage == "validate":
        report = validate_real_data(args.data_root, experiment / "validation" / "validation_report.json")
        print(json.dumps({key: report[key] for key in ("status", "image_count", "annotation_count", "errors")}, indent=2))
        print(f"Validation report: {experiment / 'validation' / 'validation_report.json'}")
        return 0

    if stage == "predict":
        if not args.model:
            raise ValueError("--model unet or --model cellpose is required")
        predict(args, experiment, config, args.model)
        return 0

    if stage == "evaluate":
        validation = validate_real_data(args.data_root, experiment / "validation" / "validation_report.json")
        if validation["status"] != "valid":
            raise ValueError("Real-data validation failed; inspect validation_report.json")
        rows = evaluate_experiment(args.data_root, experiment, max_distance=args.match_distance)
        finalize(experiment, rows, validation)
        print(f"Evaluation saved under {experiment / 'metrics'}")
        return 0

    if stage == "compare":
        rows = read_metric_rows(experiment)
        finalize(experiment, rows)
        print(f"Comparison saved under {experiment / 'metrics'}")
        return 0

    if stage == "report":
        rows = read_metric_rows(experiment)
        finalize(experiment, rows)
        print(f"Report: {experiment / 'report.md'}")
        return 0

    if stage == "all":
        validation = validate_real_data(args.data_root, experiment / "validation" / "validation_report.json")
        if validation["status"] != "valid":
            raise ValueError("Real-data validation failed; no predictions or metrics were generated")
        predict(args, experiment, config, "unet")
        predict(args, experiment, config, "cellpose")
        rows = evaluate_experiment(args.data_root, experiment, max_distance=args.match_distance)
        comparisons = compare_metrics(rows)
        summary = generate_summary(experiment, rows, comparisons, validation)
        generate_visualizations(args.data_root, experiment, rows)
        generate_report(experiment, rows, comparisons, summary, validation)
        print(f"Complete experiment: {experiment}")
        return 0
    raise ValueError(f"Unknown stage: {stage}")


def main(forced_stage=None):
    parser = parser_for(forced_stage or "run_experiment")
    args = parser.parse_args()
    stage = forced_stage or args.stage
    if not stage:
        parser.error("--stage is required")
    return run_stage(stage, args)
