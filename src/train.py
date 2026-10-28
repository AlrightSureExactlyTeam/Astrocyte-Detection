"""Training pipeline for AstrocyteUNet.

Owner: Liu (model/training) | Input: configs/config.yaml + BBBC042 dataloaders.

REDESIGN contract consumed here (see model.py / dataset.py):
- model(images) -> {"semantic_logits": [B,3,H,W], "center_logits": [B,1,H,W]}
  (raw logits; softmax/sigmoid are applied only at inference).
- batch targets: semantic_mask [B,H,W] int64 (0 bg / 1 soma / 2 process),
  center_heatmap [B,1,H,W] float32 in [0,1], valid_mask [B,H,W] bool
  (False = unlabeled/ambiguous pixel, excluded from the loss).

The real-data entry point (train()) expects dataset.build_dataloaders();
until that lands, use `python src/train.py --self-test` to verify the whole
training loop on synthetic data end to end.
"""

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

# FIX: Use an explicit package/script import branch (same pattern as baseline_cellpose).
if __package__:
    from .model import create_model
else:
    from model import create_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
NUM_SEMANTIC_CLASSES = 3  # 0 background / 1 soma / 2 visible process


# --------------------------------------------------------------------------- #
# Configuration and setup
# --------------------------------------------------------------------------- #

def _strict_int(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _finite_number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_config(config: dict) -> dict:
    """Validate training values without truncating floats or coercing strings."""
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping, got {type(config).__name__}")
    for section in ("data", "model", "training"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Config is missing required section: {section!r}")
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["training"]
    _strict_int(data_cfg.get("input_channels", 1), "data.input_channels", 1)
    image_size = data_cfg.get("image_size", (512, 512))
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("data.image_size must contain (height, width)")
    for index, value in enumerate(image_size):
        _strict_int(value, f"data.image_size[{index}]", 1)
    _strict_int(model_cfg.get("output_channels", 4), "model.output_channels", 1)
    _strict_int(train_cfg.get("epochs", 50), "training.epochs", 1)
    _strict_int(train_cfg.get("batch_size", 8), "training.batch_size", 1)
    _strict_int(train_cfg.get("num_workers", 0), "training.num_workers", 0)
    _strict_int(train_cfg.get("early_stopping_patience", 10), "training.early_stopping_patience", 1)
    if _finite_number(train_cfg.get("learning_rate", 1e-4), "training.learning_rate") <= 0:
        raise ValueError("training.learning_rate must be positive")
    weights = [
        _finite_number(train_cfg.get("semantic_weight", 1.0), "training.semantic_weight"),
        _finite_number(train_cfg.get("center_weight", 1.0), "training.center_weight"),
    ]
    if any(weight < 0 for weight in weights) or not any(weight > 0 for weight in weights):
        raise ValueError("Loss weights must be nonnegative and at least one must be positive")
    _strict_int(config.get("project", {}).get("seed", 42), "project.seed", 0)
    return config


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load and strictly validate the YAML training configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return validate_config(config)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and torch for reproducible runs."""
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # FIX (review B7): make CUDA training runs reproducible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "cuda") -> torch.device:
    """Resolve an NVIDIA CUDA device for training; never fall back to CPU."""
    if preference == "auto":  # Backward-compatible alias for existing commands.
        preference = "cuda"
    device = torch.device(preference)
    if device.type != "cuda":
        raise ValueError("Training requires an NVIDIA CUDA GPU; CPU training is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training requested but PyTorch cannot access an NVIDIA GPU. "
            "Install CUDA-enabled PyTorch and a compatible NVIDIA driver."
        )
    return device


def build_model(config: dict) -> nn.Module:
    """Create AstrocyteUNet from config (architecture/encoder/input channels)."""
    model_cfg = config["model"]
    architecture = model_cfg.get("architecture", "unet")
    if architecture != "unet":
        raise ValueError(f"Unsupported architecture: {architecture!r}")
    in_channels = _strict_int(config["data"].get("input_channels", 1), "data.input_channels", 1)
    # FIX (review B5): catch a stale output_channels config value instead of
    # silently training against a mismatched output-head contract.
    declared = model_cfg.get("output_channels", 4)
    declared = _strict_int(declared, "model.output_channels", 1)
    expected = NUM_SEMANTIC_CLASSES + 1  # 3 semantic classes + 1 center channel
    if declared != expected:
        raise ValueError(
            f"model.output_channels={declared} but the REDESIGN model emits "
            f"{expected} channels; update configs/config.yaml."
        )
    return create_model(in_channels=in_channels,
                        encoder_name=model_cfg.get("encoder", "resnet18"))


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create the optimizer from config (currently Adam)."""
    train_cfg = config["training"]
    lr = _finite_number(train_cfg.get("learning_rate", 1e-4), "training.learning_rate")
    if lr <= 0:
        raise ValueError("training.learning_rate must be finite and positive")
    name = str(train_cfg.get("optimizer", "adam")).lower()
    if name != "adam":
        raise ValueError(f"Unsupported optimizer: {name!r}")
    return torch.optim.Adam(model.parameters(), lr=lr)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

class CompositeLoss(nn.Module):
    """Semantic cross-entropy + center BCE-with-logits, masked by valid_mask.

    config.training.loss == "bce_with_logits" selects this composite loss:
    the center head uses BCE-with-logits and the semantic head uses plain
    cross-entropy over the 3 mutually exclusive classes. Pixels where
    valid_mask is False (unlabeled/ambiguous) are excluded from both terms.
    """

    def __init__(self, semantic_weight: float = 1.0, center_weight: float = 1.0):
        super().__init__()
        for name, w in (("semantic_weight", semantic_weight), ("center_weight", center_weight)):
            if not np.isfinite(w) or w < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if semantic_weight == 0 and center_weight == 0:
            raise ValueError("At least one loss weight must be positive")
        self.semantic_weight = float(semantic_weight)
        self.center_weight = float(center_weight)

    def forward(self, outputs: dict, targets: dict) -> dict:
        semantic_logits = outputs["semantic_logits"]
        center_logits = outputs["center_logits"]
        semantic_mask = targets["semantic_mask"]
        center_heatmap = targets["center_heatmap"].float()
        valid_mask = targets["valid_mask"].bool()
        semantic_valid_mask = targets.get("semantic_valid_mask", valid_mask).bool()

        if semantic_logits.ndim != 4 or semantic_logits.shape[1] != NUM_SEMANTIC_CLASSES:
            raise ValueError(f"semantic_logits must be [B,{NUM_SEMANTIC_CLASSES},H,W]")
        if center_logits.ndim != 4 or center_logits.shape[1] != 1:
            raise ValueError("center_logits must be [B,1,H,W]")
        if semantic_logits.shape[0] != center_logits.shape[0] or semantic_logits.shape[2:] != center_logits.shape[2:]:
            raise ValueError("Semantic and center logits must have matching batch and spatial shapes")
        if semantic_mask.shape != semantic_logits.shape[:1] + semantic_logits.shape[2:]:
            raise ValueError("semantic_mask shape must match logits [B,H,W]")
        if semantic_mask.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("semantic_mask must contain integer labels")
        semantic_mask = semantic_mask.long()
        if center_heatmap.shape != center_logits.shape:
            raise ValueError("center_heatmap must match center_logits [B,1,H,W]")
        if valid_mask.shape != semantic_mask.shape or semantic_valid_mask.shape != semantic_mask.shape:
            raise ValueError("valid masks must match semantic_mask [B,H,W]")
        if not torch.isfinite(center_heatmap).all() or torch.any(center_heatmap < 0) or torch.any(center_heatmap > 1):
            raise ValueError("center_heatmap must contain finite values in [0,1]")
        # FIX [Liu]: semantic validity must also respect the shared valid_mask.
        semantic_valid_mask = semantic_valid_mask & valid_mask
        if semantic_valid_mask.any() and (
            torch.any(semantic_mask[semantic_valid_mask] < 0)
            or torch.any(semantic_mask[semantic_valid_mask] >= NUM_SEMANTIC_CLASSES)
        ):
            raise ValueError("semantic_mask labels must be 0, 1, or 2 on valid pixels")

        zero = (semantic_logits.sum() + center_logits.sum()) * 0.0
        n_center_valid = int(valid_mask.sum().item())
        n_semantic_valid = int(semantic_valid_mask.sum().item())

        if n_semantic_valid:
            ce = nn.functional.cross_entropy(semantic_logits, semantic_mask, reduction="none")
            semantic_loss = (ce * semantic_valid_mask).sum() / n_semantic_valid
        else:
            semantic_loss = zero

        if n_center_valid:
            bce = nn.functional.binary_cross_entropy_with_logits(
                center_logits, center_heatmap, reduction="none"
            )
            center_loss = (bce.squeeze(1) * valid_mask).sum() / n_center_valid
        else:
            center_loss = zero

        loss = self.semantic_weight * semantic_loss + self.center_weight * center_loss
        return {"loss": loss, "semantic_loss": semantic_loss, "center_loss": center_loss}


def build_criterion(config: dict) -> CompositeLoss:
    """Build the loss from config; only bce_with_logits is currently supported."""
    train_cfg = config["training"]
    name = str(train_cfg.get("loss", "bce_with_logits")).lower()
    if name != "bce_with_logits":
        raise ValueError(f"Unsupported loss: {name!r}")
    return CompositeLoss(
        semantic_weight=_finite_number(train_cfg.get("semantic_weight", 1.0), "training.semantic_weight"),
        center_weight=_finite_number(train_cfg.get("center_weight", 1.0), "training.center_weight"),
    )


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #

def _split_batch(batch):
    """Accept (image, targets) or (image, targets, metadata) batches."""
    if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
        raise ValueError("Batches must be (image, targets[, metadata])")
    images, targets = batch[0], batch[1]
    if not isinstance(targets, dict):
        raise ValueError("targets must be a dict of tensors")
    for key in ("semantic_mask", "center_heatmap", "valid_mask"):
        if key not in targets:
            raise ValueError(f"targets is missing required key: {key!r}")
    return images, targets


def _to_device(images, targets, device):
    return images.to(device), {k: v.to(device) for k, v in targets.items()}


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    """Run one training (optimizer given) or evaluation (optimizer=None) epoch."""
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "semantic_loss": 0.0, "center_loss": 0.0}
    n_samples = 0
    for batch in loader:
        images, targets = _split_batch(batch)
        images, targets = _to_device(images, targets, device)
        batch_size = int(images.shape[0])
        with torch.set_grad_enabled(training):
            outputs = model(images)
            losses = criterion(outputs, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
        # FIX (review B6): weight batch means by batch size so a short final
        # batch is not over-represented in the epoch average.
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
        n_samples += batch_size
    if n_samples == 0:
        raise ValueError("DataLoader produced no batches")
    return {key: value / n_samples for key, value in totals.items()}


# --------------------------------------------------------------------------- #
# Checkpoints and history
# --------------------------------------------------------------------------- #

def save_checkpoint(model, optimizer, epoch: int, metrics: dict, config: dict,
                    path: str | Path) -> str:
    """Save a checkpoint with the schema predict.load_trained_model() expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "config": config,
            # REDESIGN: record the output-head schema so legacy single-heatmap
            # weights can be rejected instead of silently mis-loaded.
            "output_heads": {"semantic_classes": NUM_SEMANTIC_CLASSES, "center_channels": 1},
        },
        str(path),
    )
    return str(path)


def train(config: dict, device: torch.device, progress=print) -> dict:
    """Full training run: dataloaders -> epochs -> checkpoints + history.json.

    IMPLEMENTED [Liu]: training budget + early stopping (patience) driven by
    validation loss; best and last checkpoints are both kept. NVIDIA CUDA is
    mandatory for real training, including direct Python API calls.
    """
    device = get_device(str(device))
    # FIX (review B1): same package/script import branch as the model import,
    # so `python -m src.train` works as well as `python src/train.py`.
    # IMPLEMENTED [Liu]: real BBBC042 loaders now use the same target contract.
    if __package__:
        from .dataset import build_dataloaders
    else:
        from dataset import build_dataloaders

    validate_config(config)
    train_cfg = config["training"]
    epochs = train_cfg.get("epochs", 50)
    patience = train_cfg.get("early_stopping_patience", 10)
    checkpoint_root = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    if not checkpoint_root.is_absolute():
        checkpoint_root = PROJECT_ROOT / checkpoint_root
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        name = f"run_{timestamp}" if suffix == 0 else f"run_{timestamp}_{suffix:03d}"
        checkpoint_dir = checkpoint_root / name
        try:
            checkpoint_dir.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError(f"Could not allocate a unique run directory in {checkpoint_root}")
    config["training"]["checkpoint_dir"] = str(checkpoint_dir)
    with (checkpoint_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    history_path = checkpoint_dir / "history.json"

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    criterion = build_criterion(config)
    train_loader, val_loader = build_dataloaders(config)

    history = []
    best_val = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        record = {"epoch": epoch, "seconds": round(time.time() - started, 3)}
        record.update({f"train_{k}": v for k, v in train_metrics.items()})
        record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(record)
        # FIX: persist history after EVERY epoch so a mid-run crash does not
        # lose the whole training curve.
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        progress(f"epoch {epoch}/{epochs} " +
                 " ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "epoch"))

        save_checkpoint(model, optimizer, epoch, val_metrics, config,
                        checkpoint_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, config,
                            checkpoint_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                progress(f"Early stopping after {epoch} epochs "
                         f"(no val improvement for {patience})")
                break

    progress(f"History saved: {history_path}")
    return {"history": history, "best_val_loss": best_val,
            "checkpoint_dir": str(checkpoint_dir)}


# --------------------------------------------------------------------------- #
# Synthetic self-test (verifies the training loop without real data)
# --------------------------------------------------------------------------- #

class SyntheticAstrocyteDataset(Dataset):
    """Tiny synthetic dataset matching the REDESIGN targets contract.

    Each sample is a [1,H,W] fluorescence-like image with a few bright Gaussian
    somas; targets are semantic_mask (0/1/2), center_heatmap [1,H,W] and an
    all-True valid_mask. Used by --self-test only.
    """

    def __init__(self, n_samples: int = 16, image_size: int = 128, seed: int = 0):
        if n_samples <= 0 or image_size < 32:
            raise ValueError("n_samples must be positive and image_size >= 32")
        self.n_samples = n_samples
        self.image_size = image_size
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed * 100003 + index)
        size = self.image_size
        yy, xx = np.mgrid[0:size, 0:size]
        image = rng.normal(0.05, 0.02, (size, size)).astype(np.float32)
        semantic = np.zeros((size, size), dtype=np.int64)
        heatmap = np.zeros((size, size), dtype=np.float32)
        for _ in range(rng.integers(2, 6)):
            cx, cy = rng.integers(16, size - 16, 2)
            radius = rng.integers(4, 8)
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius ** 2))
            image += (0.6 + 0.4 * rng.random()) * blob
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            semantic[disk] = 1
            ring = ((xx - cx) ** 2 + (yy - cy) ** 2 <= (2 * radius) ** 2) & ~disk
            semantic[ring & (rng.random((size, size)) > 0.7)] = 2  # sparse processes
            heatmap = np.maximum(heatmap, np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2)).astype(np.float32))
        image = np.clip(image, 0, None)
        image /= image.max() if image.max() > 0 else 1.0
        return (
            torch.from_numpy(image[None].astype(np.float32)),
            {
                "semantic_mask": torch.from_numpy(semantic),
                "center_heatmap": torch.from_numpy(heatmap[None]),
                "valid_mask": torch.ones((size, size), dtype=torch.bool),
            },
            {"sample_id": f"synthetic_{index:04d}"},
        )


def self_test(config: dict, device: torch.device, epochs: int = 3,
              progress=print) -> dict:
    """Run the full loop on synthetic data; assert the training loss decreases."""
    # FIX (review B3): with a single epoch, first == last and the
    # "loss must decrease" assertion would always misfire.
    if not isinstance(epochs, (int, np.integer)) or epochs < 2:
        raise ValueError("self_test needs epochs >= 2 to detect a loss decrease")
    train_cfg = config["training"]
    loader_kwargs = {"batch_size": min(int(train_cfg.get("batch_size", 8)), 4),
                     "num_workers": 0}
    train_loader = DataLoader(SyntheticAstrocyteDataset(16, seed=0), shuffle=True,
                              **loader_kwargs)
    val_loader = DataLoader(SyntheticAstrocyteDataset(4, seed=1), shuffle=False,
                            **loader_kwargs)

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    criterion = build_criterion(config)

    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()}})
        progress(f"self-test epoch {epoch}/{epochs} "
                 f"train_loss={train_metrics['loss']:.4f} "
                 f"val_loss={val_metrics['loss']:.4f}")

    first, last = history[0]["train_loss"], history[-1]["train_loss"]
    if not np.isfinite(last) or last >= first:
        raise RuntimeError(
            f"Training loss did not decrease on synthetic data: {first:.4f} -> {last:.4f}"
        )

    # Verify checkpoint save/reload round-trip.
    # FIX: keep self-test artifacts out of the source tree (temp dir).
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="astrocyte_selftest_"))
    saved = save_checkpoint(model, optimizer, epochs, history[-1], config,
                            checkpoint_dir / "best.pt")
    reloaded = build_model(config)
    state = torch.load(saved, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(state["model_state_dict"])
    progress(f"Self-test passed: loss {first:.4f} -> {last:.4f}; checkpoint {saved}")
    return {"history": history, "checkpoint": saved}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train AstrocyteUNet")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="Path to config.yaml")
    parser.add_argument("--device", default=None, help="cuda or cuda:N (default: training.device or cuda)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs (self-test default: 3)")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the synthetic-data training-loop verification")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config["training"]["epochs"] = args.epochs
    set_seed(config.get("project", {}).get("seed", 42))
    device = get_device(args.device or config.get("training", {}).get("device", "cuda"))
    print(f"device: {device}")

    if args.self_test:
        self_test(config, device, epochs=args.epochs or 3)
        return

    try:
        train(config, device)
    except NotImplementedError as exc:
        raise SystemExit(
            "Real-data training is blocked: dataset.build_dataloaders() is not "
            f"implemented yet ({exc}). Use --self-test to verify the training loop."
        ) from exc


if __name__ == "__main__":
    main()
