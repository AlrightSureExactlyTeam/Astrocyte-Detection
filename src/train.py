"""Train the U-Net center-heatmap model on BBBC042 / FFPE tiles.
Owner: Liu | Inputs: config, DataLoaders, model | Outputs: checkpoints, training logs.
"""
import torch
from pathlib import Path


def create_loss(config: dict):
    """Return the loss function for predicted logits and target heatmaps."""
    # TODO [Liu]: choose a consistent loss (e.g. BCEWithLogitsLoss or sigmoid + MSE).
    # TODO [Liu]: check foreground/background imbalance and output activation.
    raise NotImplementedError("TODO: configure heatmap loss")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Run one training epoch and return mean loss."""
    # TODO [Liu]: model.train(), move tensors to device, forward pass.
    # TODO [Liu]: compute loss, backward(), optimizer.step(), track average loss.
    raise NotImplementedError("TODO: training epoch")


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    """Run validation without parameter updates; return mean loss."""
    # TODO [Liu]: model.eval(), forward pass, compute average validation loss.
    raise NotImplementedError("TODO: validation epoch")


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, path: str):
    """Save model and optimizer state, epoch, and validation metrics."""
    # TODO [Liu]: create parent folder and save a torch checkpoint.
    raise NotImplementedError("TODO: checkpoint saving")


def main():
    # TODO [Liu]: read config.yaml and set random seed / CUDA device.
    # TODO [Liu]: import create_model from model.py and loaders from dataset.py.
    # TODO [Liu]: build optimizer, run epochs, log losses, save best and last weights.
    raise NotImplementedError("TODO: connect training pipeline")


if __name__ == "__main__":
    main()
