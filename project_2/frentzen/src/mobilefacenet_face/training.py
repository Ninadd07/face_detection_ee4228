from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import load_config
from .recognizers.mobilefacenet import MobileFaceNet, NormalizedClassifierHead, save_backbone_checkpoint
from .sizes import normalize_input_size


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(slots=True)
class AlignedSample:
    identity: str
    aligned_path: Path
    split: str
    group_id: str


def read_aligned_manifest(path: str | Path) -> list[AlignedSample]:
    rows: list[AlignedSample] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("success", "").lower() != "true":
                continue
            aligned_path = Path(row["aligned_path"])
            if not aligned_path.exists():
                continue
            rows.append(
                AlignedSample(
                    identity=row["identity"],
                    aligned_path=aligned_path,
                    split=row["split"],
                    group_id=row["group_id"],
                )
            )
    return rows


def build_label_mapping(samples: list[AlignedSample]) -> dict[str, int]:
    identities = sorted({sample.identity for sample in samples if sample.split == "enrollment"})
    return {identity: index for index, identity in enumerate(identities)}


def select_group_limited_samples(
    samples: list[AlignedSample],
    split: str,
    max_samples_per_group: int,
    seed: int,
) -> list[AlignedSample]:
    by_group: dict[tuple[str, str], list[AlignedSample]] = {}
    for sample in samples:
        if sample.split != split:
            continue
        by_group.setdefault((sample.identity, sample.group_id), []).append(sample)
    rng = random.Random(seed)
    selected: list[AlignedSample] = []
    for key in sorted(by_group):
        group_samples = sorted(by_group[key], key=lambda item: item.aligned_path.name)
        rng.shuffle(group_samples)
        selected.extend(group_samples[:max_samples_per_group])
    return selected


class AlignedFaceDataset(Dataset):
    def __init__(
        self,
        samples: list[AlignedSample],
        label_mapping: dict[str, int],
        train: bool,
        input_size: tuple[int, int],
        augmentation_cfg: dict,
    ) -> None:
        self.samples = samples
        self.label_mapping = label_mapping
        self.train = train
        self.input_size = input_size
        self.transform = self._build_transform(augmentation_cfg)

    def _build_transform(self, augmentation_cfg: dict) -> transforms.Compose:
        ops: list = [transforms.ToPILImage(), transforms.Resize(self.input_size)]
        if self.train:
            if augmentation_cfg.get("horizontal_flip", True):
                ops.append(transforms.RandomHorizontalFlip())
            if augmentation_cfg.get("color_jitter", True):
                ops.append(
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.1,
                        hue=0.02,
                    )
                )
        ops.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Lambda(lambda tensor: (tensor * 255.0 - 127.5) / 128.0),
            ]
        )
        if self.train and augmentation_cfg.get("random_erasing", False):
            ops.append(transforms.RandomErasing(p=0.25, value=0.0))
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image_bgr = cv2.imread(str(sample.aligned_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read aligned face: {sample.aligned_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb)
        label = self.label_mapping[sample.identity]
        return tensor, label


class MobileFaceNetTrainer(nn.Module):
    def __init__(self, backbone: MobileFaceNet, num_classes: int, embedding_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = NormalizedClassifierHead(embedding_dim=embedding_dim, num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.backbone(images)
        logits = self.head(embeddings)
        return embeddings, logits


def create_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_training_checkpoint(checkpoint_path: str | Path) -> dict:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint}")
    if checkpoint.stat().st_size == 0:
        raise ValueError(f"Training checkpoint is empty: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu")
    except EOFError as exc:
        raise ValueError(f"Training checkpoint is corrupted or incomplete: {checkpoint}") from exc
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unsupported pretrained MobileFaceNet checkpoint format.")


def run_epoch(
    model: MobileFaceNetTrainer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        _, logits = model(images)
        loss = criterion(logits, labels)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.size(0))
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "samples": total,
    }


def train_mobilefacenet(config_path: str | Path | None = None) -> dict:
    config = load_config(config_path)
    seed_everything(int(config["training"]["random_seed"]))
    checkpoint_path = config["recognizer"].get("checkpoint_path")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError("Training requires recognizer.checkpoint_path to point to a pretrained MobileFaceNet checkpoint.")

    aligned_manifest = Path(config["preprocess"]["aligned_manifest"])
    if not aligned_manifest.exists():
        raise FileNotFoundError("Aligned manifest not found. Run mobilefacenet-prepare first.")

    samples = read_aligned_manifest(aligned_manifest)
    label_mapping = build_label_mapping(samples)
    if not label_mapping:
        raise RuntimeError("No training identities found in aligned manifest.")

    train_samples = select_group_limited_samples(
        samples,
        split="enrollment",
        max_samples_per_group=int(config["training"]["max_samples_per_group"]),
        seed=int(config["training"]["random_seed"]),
    )
    val_samples = select_group_limited_samples(
        samples,
        split="validation",
        max_samples_per_group=int(config["training"]["max_samples_per_group"]),
        seed=int(config["training"]["random_seed"]),
    )
    if not train_samples or not val_samples:
        raise RuntimeError("Training and validation aligned samples are required.")

    device = resolve_device(config.get("device", "auto"))

    input_size = normalize_input_size(config["recognizer"]["input_size"])
    augmentation_cfg = config["training"].get("augmentation", {})
    train_dataset = AlignedFaceDataset(train_samples, label_mapping, train=True, input_size=input_size, augmentation_cfg=augmentation_cfg)
    val_dataset = AlignedFaceDataset(val_samples, label_mapping, train=False, input_size=input_size, augmentation_cfg=augmentation_cfg)
    loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    backbone = MobileFaceNet(embedding_dim=int(config["recognizer"]["embedding_dim"]))
    payload = load_training_checkpoint(checkpoint_path)
    state_dict = (
        payload.get("net_state_dict")
        or payload.get("model_state_dict")
        or payload.get("state_dict")
        or payload.get("model")
        or payload
    )
    cleaned = {key.replace("module.", ""): value for key, value in state_dict.items()}
    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    if missing:
        raise ValueError(f"Pretrained checkpoint missing keys: {missing[:5]}")
    if unexpected:
        raise ValueError(f"Pretrained checkpoint has unexpected keys: {unexpected[:5]}")
    trainer = MobileFaceNetTrainer(backbone=backbone, num_classes=len(label_mapping), embedding_dim=int(config["recognizer"]["embedding_dim"])).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["training"]["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        trainer.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = create_scheduler(
        optimizer,
        warmup_epochs=int(config["training"]["warmup_epochs"]),
        total_epochs=int(config["training"]["epochs"]),
    )

    best_metric = float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    selection_metric = config["training"].get("selection_metric", "val_accuracy")
    total_epochs = int(config["training"]["epochs"])

    print(
        f"Training MobileFaceNet on {device} with {len(train_samples)} train and {len(val_samples)} val samples "
        f"for up to {total_epochs} epochs.",
        flush=True,
    )

    for epoch in range(total_epochs):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(trainer, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(trainer, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        epoch_metrics = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_metrics)
        metric_value = epoch_metrics[selection_metric]
        save_backbone_checkpoint(trainer.backbone, config["training"]["last_checkpoint_path"], epoch + 1, metrics=epoch_metrics)
        if metric_value > best_metric:
            best_metric = metric_value
            epochs_without_improvement = 0
            save_backbone_checkpoint(trainer.backbone, config["training"]["best_checkpoint_path"], epoch + 1, metrics=epoch_metrics)
        else:
            epochs_without_improvement += 1
        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch + 1}/{total_epochs} "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"time={epoch_seconds:.1f}s",
            flush=True,
        )
        if epochs_without_improvement >= int(config["training"]["early_stopping_patience"]):
            print(
                f"Early stopping triggered after epoch {epoch + 1} "
                f"with best {selection_metric}={best_metric:.4f}.",
                flush=True,
            )
            break

    training_state = {
        "label_mapping": label_mapping,
        "best_metric": best_metric,
        "selection_metric": selection_metric,
        "epochs_completed": len(history),
        "pretrained_checkpoint_path": checkpoint_path,
        "best_checkpoint_path": config["training"]["best_checkpoint_path"],
        "last_checkpoint_path": config["training"]["last_checkpoint_path"],
    }
    state_path = Path(config["training"]["training_state_path"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(training_state, handle, indent=2)

    metrics_path = Path(config["training"]["training_metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"history": history, "best_metric": best_metric, "selection_metric": selection_metric}, handle, indent=2)

    return {
        "history": history,
        "best_metric": best_metric,
        "best_checkpoint_path": config["training"]["best_checkpoint_path"],
        "last_checkpoint_path": config["training"]["last_checkpoint_path"],
    }
