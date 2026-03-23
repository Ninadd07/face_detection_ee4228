from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import ENROLL_DIR, VAL_DIR
from .finetune_config import (
    FINETUNE_BATCH_SIZE,
    FINETUNE_NUM_EPOCHS,
    FINETUNE_LR,
    FINETUNE_WEIGHT_DECAY,
    FINETUNE_CHECKPOINTS_DIR,
    FINETUNE_LOGS_DIR,
)
from .finetune_dataset import FaceClassificationDataset
from .finetune_model import FinetunedFaceNet


def build_label_map(root_dir):
    labels = sorted([p.name for p in Path(root_dir).iterdir() if p.is_dir()])
    return {label: idx for idx, label in enumerate(labels)}


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total if total > 0 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_to_idx = build_label_map(ENROLL_DIR)

    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    train_ds = FaceClassificationDataset(ENROLL_DIR, label_to_idx, transform=train_tf)
    val_ds = FaceClassificationDataset(VAL_DIR, label_to_idx, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)

    model = FinetunedFaceNet(num_classes=len(label_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR,
        weight_decay=FINETUNE_WEIGHT_DECAY,
    )

    best_val_acc = -1.0
    history = []

    for epoch in range(FINETUNE_NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_ds)
        val_acc = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_acc": val_acc,
        })

        print(f"[finetune_train] epoch={epoch+1} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = FINETUNE_CHECKPOINTS_DIR / "best_finetuned_model.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
            }, ckpt_path)

    with open(FINETUNE_LOGS_DIR / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[finetune_train] Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
