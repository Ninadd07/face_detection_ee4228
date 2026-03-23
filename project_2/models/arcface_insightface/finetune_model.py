import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .finetune_config import (
    FINETUNE_EMBED_DIM,
    FINETUNE_MODEL_NAME,
    FINETUNE_FREEZE_BACKBONE,
    FINETUNE_UNFREEZE_LAST_BLOCK,
)


class FinetunedFaceNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        if FINETUNE_MODEL_NAME == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feat_dim = backbone.fc.in_features
        else:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feat_dim = backbone.fc.in_features

        self.backbone = backbone
        self.backbone.fc = nn.Identity()

        if FINETUNE_FREEZE_BACKBONE:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if FINETUNE_UNFREEZE_LAST_BLOCK:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

        self.embedding_head = nn.Sequential(
            nn.Linear(feat_dim, FINETUNE_EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(FINETUNE_EMBED_DIM, num_classes)

    def forward(self, x, return_embedding=False):
        feats = self.backbone(x)
        emb = self.embedding_head(feats)
        emb = F.normalize(emb, p=2, dim=1)

        if return_embedding:
            return emb

        logits = self.classifier(emb)
        return logits
