from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..gallery import l2_normalize
from ..sizes import normalize_input_size
from .base import FaceEmbedder


class Bottleneck(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expansion: int) -> None:
        super().__init__()
        self.connect = stride == 1 and inp == oup
        expanded = inp * expansion
        self.conv = nn.Sequential(
            nn.Conv2d(inp, expanded, 1, 1, 0, bias=False),
            nn.BatchNorm2d(expanded),
            nn.PReLU(expanded),
            nn.Conv2d(expanded, expanded, 3, stride, 1, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.PReLU(expanded),
            nn.Conv2d(expanded, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.connect:
            return x + self.conv(x)
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        kernel: int | tuple[int, int],
        stride: int,
        padding: int,
        *,
        dw: bool = False,
        linear: bool = False,
    ) -> None:
        super().__init__()
        self.linear = linear
        groups = inp if dw else 1
        self.conv = nn.Conv2d(inp, oup, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        return self.prelu(x)


MOBILEFACENET_BOTTLENECK_SETTING = [
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1],
]


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        if embedding_dim != 128:
            raise ValueError("Xiaoccer-compatible MobileFaceNet expects embedding_dim=128.")
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.inplanes = 64
        self.blocks = self._make_layer(Bottleneck, MOBILEFACENET_BOTTLENECK_SETTING)
        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, (7, 6), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, embedding_dim, 1, 1, 0, linear=True)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                kernel_h, kernel_w = module.kernel_size
                n = kernel_h * kernel_w * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(self, block: type[Bottleneck], setting: list[list[int]]) -> nn.Sequential:
        layers: list[nn.Module] = []
        for expansion, channels, repeats, stride in setting:
            for index in range(repeats):
                layers.append(block(self.inplanes, channels, stride if index == 0 else 1, expansion))
                self.inplanes = channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        return x.view(x.size(0), -1)


class NormalizedClassifierHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 32.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        features = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(features, weights)


class MobileFaceNetEmbedder(FaceEmbedder):
    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str | None = None,
        embedding_dim: int = 128,
        input_size: int | list[int] | tuple[int, int] = (112, 96),
    ) -> None:
        self.device = device
        self.input_size = normalize_input_size(input_size)
        self.embedding_dim = embedding_dim
        self.checkpoint_path = checkpoint_path
        self.model = MobileFaceNet(embedding_dim=embedding_dim).to(device)
        self.model.eval()
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"MobileFaceNet checkpoint not found: {checkpoint}")
        if checkpoint.stat().st_size == 0:
            raise ValueError(f"MobileFaceNet checkpoint is empty: {checkpoint}")
        try:
            payload = torch.load(checkpoint, map_location="cpu")
        except EOFError as exc:
            raise ValueError(f"MobileFaceNet checkpoint is corrupted or incomplete: {checkpoint}") from exc
        if isinstance(payload, dict):
            for key in ("net_state_dict", "state_dict", "model_state_dict", "model"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
        if not isinstance(payload, dict):
            raise ValueError("Unsupported MobileFaceNet checkpoint format.")
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            normalized_key = key.replace("module.", "")
            if normalized_key.startswith("backbone."):
                normalized_key = normalized_key.removeprefix("backbone.")
            cleaned[normalized_key] = value
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            raise ValueError(f"MobileFaceNet checkpoint is missing keys: {missing[:5]}")
        if unexpected:
            raise ValueError(f"MobileFaceNet checkpoint has unexpected keys: {unexpected[:5]}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self._load_checkpoint(checkpoint_path)

    def embed(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        if not aligned_faces:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        batch = np.stack([self.preprocess(face) for face in aligned_faces]).astype(np.float32)
        tensor = torch.from_numpy(batch).to(self.device)
        with torch.inference_mode():
            embeddings = self.model(tensor).detach().cpu().numpy()
        return l2_normalize(embeddings.astype(np.float32))

    def preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        expected_height, expected_width = self.input_size
        if aligned_face.shape[:2] != (expected_height, expected_width):
            raise ValueError(
                f"Aligned face must be {expected_height}x{expected_width}, got {aligned_face.shape[:2]}"
            )
        normalized = (aligned_face.astype(np.float32) - 127.5) / 128.0
        return np.transpose(normalized, (2, 0, 1))

    def embed_tensor(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(batch_tensor)

    def checkpoint_size_bytes(self) -> int | None:
        if not self.checkpoint_path:
            return None
        path = Path(self.checkpoint_path)
        return path.stat().st_size if path.exists() else None


def save_backbone_checkpoint(
    model: MobileFaceNet,
    path: str | Path,
    epoch: int,
    metrics: dict[str, float] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
        },
        target,
    )
