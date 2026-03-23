import cv2
import numpy as np
import torch
from torchvision import transforms

from .finetune_config import FINETUNE_IMAGE_SIZE, FINETUNE_CHECKPOINTS_DIR
from .finetune_model import FinetunedFaceNet
from .utils import normalize_embedding


class FinetunedEmbedder:
    def __init__(self):
        ckpt = torch.load(FINETUNE_CHECKPOINTS_DIR / "best_finetuned_model.pth", map_location="cpu")
        num_classes = len(ckpt["label_to_idx"])
        self.model = FinetunedFaceNet(num_classes=num_classes)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def crop_from_detection(self, image, bbox, pad_ratio=0.35):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * pad_ratio / 2)
        py = int(bh * pad_ratio / 2)

        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)

        crop = image[y1:y2, x1:x2]
        return crop

    def get_embedding(self, image, detection):
        bbox = detection["bbox"]
        crop = self.crop_from_detection(image, bbox)

        if crop.size == 0:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (FINETUNE_IMAGE_SIZE, FINETUNE_IMAGE_SIZE))
        tensor = self.tf(crop).unsqueeze(0)

        with torch.no_grad():
            emb = self.model(tensor, return_embedding=True).squeeze(0).cpu().numpy()

        return normalize_embedding(emb.astype(np.float32))
