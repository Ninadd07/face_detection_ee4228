from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset

from .finetune_config import FINETUNE_IMAGE_SIZE
from .utils import list_images


class FaceClassificationDataset(Dataset):
    def __init__(self, root_dir, label_to_idx, transform=None):
        self.root_dir = Path(root_dir)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.samples = []

        for person_dir in sorted(self.root_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            label = person_dir.name
            if label not in self.label_to_idx:
                continue
            for img_path in list_images(person_dir):
                self.samples.append((img_path, self.label_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (FINETUNE_IMAGE_SIZE, FINETUNE_IMAGE_SIZE))

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label
