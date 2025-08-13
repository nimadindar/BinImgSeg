import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class ScribbleSegTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.scribbles_dir = os.path.join(root_dir, "scribbles")
        self.masks_dir = os.path.join(root_dir, "ground_truth")
        self.transform = transform

        self.image_ids = [os.path.splitext(f)[0] for f in os.listdir(self.images_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        img_path = os.path.join(self.images_dir, img_id + ".jpg")
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        scribble_path = os.path.join(self.scribbles_dir, img_id + ".png")
        scribble = np.array(Image.open(scribble_path), dtype=np.int64)

        mask_path = os.path.join(self.masks_dir, img_id + ".png")
        mask = np.array(Image.open(mask_path), dtype=np.int64)

        class_label = torch.tensor(int(mask.max() > 0), dtype=torch.long)

        if self.transform:
            image, scribble, mask = self.transform(image, scribble, mask)

        return image, scribble, mask, class_label


class ScribbleSegTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to test dataset folder (contains 'images', 'scribbles')
            transform: Transform for images and scribbles
        """
        self.images_dir = os.path.join(root_dir, "images")
        self.scribbles_dir = os.path.join(root_dir, "scribbles")

        self.image_files = sorted(os.listdir(self.images_dir))
        self.scribble_files = sorted(os.listdir(self.scribbles_dir))

        assert len(self.image_files) == len(self.scribble_files), \
            "Mismatch between images and scribbles counts."

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        scribble_path = os.path.join(self.scribbles_dir, self.scribble_files[idx])

        image = Image.open(img_path).convert("RGB")
        scribble = Image.open(scribble_path).convert("L")

        if self.transform:
            image = self.transform(image)
            scribble = self.transform(scribble)

        return image, scribble

