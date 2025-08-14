import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

class ScribbleSegDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, load_gt=False):
        """
        Args:
            root_dir (str): Path to dataset folder (e.g., ./dataset/train or ./dataset/test1)
            split (str): 'train' or 'test' to determine behavior
            transform: Transform for images and scribbles
            load_gt (bool): Whether to load ground truth (for validation in train split)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.load_gt = load_gt

        self.images_dir = os.path.join(root_dir, "images")
        self.scribbles_dir = os.path.join(root_dir, "scribbles")
        if self.load_gt:
            self.masks_dir = os.path.join(root_dir, "ground_truth")

        self.image_ids = [os.path.splitext(f)[0] for f in os.listdir(self.images_dir) if f.endswith(".jpg")]
        if split == "test":
            self.scribble_ids = [os.path.splitext(f)[0] for f in os.listdir(self.scribbles_dir) if f.endswith(".png")]
            assert len(self.image_ids) == len(self.scribble_ids), "Mismatch between images and scribbles counts."
            self.image_ids = sorted(self.image_ids) 

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        img_path = os.path.join(self.images_dir, img_id + ".jpg")
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.astype(np.float32) / 255.0  # Normalize to [0,1]

        scribble_path = os.path.join(self.scribbles_dir, img_id + ".png")
        scribble = np.array(Image.open(scribble_path).convert('L'), dtype=np.int64)  
        scribble = scribble.squeeze()  # Remove single-channel dimension if present (e.g., (H, W, 1) -> (H, W))
        if len(scribble.shape) != 2:
            raise ValueError(f"Scribble {img_id} has invalid shape {scribble.shape}. Expected 2D array (H, W).")

        scribble = np.where(scribble > 0, 1, scribble)  # Binary: 1 for train, 0/255 for bg/ignore

        class_label = torch.tensor([0, int(scribble.max() > 0)], dtype=torch.long)  # [bg=0, fg=1]

        mask = None
        if self.load_gt and self.split == "train":
            mask_path = os.path.join(self.masks_dir, img_id + ".png")
            mask = np.array(Image.open(mask_path).convert('L'), dtype=np.int64) 
            mask = mask.squeeze()
            if len(mask.shape) != 2:
                raise ValueError(f"Mask {img_id} has invalid shape {mask.shape}. Expected 2D array (H, W).")
            # Binary: 0=background, 1=foreground
            mask = np.where(mask > 0, 1, 0)

        if self.transform is not None:
            image, scribble = self.transform(image, scribble)

        if mask is not None and self.transform is not None:
            h, w = image.shape[1], image.shape[2] if isinstance(image, torch.Tensor) else image.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.int64)
            mask = np.where(mask > 0, 1, 0)  

        # self.check_shapes(image, scribble, mask)

        if mask is not None:
            return image, scribble, mask, class_label, img_path
        return image, scribble, class_label, img_path

    def check_shapes(self, image, scribble, mask=None):
        if image.shape[:2] != scribble.shape:
            raise RuntimeError(f"Shape mismatch: image {image.shape}, scribble {scribble.shape}")
        if mask is not None and image.shape[:2] != mask.shape:
            raise RuntimeError(f"Shape mismatch: image {image.shape}, mask {mask.shape}")