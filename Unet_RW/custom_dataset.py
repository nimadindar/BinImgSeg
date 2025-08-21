from PIL import Image
from typing import Optional
from dataclasses import dataclass
import os, cv2, glob, numpy as np, albumentations as A

import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from .utils import imread, imread_gray


    
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

        scribble_path = os.path.join(self.scribbles_dir, img_id + ".png")
        scribble = np.array(Image.open(scribble_path).convert('L'), dtype=np.int64)  
        scribble = scribble.squeeze() 
        if len(scribble.shape) != 2:
            raise ValueError(f"Scribble {img_id} has invalid shape {scribble.shape}. Expected 2D array (H, W).")

        scribble = np.where(scribble > 0, 1, scribble)   

        class_label = torch.tensor([0, int(scribble.max() > 0)], dtype=torch.long)  

        mask = None
        if self.load_gt and self.split == "train":
            mask_path = os.path.join(self.masks_dir, img_id + ".png")
            mask = np.array(Image.open(mask_path).convert('L'), dtype=np.int64) 
            mask = mask.squeeze()
            if len(mask.shape) != 2:
                raise ValueError(f"Mask {img_id} has invalid shape {mask.shape}. Expected 2D array (H, W).")

            mask = np.where(mask > 0, 1, 0)

        if self.transform is not None:
            image, scribble = self.transform(image, scribble)

        if mask is not None and self.transform is not None:
            h, w = image.shape[1], image.shape[2] if isinstance(image, torch.Tensor) else image.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.int64)
            mask = np.where(mask > 0, 1, 0)  

        if mask is not None:
            return image, scribble, mask, class_label, img_path
        return image, scribble, class_label, img_path

    def check_shapes(self, image, scribble, mask=None):
        if image.shape[:2] != scribble.shape:
            raise RuntimeError(f"Shape mismatch: image {image.shape}, scribble {scribble.shape}")
        if mask is not None and image.shape[:2] != mask.shape:
            raise RuntimeError(f"Shape mismatch: image {image.shape}, mask {mask.shape}")
        
@dataclass
class Sample:
    image: np.ndarray
    scribble: np.ndarray  
    mask: Optional[np.ndarray] = None

class ScribbleDataset(Dataset):
    def __init__(self, img_dir, scrib_dir, mask_dir=None, augment=False, size=None, ext=".png"):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, f"*{ext}")))
        self.scrib_paths = [os.path.join(scrib_dir, os.path.basename(p)) for p in self.img_paths]
        self.mask_dir = mask_dir
        self.mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in self.img_paths] if mask_dir else None
        self.augment = augment
        self.size = size
        tfms = []
        if size:
            tfms += [A.LongestMaxSize(max_size=size), A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT)]
        if augment:
            tfms += [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),

            A.ShiftScaleRotate(
                shift_limit=0.02, scale_limit=0.05, rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.3
        ),

            ]
        tfms += [A.Normalize(), ToTensorV2()]
        self.tfms = A.Compose(tfms)
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = imread(self.img_paths[idx])
        scrib = imread_gray(self.scrib_paths[idx])
        mask = imread_gray(self.mask_paths[idx]) if self.mask_dir else None

        scrib = np.where((scrib==2)|(scrib==255), 255, scrib).astype(np.uint8)
        if mask is not None:
            mask = (mask>0).astype(np.uint8)
        data = self.tfms(image=img, mask=mask if mask is not None else None)
        img_t = data["image"]

        scrib_rs = cv2.resize(scrib, (img_t.shape[2], img_t.shape[1]), interpolation=cv2.INTER_NEAREST)
        item = {"image": img_t, "scribble": torch.from_numpy(scrib_rs).long()}
        if mask is not None:
            mask_rs = cv2.resize(mask, (img_t.shape[2], img_t.shape[1]), interpolation=cv2.INTER_NEAREST)
            item["mask"] = torch.from_numpy(mask_rs).float().unsqueeze(0)
        return item
    

class HybridTrainDataset(Dataset):
    def __init__(self, img_dir, scrib_dir, gt_dir, tfms, exts=(".png",".jpg",".jpeg")):
        self.img_paths = []
        for e in exts:
            self.img_paths += sorted(glob.glob(os.path.join(img_dir, f"*{e}")))
        if len(self.img_paths)==0:
            raise RuntimeError(f"No images in {img_dir} with {exts}")
        self.scrib_dir = scrib_dir
        self.gt_dir = gt_dir
        self.exts = exts
        self.tfms = tfms

    def _match(self, p, folder):
        base = os.path.splitext(os.path.basename(p))[0]
        for e in self.exts:
            q = os.path.join(folder, base + e)
            if os.path.exists(q): return q
        return None

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        p_img = self.img_paths[i]
        p_scrib = self._match(p_img, self.scrib_dir)
        p_gt    = self._match(p_img, self.gt_dir)
        if p_scrib is None or p_gt is None:
            raise FileNotFoundError(f"Missing scribble/gt for {p_img}")

        img  = imread(p_img)                
        scrib = imread_gray(p_scrib)        
        gt    = imread_gray(p_gt)         

        scrib = np.where((scrib==255)|(scrib==2), 255, scrib).astype(np.uint8)
        gt    = (gt>0).astype(np.uint8)

        out = self.tfms(image=img, scribble=scrib, gt=gt)
        img_t  = out["image"]             
        scrib_t= out["scribble"]           
        gt_t   = out["gt"].unsqueeze(0).float() 

        return {
            "image": img_t,
            "scribble": scrib_t.cpu().numpy().astype(np.uint8),
            "mask": gt_t
        }
    
class ValProxy(Dataset):
    def __init__(self, base, indices, val_tfms):
        self.base = base; self.indices = indices; self.val_tfms = val_tfms
    def __len__(self): return len(self.indices)
    def __getitem__(self, j):
        i = self.indices[j]
        p_img = self.base.img_paths[i]
        p_scrib = self.base._match(p_img, self.base.scrib_dir)
        p_gt    = self.base._match(p_img, self.base.gt_dir)
        img  = imread(p_img)
        scrib = imread_gray(p_scrib)
        gt    = imread_gray(p_gt)
        scrib = np.where((scrib==255)|(scrib==2), 255, scrib).astype(np.uint8)
        gt    = (gt>0).astype(np.uint8)
        out = self.val_tfms(image=img, scribble=scrib, gt=gt)
        return {
            "image": out["image"],
            "scribble": out["scribble"].cpu().numpy().astype(np.uint8),
            "mask": out["gt"].unsqueeze(0).float()
        }