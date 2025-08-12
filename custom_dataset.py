import os
from PIL import Image

from torch.utils.data import Dataset

class ScribbleSegTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Args:
            root_dir (str): Path to train dataset folder (contains 'images', 'scribbles', 'ground_truth')
            transform: Transform for images and scribbles (same spatial transforms)
            target_transform: Transform for ground truth masks
        """
        self.images_dir = os.path.join(root_dir, "images")
        self.scribbles_dir = os.path.join(root_dir, "scribbles")
        self.gt_dir = os.path.join(root_dir, "ground_truth")

        self.image_files = sorted(os.listdir(self.images_dir))
        self.scribble_files = sorted(os.listdir(self.scribbles_dir))
        self.gt_files = sorted(os.listdir(self.gt_dir))

        assert len(self.image_files) == len(self.scribble_files) == len(self.gt_files), \
            "Mismatch between images, scribbles, and ground truth counts."

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        scribble_path = os.path.join(self.scribbles_dir, self.scribble_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])

        image = Image.open(img_path).convert("RGB")
        scribble = Image.open(scribble_path).convert("L")
        gt = Image.open(gt_path).convert("L")

        if self.transform:
            image = self.transform(image)
            scribble = self.transform(scribble)  

        if self.target_transform:
            gt = self.target_transform(gt)
        # else:
        #     gt = transforms.ToTensor()(gt)  # default: convert to tensor in [0,1]

        return image, scribble, gt


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

