from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset 

import os, glob, math, argparse, random, cv2, numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from skimage.segmentation import random_walker
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

HAS_CRF = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
    HAS_CRF = True
except Exception as e:
    HAS_CRF = False

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.mps.manual_seed(seed)

def imread(fp):
    img = cv2.imread(fp, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(fp)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imread_gray(fp):
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(fp)
    return img

def save_mask(fp, mask):
    mask = (mask>0.5).astype(np.uint8)*255
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    cv2.imwrite(fp, mask)

def normalize01(img):
    return (img/255.0).astype(np.float32)

# ----------------------------
# Data
# ----------------------------
@dataclass
class Sample:
    image: np.ndarray
    scribble: np.ndarray  # 0=bg,1=fg,255 or 2 = unlabeled
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
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.2),
                A.ColorJitter(p=0.2, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                A.GaussNoise(p=0.15),
            ]
        tfms += [A.Normalize(), ToTensorV2()]
        self.tfms = A.Compose(tfms)
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = imread(self.img_paths[idx])
        scrib = imread_gray(self.scrib_paths[idx])
        mask = imread_gray(self.mask_paths[idx]) if self.mask_dir else None
        # harmonize labels: unlabeled=255 or 2 -> 255
        scrib = np.where((scrib==2)|(scrib==255), 255, scrib).astype(np.uint8)
        if mask is not None:
            mask = (mask>0).astype(np.uint8)
        data = self.tfms(image=img, mask=mask if mask is not None else None)
        img_t = data["image"]
        # resize scribble to match after tfms
        scrib_rs = cv2.resize(scrib, (img_t.shape[2], img_t.shape[1]), interpolation=cv2.INTER_NEAREST)
        item = {"image": img_t, "scribble": torch.from_numpy(scrib_rs).long()}
        if mask is not None:
            mask_rs = cv2.resize(mask, (img_t.shape[2], img_t.shape[1]), interpolation=cv2.INTER_NEAREST)
            item["mask"] = torch.from_numpy(mask_rs).float().unsqueeze(0)
        return item

# ----------------------------
# Random Walker pseudo-labels
# ----------------------------
# --- Random Walker pseudo-labels ---
def seeds_from_scribble(scrib):
    # scrib: HxW with {0,1,255} -> {-1,0,1}; -1 means unlabeled for random_walker
    seeds = np.full_like(scrib, fill_value=-1, dtype=np.int32)
    seeds[scrib == 0] = 0
    seeds[scrib == 1] = 1
    return seeds

def rw_proba(img, scrib, beta=130, gamma=0.0, mode="cg_mg"):
    # img: HxWx3 uint8; scrib: HxW {0,1,255}
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab = cv2.GaussianBlur(lab, (0, 0), 0.8)

    seeds = seeds_from_scribble(scrib)  # -1 unlabeled, 0 bg, 1 fg

    prob = random_walker(
        lab,
        seeds,
        beta=beta,
        mode=mode,
        return_full_prob=True,
        channel_axis=-1,  # HxWx3 image
    )

    # Normalize to (H, W, C)
    if prob.ndim != 3:
        raise ValueError(f"Unexpected prob ndim from random_walker: {prob.ndim}")
    if prob.shape[0] in (1, 2, 3) and prob.shape[-1] not in (1, 2, 3):
        # (C, H, W) -> (H, W, C)
        prob = np.moveaxis(prob, 0, -1)

    C = prob.shape[-1]
    has_fg = (scrib == 1).any()
    has_bg = (scrib == 0).any()

    if C == 2:
        pfg = prob[..., 1]
    elif C == 1:
        # Only one class present in seeds; map the single channel accordingly
        single = prob[..., 0]
        if has_fg and not has_bg:
            pfg = single
        elif has_bg and not has_fg:
            pfg = 1.0 - single
        else:
            # pathological; fall back to 0.5
            pfg = np.full(scrib.shape, 0.5, dtype=np.float32)
    else:
        raise ValueError(f"Unexpected prob shape from random_walker: {prob.shape}")

    # Clean up
    pfg = np.squeeze(pfg)
    pfg = np.nan_to_num(pfg, nan=0.5)

    # Ensure pfg matches scrib shape
    if pfg.shape != scrib.shape:
        pfg = cv2.resize(pfg, (scrib.shape[1], scrib.shape[0]), interpolation=cv2.INTER_LINEAR)

    return pfg




# ----------------------------
# Model
# ----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1):
        super().__init__()
        p = k//2
        self.seq = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)

class UNetBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = ConvBNReLU(c_in, c_out)
        self.conv2 = ConvBNReLU(c_out, c_out)
    def forward(self, x): return self.conv2(self.conv1(x))

class UNet(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.enc1 = UNetBlock(3, ch)
        self.enc2 = UNetBlock(ch, ch*2)
        self.enc3 = UNetBlock(ch*2, ch*4)
        self.enc4 = UNetBlock(ch*4, ch*8)
        self.pool = nn.MaxPool2d(2,2)
        self.up4 = nn.ConvTranspose2d(ch*8, ch*4, 2, 2)
        self.dec4 = UNetBlock(ch*8, ch*4)
        self.up3 = nn.ConvTranspose2d(ch*4, ch*2, 2, 2)
        self.dec3 = UNetBlock(ch*4, ch*2)
        self.up2 = nn.ConvTranspose2d(ch*2, ch, 2, 2)
        self.dec2 = UNetBlock(ch*2, ch)
        self.out = nn.Conv2d(ch, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2); p3 = self.pool(e3)
        e4 = self.enc4(p3)
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        return self.out(d2)

# ----------------------------
# Loss (masked BCE+Dice)
# ----------------------------
class MaskedBCEDice(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__(); self.eps = eps
    def forward(self, logits, target, mask):
        # mask=1 where supervised, 0 ignore
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy(probs, target, reduction='none')
        bce = (bce*mask).sum() / (mask.sum()+1e-6)
        # Dice
        inter = (probs*target*mask).sum()
        denom = (probs*mask).sum() + (target*mask).sum() + self.eps
        dice = 1 - (2*inter + self.eps) / denom
        return bce + dice

# ----------------------------
# Training
# ----------------------------
def make_confidence_mask(scrib, pfg, hi=0.9, lo=0.1):
    # Ensure shapes match and pfg is 2D
    pfg = np.squeeze(pfg)
    if pfg.shape != scrib.shape:
        pfg = cv2.resize(pfg, (scrib.shape[1], scrib.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask = np.zeros_like(pfg, dtype=np.float32)
    target = np.zeros_like(pfg, dtype=np.float32)

    # supervise on scribbles
    scrib_fg = (scrib == 1)
    scrib_bg = (scrib == 0)
    mask[scrib_fg | scrib_bg] = 1.0
    target[scrib_fg] = 1.0
    target[scrib_bg] = 0.0

    # confident pseudo-labels
    hi_conf = (pfg >= hi)
    lo_conf = (pfg <= lo)
    mask[hi_conf | lo_conf] = 1.0
    target[hi_conf] = 1.0
    target[lo_conf] = 0.0

    return target, mask


def train(args):
    set_seed(args.seed)
    device = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = ScribbleDataset(os.path.join(args.data, 'train/images'),
                         os.path.join(args.data, 'train/scribbles'),
                         os.path.join(args.data, 'train/ground_truth'),
                         augment=True, size=args.size, ext=args.ext)

    n_val = max(1, int(0.15*len(ds)))
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    val_idx = set(idxs[:n_val])
    train_idx = [i for i in idxs if i not in val_idx]

    # >>> use top-level Subset so it's picklable
    ds_tr = Subset(ds, train_idx)
    ds_va = Subset(ds, list(val_idx))

    # sensible worker/pin_memory defaults for MPS
    num_workers = 0 if device == 'mps' else 2
    pin_mem = True if device == 'cuda' else False

    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_mem)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False,
                       num_workers=0, pin_memory=pin_mem)

    model = UNet(ch=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = MaskedBCEDice()

    best = 0.0; patience=0
    for epoch in range(1, args.epochs+1):
        model.train()
        for batch in dl_tr:
            img = batch['image'].to(device)
            scrib = batch['scribble'].numpy()
            # build pseudo-labels per-sample using RW on CPU
            targets=[]; masks=[]
            for i in range(img.shape[0]):
                im_np = (img[i].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
                pfg = rw_proba(im_np, scrib[i], beta=args.rw_beta)
                tgt, m = make_confidence_mask(scrib[i], pfg, hi=args.pl_hi, lo=args.pl_lo)
                targets.append(torch.from_numpy(tgt).unsqueeze(0))
                masks.append(torch.from_numpy(m).unsqueeze(0))
            target = torch.stack(targets).to(device).float()
            conf = torch.stack(masks).to(device).float()
            logits = model(img)
            loss = lossf(logits, target, conf)
            opt.zero_grad(); loss.backward(); opt.step()
        # validation IoU
        model.eval()
        inter0=0; union0=0; inter1=0; union1=0
        with torch.no_grad():
            for batch in dl_va:
                img = batch['image'].to(device)
                mask = batch['mask'].to(device)
                pr = torch.sigmoid(model(img))
                pr_bin = (pr>0.5).float()
                # mIoU
                inter1 += ((pr_bin*mask)==1).sum().item()
                union1 += ((pr_bin+mask)>=1).sum().item()
                pr0 = 1-pr_bin; gt0 = 1-mask
                inter0 += ((pr0*gt0)==1).sum().item()
                union0 += ((pr0+gt0)>=1).sum().item()
        miou = 0.5*((inter0/(union0+1e-6)) + (inter1/(union1+1e-6)))
        if miou > best:
            best = miou; patience=0
            os.makedirs(args.out, exist_ok=True)
            torch.save({'model':model.state_dict(), 'args':vars(args)}, os.path.join(args.out, 'best.pt'))
        else:
            patience += 1
        print(f"Epoch {epoch}: val mIoU={miou:.4f} best={best:.4f} patience={patience}")
        if patience>=args.early_stop: break

def crf_refine(img, prob):
    if not HAS_CRF: return prob
    H, W = prob.shape
    U = np.zeros((2, H, W), dtype=np.float32)
    U[1] = prob; U[0] = 1.0 - prob
    U = unary_from_softmax(U)
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(U)
    feat_gauss = create_pairwise_gaussian(sdims=(3,3), shape=(H,W))
    d.addPairwiseEnergy(feat_gauss, compat=3)
    feat_bi = create_pairwise_bilateral(sdims=(50,50), schan=(5,5,5), img=img, chdim=2)
    d.addPairwiseEnergy(feat_bi, compat=5)
    Q = d.inference(5)
    return np.array(Q)[1].reshape(H, W)

def infer(args):
    device = 'mps' if torch.mps.is_available() else 'cpu'
    ckpt = torch.load(os.path.join(args.out, 'best.pt'), map_location=device)
    model = UNet(ch=ckpt['args'].get('width',32)); model.load_state_dict(ckpt['model']); model = model.to(device).eval()
    tfms = A.Compose([A.LongestMaxSize(max_size=args.size), A.PadIfNeeded(args.size, args.size, border_mode=cv2.BORDER_CONSTANT),
                      A.Normalize(), ToTensorV2()])
    img_paths = sorted(glob.glob(os.path.join(args.data, 'test/images', f"*{args.ext}")))
    os.makedirs(args.save, exist_ok=True)
    for p in img_paths:
        img = imread(p)
        t = tfms(image=img); x = t['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            pr = torch.sigmoid(model(x))[0,0].cpu().numpy()
        pr = cv2.resize(pr, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if args.crf:
            pr = crf_refine(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), pr)
        save_mask(os.path.join(args.save, os.path.basename(p)), pr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--out', type=str, default='outputs')
    parser.add_argument('--save', type=str, default='preds')
    parser.add_argument('--mode', type=str, choices=['train','infer'], default='train')
    parser.add_argument('--size', type=int, default=384)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--rw_beta', type=float, default=130.0)
    parser.add_argument('--pl_hi', type=float, default=0.9)
    parser.add_argument('--pl_lo', type=float, default=0.1)
    parser.add_argument('--ext', type=str, default='.png')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--crf', action='store_true')
    args = parser.parse_args()
    if args.mode=='train':
        train(args)
    else:
        infer(args)

if __name__ == '__main__':
    main()
