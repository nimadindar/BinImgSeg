import os
import cv2 
import torch
import random
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


HAS_CRF = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
    HAS_CRF = True
except Exception as e:
    HAS_CRF = False


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

def jpg_to_png(input_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(input_path):
        if file.endswith(".jpg"):
            img = Image.open(os.path.join(input_path, file))

            new_name = os.path.splitext(file)[0] + ".png"
            img.save(os.path.join(output_path, new_name))

def make_confidence_mask(scrib, pfg, hi=0.9, lo=0.1):

    pfg = np.squeeze(pfg)
    if pfg.shape != scrib.shape:
        pfg = cv2.resize(pfg, (scrib.shape[1], scrib.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask = np.zeros_like(pfg, dtype=np.float32)
    target = np.zeros_like(pfg, dtype=np.float32)

    scrib_fg = (scrib == 1)
    scrib_bg = (scrib == 0)
    mask[scrib_fg | scrib_bg] = 1.0
    target[scrib_fg] = 1.0
    target[scrib_bg] = 0.0

    hi_conf = (pfg >= hi)
    lo_conf = (pfg <= lo)
    mask[hi_conf | lo_conf] = 1.0
    target[hi_conf] = 1.0
    target[lo_conf] = 0.0

    return target, mask

class ConfusionTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
    def update_numpy(self, pred_bin_np, gt_np):

        self.tp += int(((pred_bin_np == 1) & (gt_np == 1)).sum())
        self.fp += int(((pred_bin_np == 1) & (gt_np == 0)).sum())
        self.fn += int(((pred_bin_np == 0) & (gt_np == 1)).sum())
        self.tn += int(((pred_bin_np == 0) & (gt_np == 0)).sum())
    def update_torch(self, pred_bin_t, gt_t):

        pred = pred_bin_t.int()
        gt   = (gt_t > 0.5).int()
        self.tp += int(((pred == 1) & (gt == 1)).sum().item())
        self.fp += int(((pred == 1) & (gt == 0)).sum().item())
        self.fn += int(((pred == 0) & (gt == 1)).sum().item())
        self.tn += int(((pred == 0) & (gt == 0)).sum().item())
    def summary(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-9
        acc   = (tp + tn) / max(tp+tn+fp+fn, 1)
        prec  = tp / max(tp + fp, 1)
        rec   = tp / max(tp + fn, 1)    
        f1    = 2*prec*rec / max(prec + rec, eps)
        dice  = 2*tp / max(2*tp + fp + fn, 1)   
        iou_fg = tp / max(tp + fp + fn, 1)
        iou_bg = tn / max(tn + fp + fn, 1)
        miou   = 0.5*(iou_fg + iou_bg)
        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "dice_fg": dice, "iou_fg": iou_fg,
            "iou_bg": iou_bg, "miou": miou
        }
    def pretty_matrix(self):

        return [[self.tn, self.fp],
                [self.fn, self.tp]]

def evaluate_split_confusion(dataloader, model, device, threshold=0.5):
    """
    Runs the model over a dataloader (must yield 'image' and 'mask') and
    returns confusion counts + metrics.
    """
    tracker = ConfusionTracker()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            img = batch['image'].to(device)
            gt  = batch['mask'].to(device)              # (B,1,H,W) float {0,1}
            pr  = torch.sigmoid(model(img))             # (B,1,H,W)
            pred_bin = (pr > threshold).float()
            tracker.update_torch(pred_bin, gt)
    return tracker.summary(), tracker.pretty_matrix()

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

def build_train_tfms(size: int):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.2),

        A.ShiftScaleRotate(
            shift_limit=0.02, scale_limit=0.05, rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.3
        ),

        A.Normalize(),
        ToTensorV2(),
    ],
    additional_targets={
        "scribble": "mask",
        "gt": "mask",
    })

def build_val_tfms(size: int): 
    return A.Compose([ 
        A.LongestMaxSize(max_size=size), 
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0), 
        A.Normalize(), ToTensorV2()],
        additional_targets={ "scribble": "mask", "gt": "mask", })

def clean_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            fpath = os.path.join(input_dir, fname)
            outpath = os.path.join(output_dir, fname)
            with Image.open(fpath) as im:
                # Convert to RGB to avoid profile persistence
                im = im.convert("RGB")
                # Save WITHOUT icc_profile metadata
                im.save(outpath, "PNG", icc_profile=None)

