# save as: robustness_eval.py
import os, glob, csv, cv2, argparse
import numpy as np
from tqdm import tqdm

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..Unet_RW.U_net import UNet
from ..Unet_RW.utils import imread, save_mask, crf_refine  # (crf not used here, but kept for parity)

# --------------------- helpers ---------------------
def to_bin(mask_f32, thr=0.5):
    return (mask_f32 >= thr).astype(np.uint8)

def iou_binary(a, b):
    inter = np.logical_and(a==1, b==1).sum()
    union = np.logical_or(a==1, b==1).sum()
    return float(inter) / (float(union) + 1e-6)

def fg_bg_miou(m0_bin, m1_bin):
    fg_iou = iou_binary(m0_bin, m1_bin)
    bg0 = 1 - m0_bin; bg1 = 1 - m1_bin
    bg_iou = iou_binary(bg0, bg1)
    miou = 0.5 * (fg_iou + bg_iou)
    return fg_iou, bg_iou, miou

def load_clean_mask(clean_dir, image_path, thr=0.5):
    base = os.path.basename(image_path)
    p_mask = os.path.join(clean_dir, base)
    m = cv2.imread(p_mask, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Clean mask not found for {base} at {p_mask}")
    m = m.astype(np.float32) / 255.0
    return to_bin(m, thr=thr)

def build_infer_tfms(size):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(),
        ToTensorV2(),
    ])

@torch.no_grad()
def predict_unet_prob(model, device, img_u8, size=384):
    h, w = img_u8.shape[:2]
    tfms = build_infer_tfms(size)
    t = tfms(image=img_u8)
    x = t["image"].unsqueeze(0).to(device)
    pr = torch.sigmoid(model(x))[0,0].detach().cpu().numpy()
    pr = cv2.resize(pr, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return pr

# --------------------- perturbations ---------------------
def make_gaussian_noise_aug(std):
    def add_noise(img):
        img = img.astype(np.float32)
        noise = np.random.normal(loc=0.0, scale=std, size=img.shape).astype(np.float32)
        noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
        return noisy
    return add_noise

def make_rotation_aug(deg):
    aff = A.Affine(rotate=(-deg, deg), translate_percent=0.0, scale=1.0,
                   cval=0, p=1.0, fit_output=False)
    def rot(img):
        return aff(image=img)["image"]
    return rot

# --------------------- main evaluation ---------------------
def evaluate_robustness(args):
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    ckpt = torch.load(os.path.join(args.out, 'best.pt'), map_location=device)
    model = UNet(ch=ckpt['args'].get('width', 32))
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()

    # choose perturbation
    if args.perturb == "gaussian":
        perturb = make_gaussian_noise_aug(args.std)
        perturb_desc = f"gaussian_std{args.std}"
    elif args.perturb == "rotation":
        perturb = make_rotation_aug(args.deg)
        perturb_desc = f"rotation_deg{args.deg}"
    else:
        raise ValueError("perturb must be 'gaussian' or 'rotation'")

    # gather test images
    img_paths = sorted(glob.glob(os.path.join(args.data, 'test/images', f"*{args.ext}")))
    if not img_paths:
        raise RuntimeError(f"No test images found in {os.path.join(args.data, 'test/images')} with *{args.ext}")

    os.makedirs(args.save_csv_dir, exist_ok=True)

    rows = []
    fg_list, bg_list, miou_list = [], [], []
    for p in tqdm(img_paths, desc=f"Robustness: {perturb_desc}"):
        img = imread(p)  # RGB uint8
        m_clean = load_clean_mask(args.clean_dir, p, thr=args.thr)

        img_p = perturb(img.copy())
        pr_pert = predict_unet_prob(model, device, img_p, size=args.size)
        m_pert = to_bin(pr_pert, thr=args.thr)

        fg_iou, bg_iou, miou = fg_bg_miou(m_clean, m_pert)
        fg_drop = 1.0 - fg_iou
        bg_drop = 1.0 - bg_iou
        miou_drop = 1.0 - miou

        rows.append([
            os.path.basename(p),
            args.perturb,
            perturb_desc,
            f"{fg_iou:.6f}",
            f"{bg_iou:.6f}",
            f"{miou:.6f}",
            f"{fg_drop:.6f}",
            f"{bg_drop:.6f}",
            f"{miou_drop:.6f}",
        ])
        fg_list.append(fg_iou); bg_list.append(bg_iou); miou_list.append(miou)

    # pick two images with largest absolute change (use mIoU drop)
    drops = np.array([float(r[8]) for r in rows])  # miou_change(1-iou)
    top2_idx = np.argsort(-drops)[:2]
    top2_names = [rows[i][0] for i in top2_idx]

    # compute means
    mean_fg = float(np.mean(fg_list)) if fg_list else 0.0
    mean_bg = float(np.mean(bg_list)) if bg_list else 0.0
    mean_miou = float(np.mean(miou_list)) if miou_list else 0.0

    # write CSV
    csv_path = os.path.join(args.save_csv_dir, f"robustness_{args.perturb}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "perturb", "params", "fg_iou", "bg_iou", "miou",
                    "fg_change(1-iou)", "bg_change(1-iou)", "miou_change(1-iou)"])
        w.writerows(rows)
        # summary footer
        w.writerow([])
        w.writerow(["mean_fg_iou", f"{mean_fg:.6f}"])
        w.writerow(["mean_bg_iou", f"{mean_bg:.6f}"])
        w.writerow(["mean_miou",  f"{mean_miou:.6f}"])
        w.writerow([])
        w.writerow(["top2_largest_change_images"] + top2_names)

    print(f"[Saved] {csv_path}")
    print(f"Top-2 highest absolute change images: {top2_names}")
    print(f"Means — FG IoU: {mean_fg:.4f} | BG IoU: {mean_bg:.4f} | mIoU: {mean_miou:.4f}")

# --------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset")
    ap.add_argument("--out", type=str, default="outputs")           # where best.pt lives
    ap.add_argument("--clean_dir", type=str, required=True,         # directory of clean predicted masks
                    help="Directory containing clean predictions saved earlier (same filenames as test images).")
    ap.add_argument("--save_csv_dir", type=str, default="robustness_csv")
    ap.add_argument("--ext", type=str, default=".png")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--thr", type=float, default=0.5)

    ap.add_argument("--perturb", type=str, choices=["gaussian","rotation"], required=True)
    ap.add_argument("--std", type=float, default=10.0, help="Gaussian noise std (0..255), used when --perturb gaussian")
    ap.add_argument("--deg", type=float, default=5.0,  help="Rotation degrees (+/-), used when --perturb rotation")

    args = ap.parse_args()
    evaluate_robustness(args)

if __name__ == "__main__":
    main()
