import os, json, csv, math, time, random, argparse
from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .utils import imread_gray, imread, make_confidence_mask, build_train_tfms, build_val_tfms
from .random_walk import rw_proba
from .custom_dataset import HybridTrainDataset
from .U_net import UNet
from .loss import MaskedBCEDice


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def select_device():
    if hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class ConfusionTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.tp = 0; self.fp = 0; self.fn = 0; self.tn = 0
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

@torch.no_grad()
def evaluate_loader(dataloader, model, device, thr=0.5):
    model.eval()
    tracker = ConfusionTracker()
    for batch in dataloader:
        img = batch["image"].to(device)           
        gt  = batch["mask"].to(device).float()     
        pr  = torch.sigmoid(model(img))            
        pred = (pr > thr).float()
        tracker.update_torch(pred, gt)
    return tracker.summary()


def train_one_epoch_hybrid(model, dl_tr, opt, lossf, device,
                           lambda_rw: float, rw_beta: float,
                           pl_hi: float, pl_lo: float):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for batch in dl_tr:
        img  = batch["image"].to(device)           
        gt   = batch["mask"].to(device).float()     
        scrib_np = batch["scribble"]               

        logits = model(img)

        conf_gt = torch.ones_like(gt)
        loss_gt = lossf(logits, gt, conf_gt)

        targets_rw = []; masks_rw = []
        B = img.shape[0]
        for i in range(B):
            scrib_i = scrib_np[i]
            has_fg = (scrib_i == 1).any()
            has_bg = (scrib_i == 0).any()
            if not (has_fg and has_bg):

                tgt_i = np.zeros_like(scrib_i, dtype=np.float32)
                m_i   = np.zeros_like(scrib_i, dtype=np.float32)
            else:
                im_np = img[i].detach().cpu().permute(1,2,0).numpy()
                im_np = (np.clip(im_np, 0, 1) * 255).astype(np.uint8)

                pfg = rw_proba(im_np, scrib_i, beta=rw_beta)
                tgt_i, m_i = make_confidence_mask(scrib_i, pfg, hi=pl_hi, lo=pl_lo)

            targets_rw.append(torch.from_numpy(tgt_i).unsqueeze(0))
            masks_rw.append(torch.from_numpy(m_i).unsqueeze(0))   

        target_rw = torch.stack(targets_rw).to(device).float()
        conf_rw   = torch.stack(masks_rw).to(device).float()

        loss_rw = lossf(logits, target_rw, conf_rw)
        loss = loss_gt + lambda_rw * loss_rw

        opt.zero_grad(); loss.backward(); opt.step()

        epoch_loss += loss.item()
        n_batches += 1

    return epoch_loss / max(n_batches, 1)


def kfold_indices(n: int, k: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    rng = np.random.RandomState(seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    folds = np.array_split(idxs, k)
    pairs = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        pairs.append((train_idx.tolist(), val_idx.tolist()))
    return pairs


@dataclass
class RunConfig:
    width: int
    lr: float
    bs: int
    size: int
    epochs: int
    early_stop: int
    lambda_rw: float
    rw_beta: float
    pl_hi: float
    pl_lo: float
    threshold: float

def param_grid_to_list(grid: Dict[str, List[Any]]) -> List[RunConfig]:
    keys = list(grid.keys())
    all_vals = list(product(*[grid[k] for k in keys]))
    runs = []
    for vals in all_vals:
        d = dict(zip(keys, vals))
        runs.append(RunConfig(**d))
    return runs

def grid_search(args):
    set_seed(args.seed)
    device = select_device()
    os.makedirs(args.out, exist_ok=True)

    tfms_train = build_train_tfms(args.size)
    tfms_val   = build_val_tfms(args.size)

    ds_full = HybridTrainDataset(
        os.path.join(args.data, 'train/images'),
        os.path.join(args.data, 'train/scribbles'),
        os.path.join(args.data, 'train/ground_truth'),
        tfms_train,
        exts=(args.ext,) if hasattr(args, "ext") else (".png",".jpg",".jpeg")
    )


    grid = {
        "width":      [8, 12,16,32],         # UNet base channels
        "lr":         [1e-5, 1e-4, 1e-3],
        "bs":         [4],              
        "size":       [args.size],      # keep fixed to align TFMS; or try [320, 384]
        "epochs":     [args.epochs],    # e.g., 25 for search, 60 later
        "early_stop": [5],              # patience
        "lambda_rw":  [0.1, 0.2, 0.3],  # weight of RW aux loss
        "rw_beta":    [60, 80, 100],     # RW smoothness
        "pl_hi":      [0.9],            # confident FG
        "pl_lo":      [0.1],            # confident BG
        "threshold":  [0.4,0.5,0.6],            # binarization for metrics
    }
    runs = param_grid_to_list(grid)

    csv_path = os.path.join(args.out, "grid_search_results.csv")
    header = [
        "combo_id","fold","epoch_best","train_loss_best",
        "miou","dice_fg","iou_fg","iou_bg","precision","recall","f1","accuracy",
        "tp","fp","fn","tn","params_json","checkpoint_path"
    ]
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    n = len(ds_full)
    folds = kfold_indices(n, args.kfolds, args.seed)

    best_overall = None 
    combo_id = 0

    for cfg in runs:
        combo_id += 1
        params_dict = asdict(cfg)
        print(f"\n=== Combo {combo_id}/{len(runs)} ===")
        print(json.dumps(params_dict, indent=2))
        fold_scores = []

        for fold_no, (tr_idx, va_idx) in enumerate(folds, start=1):

            ds_tr = Subset(ds_full, tr_idx)


            class _ValProxy(torch.utils.data.Dataset):
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
                    return {"image": out["image"],
                            "scribble": out["scribble"].cpu().numpy().astype(np.uint8),
                            "mask": out["gt"].unsqueeze(0).float()}
            ds_va = _ValProxy(ds_full, va_idx, tfms_val)

            num_workers = 0 if device=="mps" else 2
            pin_mem = True if device=="cuda" else False
            dl_tr = DataLoader(ds_tr, batch_size=cfg.bs, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
            dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=0, pin_memory=pin_mem)

            model = UNet(ch=cfg.width).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
            lossf = MaskedBCEDice()

            best_miou = -1.0
            best_epoch = 0
            best_train_loss = float("inf")
            patience = 0

            ckpt_dir = os.path.join(args.out, f"gs_combo{combo_id}_fold{fold_no}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "best.pt")

            for epoch in range(1, cfg.epochs+1):

                tr_loss = train_one_epoch_hybrid(
                    model, dl_tr, opt, lossf, device,
                    lambda_rw=cfg.lambda_rw, rw_beta=cfg.rw_beta,
                    pl_hi=cfg.pl_hi, pl_lo=cfg.pl_lo
                )

                metrics = evaluate_loader(dl_va, model, device, thr=cfg.threshold)
                miou = metrics["miou"]

                improved = miou > best_miou + 1e-6
                if improved:
                    best_miou = miou
                    best_epoch = epoch
                    best_train_loss = tr_loss
                    patience = 0
                    torch.save({"model": model.state_dict(),
                                "params": params_dict,
                                "metrics": metrics},
                               ckpt_path)
                else:
                    patience += 1

                print(f"[Combo {combo_id} | Fold {fold_no} | Epoch {epoch}] "
                      f"loss={tr_loss:.4f} miou={miou:.4f} best_miou={best_miou:.4f} patience={patience}")

                if patience >= cfg.early_stop:
                    break

            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    combo_id, fold_no, best_epoch, f"{best_train_loss:.6f}",
                    f"{best_miou:.6f}",
                    f"{metrics['dice_fg']:.6f}", f"{metrics['iou_fg']:.6f}",
                    f"{metrics['iou_bg']:.6f}", f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}", f"{metrics['f1']:.6f}",
                    f"{metrics['accuracy']:.6f}",
                    metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"],
                    json.dumps(params_dict), ckpt_path
                ])
            fold_scores.append(best_miou)

        avg_miou = float(np.mean(fold_scores)) if fold_scores else -1.0
        if (best_overall is None) or (avg_miou > best_overall[0] + 1e-6):
            best_overall = (avg_miou, combo_id, params_dict)

    if best_overall is not None:
        best_json = os.path.join(args.out, "best_params.json")
        with open(best_json, "w") as f:
            json.dump({
                "avg_cv_miou": best_overall[0],
                "combo_id": best_overall[1],
                "params": best_overall[2]
            }, f, indent=2)
        print("\n=== BEST PARAMS ===")
        print(json.dumps(best_overall[2], indent=2))
        print(f"Average CV mIoU: {best_overall[0]:.4f}")
        print(f"Saved to: {best_json}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="dataset")
    p.add_argument("--out", type=str, default="outputs_grid")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ext", type=str, default=".png")
    args = p.parse_args()
    grid_search(args)

if __name__ == "__main__":
    main()


##### Run the script #####
# =========================
# python Unet_RW/grid_search.py \
#   --data dataset \
#   --out outputs_grid \
#   --size 384 \
#   --epochs 25 \
#   --kfolds 5