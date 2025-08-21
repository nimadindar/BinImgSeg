from torch.utils.data import DataLoader, Subset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch

import os, glob, argparse, random, cv2, numpy as np

from .utils import set_seed, make_confidence_mask, evaluate_split_confusion, \
                    crf_refine, save_mask, build_train_tfms, build_val_tfms, imread
from .custom_dataset import ScribbleDataset, HybridTrainDataset, ValProxy
from .random_walk import rw_proba
from .loss import MaskedBCEDice
from .U_net import UNet

# import warnings
# warnings.filterwarnings("ignore")

HAS_CRF = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
    HAS_CRF = True
except Exception as e:
    HAS_CRF = False


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

    ds_tr = Subset(ds, train_idx)
    ds_va = Subset(ds, list(val_idx))

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
        model.eval()
        inter0=0; union0=0; inter1=0; union1=0
        with torch.no_grad():
            for batch in dl_va:
                img = batch['image'].to(device)
                mask = batch['mask'].to(device)
                pr = torch.sigmoid(model(img))
                pr_bin = (pr>0.5).float()
                inter1 += ((pr_bin*mask)==1).sum().item()
                union1 += ((pr_bin+mask)>=1).sum().item()
                pr0 = 1-pr_bin; gt0 = 1-mask
                inter0 += ((pr0*gt0)==1).sum().item()
                union0 += ((pr0+gt0)>=1).sum().item()
        miou = 0.5*((inter0/(union0+1e-6)) + (inter1/(union1+1e-6)))
        val_metrics, val_matrix = evaluate_split_confusion(dl_va, model, device, threshold=0.5)
        print("Val confusion matrix [[tn, fp], [fn, tp]]:", val_matrix)
        print("Val metrics:",
            {k: (round(v,4) if isinstance(v, float) else v) for k,v in val_metrics.items()})

        if miou > best:
            best = miou; patience=0
            os.makedirs(args.out, exist_ok=True)
            torch.save({'model':model.state_dict(), 'args':vars(args)}, os.path.join(args.out, 'best.pt'))
        else:
            patience += 1
        print(f"Epoch {epoch}: val mIoU={miou:.4f} best={best:.4f} patience={patience}")
        if patience>=args.early_stop: break


def train_hybrid(args,
                 lambda_rw: float = 0.2,
                 hi: float = 0.9,
                 lo: float = 0.1):
    """
    Hybrid training:
      - Primary loss: full-supervised BCE+Dice on ground-truth masks.
      - Auxiliary loss: masked BCE+Dice on confident pseudo-labels expanded from scribbles via Random Walker.
      - Total loss = loss_gt + lambda_rw * loss_rw

    Args:
      lambda_rw : weight for RW auxiliary loss (0.1–0.5 is typical).
      hi, lo    : thresholds for confident pseudo-labels from RW (p >= hi -> fg, p <= lo -> bg).
    """

    set_seed(args.seed)
    device = 'mps' if hasattr(torch, "mps") and torch.mps.is_available() else \
             'cuda' if torch.cuda.is_available() else 'cpu'

    tfms_train = build_train_tfms(args.size)
    tfms_val   = build_val_tfms(args.size)

    ds_full = HybridTrainDataset(
        os.path.join(args.data, 'train/images'),
        os.path.join(args.data, 'train/scribbles'),
        os.path.join(args.data, 'train/ground_truth'),
        tfms_train,
        exts=(args.ext,) if hasattr(args, "ext") else (".png",".jpg",".jpeg"),
    )

    n_val = max(1, int(0.15*len(ds_full)))
    idxs = list(range(len(ds_full))); random.shuffle(idxs)
    val_idx = set(idxs[:n_val])
    train_idx = [i for i in idxs if i not in val_idx]

    ds_tr = Subset(ds_full, train_idx)
    ds_va = ValProxy(ds_full, list(val_idx),tfms_val)

    num_workers = 0 if device == 'mps' else 2
    pin_mem = True if device == 'cuda' else False

    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_mem)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False,
                       num_workers=0, pin_memory=pin_mem)

    model = UNet(ch=args.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = MaskedBCEDice()

    best = 0.0; patience = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        for batch in dl_tr:
            img  = batch['image'].to(device)           
            gt   = batch['mask'].to(device).float()    
            scrib_np = batch['scribble']              

            logits = model(img)                         
            conf_gt = torch.ones_like(gt)           
            loss_gt = lossf(logits, gt, conf_gt)

            targets_rw = []; masks_rw = []
            for i in range(img.shape[0]):
                im_np = (img[i].detach().cpu().clamp(-10,10).permute(1,2,0).numpy())
                im_np = (np.clip(im_np, 0, 1) * 255).astype(np.uint8)

                pfg = rw_proba(im_np, scrib_np[i], beta=args.rw_beta)
                tgt_i, m_i = make_confidence_mask(scrib_np[i], pfg, hi=hi, lo=lo)
                targets_rw.append(torch.from_numpy(tgt_i).unsqueeze(0))
                masks_rw.append(torch.from_numpy(m_i).unsqueeze(0))
            target_rw = torch.stack(targets_rw).to(device).float()   
            conf_rw   = torch.stack(masks_rw).to(device).float()    

            loss_rw = lossf(logits, target_rw, conf_rw)

            loss = loss_gt + lambda_rw * loss_rw
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        inter0=0; union0=0; inter1=0; union1=0
        with torch.no_grad():
            for batch in dl_va:
                img = batch['image'].to(device)
                mask = batch['mask'].to(device)
                pr = torch.sigmoid(model(img))
                pr_bin = (pr>0.5).float()

                inter1 += ((pr_bin*mask)==1).sum().item()
                union1 += ((pr_bin+mask)>=1).sum().item()

                pr0 = 1-pr_bin; gt0 = 1-mask
                inter0 += ((pr0*gt0)==1).sum().item()
                union0 += ((pr0+gt0)>=1).sum().item()

        miou = 0.5*((inter0/(union0+1e-6)) + (inter1/(union1+1e-6)))

        val_metrics, val_matrix = evaluate_split_confusion(dl_va, model, device, threshold=0.5)
        print("Val confusion matrix [[tn, fp], [fn, tp]]:", val_matrix)
        print("Val metrics:",
            {k: (round(v,4) if isinstance(v, float) else v) for k,v in val_metrics.items()})

        if miou > best:
            best = miou; patience=0
            os.makedirs(args.out, exist_ok=True)
            torch.save({'model': model.state_dict(), 'args': vars(args)},
                       os.path.join(args.out, 'best.pt'))
        else:
            patience += 1
        print(f"Epoch {epoch}: val mIoU={miou:.4f} best={best:.4f} patience={patience}")
        if patience>=args.early_stop:
            break

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
        # train_hybrid(args)
    else:
        infer(args)

if __name__ == '__main__':
    main()
