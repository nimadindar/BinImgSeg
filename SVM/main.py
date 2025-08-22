
import os, glob, json, argparse, time
from tqdm import tqdm
import numpy as np
import cv2
from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight


def imread(fp):
    img = cv2.imread(fp, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(fp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imread_gray(fp):
    m = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(fp)
    return m

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _norm01(x, eps=1e-6):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def _coords_hw(h, w):
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ys = ys.astype(np.float32) / max(h-1, 1)
    xs = xs.astype(np.float32) / max(w-1, 1)
    return ys, xs

def _distance_features_from_scribble(scrib):
    s = scrib.copy()
    s = np.where((s == 255) | (s == 2), 255, s).astype(np.uint8)
    fg = (s == 1).astype(np.uint8)
    bg = (s == 0).astype(np.uint8)

    inv_fg = 1 - fg
    d_fg = cv2.distanceTransform(inv_fg, distanceType=cv2.DIST_L2, maskSize=3)
    d_fg = _norm01(d_fg)

    inv_bg = 1 - bg
    d_bg = cv2.distanceTransform(inv_bg, distanceType=cv2.DIST_L2, maskSize=3)
    d_bg = _norm01(d_bg)

    k_fg = np.exp(- (d_fg**2) / (2 * (0.15**2) + 1e-6))
    k_bg = np.exp(- (d_bg**2) / (2 * (0.15**2) + 1e-6))
    return d_fg, d_bg, k_fg, k_bg

def _scharr_grad_mag(gray01):

    gx = cv2.Scharr(gray01, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray01, cv2.CV_32F, 0, 1)
    gmag = cv2.magnitude(gx, gy)
    return _norm01(gmag)

def extract_features(img_rgb_u8, scrib_u8, use_lbp=False):
    """
    Returns (H, W, D) feature tensor.
    Features:
      - Lab channels (3), normalized
      - Gradient magnitude (Scharr) (1)
      - Local mean/std (box 7x7) on Lab (6)
      - Spatial coords y,x normalized (2)
      - Distances to scribbles (d_fg, d_bg) + soft kernels (k_fg, k_bg) (4)
    Optional:
      - LBP (disabled by default)
    """
    h, w = scrib_u8.shape[:2]

    if img_rgb_u8.shape[:2] != (h, w):
        img_rgb_u8 = cv2.resize(img_rgb_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

    gmag = _scharr_grad_mag(L)

    ksz = 7
    def box_stats(ch):
        mean = cv2.blur(ch, (ksz, ksz))
        sq   = cv2.blur(ch*ch, (ksz, ksz))
        var  = np.clip(sq - mean*mean, 0, None)
        std  = np.sqrt(var + 1e-8)
        return mean, std
    mL, sL = box_stats(L); mA, sA = box_stats(A); mB, sB = box_stats(B)

    ys, xs = _coords_hw(h, w)

    d_fg, d_bg, k_fg, k_bg = _distance_features_from_scribble(scrib_u8)

    feats = [L, A, B, gmag, mL, sL, mA, sA, mB, sB, ys, xs, d_fg, d_bg, k_fg, k_bg]

    if use_lbp:
        mean3 = cv2.blur(L, (3,3))
        lbp = (L >= mean3).astype(np.float32)
        feats.append(lbp)

    F = np.stack(feats, axis=-1).astype(np.float32)
    return F

def refine_prob(rgb_u8, prob_01, do_bilateral=True):
    p = prob_01.astype(np.float32)
    if do_bilateral:
        p = cv2.bilateralFilter(p, d=0, sigmaColor=0.1, sigmaSpace=10)
    p_bin = (p > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    p_bin = cv2.morphologyEx(p_bin, cv2.MORPH_OPEN, k, iterations=1)
    p_bin = cv2.morphologyEx(p_bin, cv2.MORPH_CLOSE, k, iterations=1)
    return p, p_bin

def list_with_exts(folder, exts=(".png", ".jpg", ".jpeg")):
    paths = []
    for e in exts:
        paths += sorted(glob.glob(os.path.join(folder, f"*{e}")))
    return paths

def match_by_basename(src_paths, dst_folder, exts=(".png",".jpg",".jpeg")):
    out = []
    for p in src_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        found = None
        for e in exts:
            q = os.path.join(dst_folder, base + e)
            if os.path.exists(q):
                found = q; break
        out.append(found)
    return out

def sample_training_pixels(gt01, n_fg=1000, n_bg=1000, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    fg_yx = np.argwhere(gt01 == 1)
    bg_yx = np.argwhere(gt01 == 0)
    if len(fg_yx) == 0 or len(bg_yx) == 0:
        return None, None
    fg_sel = fg_yx[rng.choice(len(fg_yx), size=min(n_fg, len(fg_yx)), replace=False)]
    bg_sel = bg_yx[rng.choice(len(bg_yx), size=min(n_bg, len(bg_yx)), replace=False)]
    sel = np.vstack([fg_sel, bg_sel])
    y = np.concatenate([np.ones(len(fg_sel), np.uint8), np.zeros(len(bg_sel), np.uint8)])
    return sel, y

def gather_training_data(train_img_dir, train_scrib_dir, train_gt_dir,
                         size=384, per_image_fg=1200, per_image_bg=1200,
                         exts=(".png",".jpg",".jpeg"), use_lbp=False, augment_flips=True,
                         return_groups: bool = False):
    img_paths  = list_with_exts(train_img_dir, exts)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {train_img_dir}. Check --data path and folder structure.")
    scrib_paths= match_by_basename(img_paths, train_scrib_dir, exts)
    gt_paths   = match_by_basename(img_paths, train_gt_dir, exts)

    X_list, y_list = [], []
    groups_list = [] if return_groups else None

    for img_id, (p_img, p_scrib, p_gt) in enumerate(
            tqdm(zip(img_paths, scrib_paths, gt_paths),
                 total=len(img_paths), desc="Build SVM train set")):
        if p_scrib is None or p_gt is None:
            continue

        img   = imread(p_img)
        scrib = imread_gray(p_scrib)
        gt    = imread_gray(p_gt)
        gt01  = (cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)

        if size is not None:
            img   = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            scrib = cv2.resize(scrib, (size, size), interpolation=cv2.INTER_NEAREST)
            gt01  = cv2.resize(gt01, (size, size), interpolation=cv2.INTER_NEAREST)

        F = extract_features(img, scrib, use_lbp=use_lbp)   
        sel, y = sample_training_pixels(gt01, per_image_fg, per_image_bg)
        if sel is None:
            continue
        X = F[sel[:,0], sel[:,1], :]
        X_list.append(X)
        y_list.append(y)
        if return_groups:
            groups_list.append(np.full(len(y), img_id, dtype=np.int32))

        if augment_flips:
            img_f   = np.ascontiguousarray(img[:, ::-1, :])
            scrib_f = np.ascontiguousarray(scrib[:, ::-1])
            gt_f    = np.ascontiguousarray(gt01[:, ::-1])

            Ff = extract_features(img_f, scrib_f, use_lbp=use_lbp)
            sel_f, y_f = sample_training_pixels(gt_f, per_image_fg//2, per_image_bg//2)
            if sel_f is not None:
                Xf = Ff[sel_f[:,0], sel_f[:,1], :]
                X_list.append(Xf)
                y_list.append(y_f)
                if return_groups:
                    groups_list.append(np.full(len(y_f), img_id, dtype=np.int32))

    if len(X_list) == 0:
        raise RuntimeError("No training samples found. Do your scribbles and ground_truth match image basenames?")
    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.concatenate(y_list).astype(np.uint8)

    if return_groups:
        groups_all = np.concatenate(groups_list).astype(np.int32)
        return X_all, y_all, groups_all
    else:
        return X_all, y_all


def train_svm_classifier(X, y, C=2.0, gamma="scale"):
    steps = [("scaler", StandardScaler())]

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    svc = SVC(
        C=C, kernel="linear", gamma=gamma,
        probability=False, class_weight="balanced", random_state=0, verbose=True, shrinking=False
    )
    steps.append(("svc", svc))
    clf = Pipeline(steps)

    print("\n[SVM] ===== Training configuration =====")
    print(f"[SVM] C={C}, gamma={gamma}, samples={len(y)}, "
          f"features={X.shape[1]}, class balance={np.bincount(y)}")
    print("[SVM] Starting training...")

    with tqdm(total=3, desc="Train SVM", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:

        clf.named_steps["scaler"].fit(X)
        pbar.update(1)

        clf.named_steps["svc"].fit(
            clf.named_steps["scaler"].transform(X), y
        )
        pbar.update(2)

        time.sleep(0.1)  
        pbar.update(3)

    print(f"[SVM] Training complete. #Support vectors: {len(clf.named_steps['svc'].support_)}")
    return clf


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _predict_scores_chunk(clf, X_chunk):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_chunk)[:, 1]
    else:
        s = clf.decision_function(X_chunk)
        if s.ndim > 1:  
            s = s[:, 0]
        return _sigmoid(s)

def predict_image_svm(clf, img_rgb_u8, scrib_u8, size=384, refine=True, thr=0.5,
                      n_jobs=1, batch_size=262_144):
    """
    Parallelized over pixel-batches.
      - n_jobs: processes to use (<= #CPU cores). 1 = no parallelism.
      - batch_size: number of pixels per chunk (tune for memory/CPU).
    """
    if size is not None:
        img = cv2.resize(img_rgb_u8, (size, size), interpolation=cv2.INTER_LINEAR)
        scrib = cv2.resize(scrib_u8, (size, size), interpolation=cv2.INTER_NEAREST)
    else:
        img, scrib = img_rgb_u8, scrib_u8

    F = extract_features(img, scrib, use_lbp=False)  
    H, W, D = F.shape
    X = F.reshape(-1, D)

    if n_jobs == 1:
        probs_flat = _predict_scores_chunk(clf, X).astype(np.float32)
    else:
        indices = list(range(0, X.shape[0], batch_size))
        chunks = (X[i:i+batch_size] for i in indices)
        results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=1)(
            delayed(_predict_scores_chunk)(clf, chunk) for chunk in chunks
        )
        probs_flat = np.concatenate(results, axis=0).astype(np.float32)

    probs = probs_flat.reshape(H, W)

    if refine:
        _, pred_bin = refine_prob(img, probs)
    else:
        pred_bin = (probs > thr).astype(np.uint8)

    return probs, pred_bin



def confusion_and_scores(pred01, gt01):
    pr = pred01.astype(np.uint8); gt = gt01.astype(np.uint8)
    tp = int(((pr == 1) & (gt == 1)).sum())
    fp = int(((pr == 1) & (gt == 0)).sum())
    fn = int(((pr == 0) & (gt == 1)).sum())
    tn = int(((pr == 0) & (gt == 0)).sum())
    iou_fg = tp / max(tp + fp + fn, 1)
    iou_bg = tn / max(tn + fp + fn, 1)
    miou   = 0.5*(iou_fg + iou_bg)
    dice   = 2*tp / max(2*tp + fp + fn, 1)
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"iou_fg":iou_fg,"iou_bg":iou_bg,"miou":miou,"dice":dice}

def main():
    train = False
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset")
    ap.add_argument("--out", type=str, default="SVM/outputs_svm_balanced")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--exts", type=str, default=".png,.jpg,.jpeg")
    ap.add_argument("--per_image_fg", type=int, default=400)
    ap.add_argument("--per_image_bg", type=int, default=100)
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--gamma", type=str, default="scale")  
    ap.add_argument("--refine", action="store_true", help="edge-aware postprocessing")
    ap.add_argument("--eval_split", action="store_true", help="evaluate on a 15% holdout from train")
    ap.add_argument("--save_test_preds", action="store_true", help="run on dataset/test and save PNGs")
    args = ap.parse_args()

    ensure_dir(args.out)
    exts = tuple([e.strip() for e in args.exts.split(",") if len(e.strip())])

    tr_img_dir = os.path.join(args.data, "train/images")
    tr_scr_dir = os.path.join(args.data, "train/scribbles")
    tr_gt_dir  = os.path.join(args.data, "train/ground_truth")
    X, y = gather_training_data(tr_img_dir, tr_scr_dir, tr_gt_dir,
                                size=args.size,
                                per_image_fg=args.per_image_fg,
                                per_image_bg=args.per_image_bg,
                                exts=exts,
                                use_lbp=False,
                                augment_flips=True)
    
    if train:
        print(f"[SVM] Training samples: {X.shape}, positives={int(y.sum())}, negatives={int((y==0).sum())}")

        gamma = float(args.gamma) if args.gamma not in ("scale", "auto") else args.gamma
        clf = train_svm_classifier(X, y, C=args.C, gamma=gamma)

        import joblib
        joblib.dump(clf, os.path.join(args.out, "svm_pipeline.joblib"), compress=3)
        with open(os.path.join(args.out, "svm_meta.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    if args.eval_split:
        img_paths = list_with_exts(tr_img_dir, exts)
        if len(img_paths) == 0:
            print("[Eval] No images found; skipping.")
        else:
            rng = np.random.default_rng(42)
            n = len(img_paths)
            idxs = np.arange(n); rng.shuffle(idxs)
            n_val = max(1, int(0.15*n))
            val_idx = set(idxs[:n_val])

            stats = []
            for i, p_img in enumerate(tqdm(img_paths, desc="Eval holdout")):
                if i not in val_idx: continue
                base = os.path.splitext(os.path.basename(p_img))[0]
                p_scrib = None; p_gt = None
                for e in exts:
                    q = os.path.join(tr_scr_dir, base+e)
                    if os.path.exists(q): p_scrib=q; break
                for e in exts:
                    q = os.path.join(tr_gt_dir, base+e)
                    if os.path.exists(q): p_gt=q; break
                if p_scrib is None or p_gt is None: continue

                img  = imread(p_img)
                scrib= imread_gray(p_scrib)
                gt   = imread_gray(p_gt)
                gt01 = (cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)

                probs, pred = predict_image_svm(clf, img, scrib, size=args.size, refine=args.refine, thr=0.5)
                pred_full = cv2.resize(pred.astype(np.uint8), (gt01.shape[1], gt01.shape[0]), interpolation=cv2.INTER_NEAREST)
                scores = confusion_and_scores(pred_full, gt01)
                stats.append(scores)

            if stats:
                agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0].keys()}
                print(f"[Holdout] mIoU={agg['miou']:.4f}  Dice={agg['dice']:.4f}  IoU_fg={agg['iou_fg']:.4f}  IoU_bg={agg['iou_bg']:.4f}")
                with open(os.path.join(args.out, "holdout_metrics.json"), "w") as f:
                    json.dump(agg, f, indent=2)

    if args.save_test_preds:
        te_img_dir = os.path.join(args.data, "test/images")
        te_scr_dir = os.path.join(args.data, "test/scribbles")
        pred_dir = os.path.join(args.out, "predictions_test")
        ensure_dir(pred_dir)

        img_paths = list_with_exts(te_img_dir, exts)
        for p_img in tqdm(img_paths, desc="Predict test"):
            base = os.path.splitext(os.path.basename(p_img))[0]
            p_scrib = None
            for e in exts:
                q = os.path.join(te_scr_dir, base+e)
                if os.path.exists(q): p_scrib=q; break
            if p_scrib is None:
                continue
            img  = imread(p_img)
            scrib= imread_gray(p_scrib)

            probs, pred = predict_image_svm(clf, img, scrib, size=args.size, refine=args.refine, thr=0.5)
            out = (pred*255).astype(np.uint8)
            cv2.imwrite(os.path.join(pred_dir, base + ".png"), out)

if __name__ == "__main__":
    main()