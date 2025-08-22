import os, glob, json, argparse
import numpy as np
import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample

from SVM.main import gather_training_data as _GTD_FUNC

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def imread_rgb(fp):
    x = cv2.imread(fp, cv2.IMREAD_COLOR)
    if x is None: raise FileNotFoundError(fp)
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

def imread_gray(fp):
    x = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    if x is None: raise FileNotFoundError(fp)
    return x

def image_summary_features(img_rgb_u8, gt_mask_u8):
    """Return a small, robust feature vector per image."""
    h, w = gt_mask_u8.shape
    if img_rgb_u8.shape[:2] != (h, w):
        img_rgb_u8 = cv2.resize(img_rgb_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    L, A, B = lab[...,0], lab[...,1], lab[...,2]
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx*gx + gy*gy)

    fg = (gt_mask_u8 > 0).astype(np.uint8)
    fg_prop = float(fg.mean())

    def stats(x):
        return [float(np.mean(x)), float(np.std(x))]
    vec = []
    vec += stats(L) + stats(A) + stats(B)
    vec += stats(gmag)
    vec += [fg_prop]
    return np.array(vec, dtype=np.float32)

def build_image_level_matrix(root, exts=(".png",".jpg",".jpeg")):
    img_dir = os.path.join(root, "train/images")
    gt_dir  = os.path.join(root, "train/ground_truth")
    img_paths = []
    for e in exts:
        img_paths += sorted(glob.glob(os.path.join(img_dir, f"*{e}")))
    rows = []
    ids = []
    for p_img in tqdm(img_paths, desc="Image-level summaries"):
        base = os.path.splitext(os.path.basename(p_img))[0]
        p_gt = None
        for e in exts:
            q = os.path.join(gt_dir, base+e)
            if os.path.exists(q):
                p_gt = q; break
        if p_gt is None: continue
        img = imread_rgb(p_img)
        gt  = imread_gray(p_gt)
        rows.append(image_summary_features(img, gt))
        ids.append(base)
    if len(rows)==0:
        raise RuntimeError("No images with GT found.")
    X = np.vstack(rows)   
    return ids, X

def safe_scale_pca(X, pca_dim=None):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)

    if pca_dim is not None and pca_dim > 0 and pca_dim < Xs.shape[1]:
        pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=0)
        Xp = pca.fit_transform(Xs)

        Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)
        return Xp, {"scaler": scaler, "pca": pca}
    return Xs, {"scaler": scaler, "pca": None}

def run_tsne(X, perplexity=30, random_state=0):
    tsne = TSNE(
        n_components=2, perplexity=perplexity, learning_rate="auto",
        init="random", random_state=random_state, metric="euclidean", verbose=2
    )
    Z = tsne.fit_transform(X)
    return Z

def plot_tsne(Z, colors, title, out_png, legend_labels=None, dpi=140, alpha=0.7):
    plt.figure(figsize=(7.5, 6), dpi=dpi)
    if legend_labels is None:
        plt.scatter(Z[:,0], Z[:,1], s=6, c=colors, alpha=alpha)
    else:
        uniq = np.unique(colors)
        for u in uniq:
            idx = (colors==u)
            plt.scatter(Z[idx,0], Z[idx,1], s=6, alpha=alpha, label=legend_labels.get(u, str(u)))
        plt.legend(frameon=False, markerscale=3)
    plt.title(title)
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def tsne_pixel_level(args, out_dir):
    if _GTD_FUNC is None:
        print("[Info] Skipping pixel-level t-SNE: gather_training_data not found.")
        return

    X, y, group = _GTD_FUNC(
            train_img_dir="dataset/train/images",
            train_scrib_dir="dataset/train/scribbles",
            train_gt_dir= "dataset/train/ground_truth",
            size=args.size,
            per_image_fg=args.per_image_fg,
            per_image_bg=args.per_image_bg,
            return_groups=True)

    mask = np.isfinite(X).all(axis=1)
    X = X[mask]; y = y[mask]

    n_target = min(args.pixel_tsne_samples, len(y))
    n_each = n_target // 2
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    pos_idx = resample(pos, replace=False, n_samples=min(n_each, len(pos)), random_state=0)
    neg_idx = resample(neg, replace=False, n_samples=min(n_each, len(neg)), random_state=0)
    idx = np.concatenate([pos_idx, neg_idx])
    Xs, ys = X[idx], y[idx]

    Xp, _ = safe_scale_pca(Xs, pca_dim=None)

    perplexities = [10, 30, 50]
    for per in perplexities:
        Z = run_tsne(Xp, perplexity=per, random_state=42)

        col = np.where(ys==1, "#2ca02c", "#7f7f7f")
        fn = os.path.join(out_dir, f"tsne_pixels_perp{per}.png")
        plot_tsne(Z, col, f"Pixel t-SNE (perplexity={per})", fn)

        np.savez_compressed(os.path.join(out_dir, f"tsne_pixels_perp{per}.npz"),
                            Z=Z, y=ys.astype(np.uint8))

    with open(os.path.join(out_dir, "pixel_tsne_meta.json"), "w") as f:
        json.dump({
            "n_pixels_used": int(len(ys)),
            "n_fg": int((ys==1).sum()),
            "n_bg": int((ys==0).sum()),
            "perplexities": perplexities
        }, f, indent=2)

def tsne_image_level(args, out_dir):
    ids, X = build_image_level_matrix(args.data, exts=tuple(args.exts))

    Xp, _ = safe_scale_pca(X, pca_dim=min(10, X.shape[1]-1) if X.shape[1]>2 else None)
    Z = run_tsne(Xp, perplexity=args.image_tsne_perplexity, random_state=42)


    fg_prop = X[:, -1]

    c = (fg_prop - fg_prop.min()) / (np.ptp(fg_prop) + 1e-9)

    fn = os.path.join(out_dir, "tsne_images.png")
    plt.figure(figsize=(7.5,6), dpi=140)
    sc = plt.scatter(Z[:,0], Z[:,1], s=18, c=c, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="FG proportion")
    plt.title("Image-level t-SNE (colored by FG proportion)")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(fn, bbox_inches="tight"); plt.close()


    import csv
    csv_path = os.path.join(out_dir, "tsne_images_embedding.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id","tsne_x","tsne_y","fg_proportion"])
        for img_id, (x,y_), p in zip(ids, Z, fg_prop):
            w.writerow([img_id, float(x), float(y_), float(p)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--exts", type=str, default=".png,.jpg,.jpeg")
    ap.add_argument("--out", type=str, default="./Data_Analysis/t_SNE_results_4000_1000")


    ap.add_argument("--per_image_fg", type=int, default=4000)
    ap.add_argument("--per_image_bg", type=int, default=1000)
    ap.add_argument("--pixel_tsne_samples", type=int, default=60000, help="max pixels used (stratified)")


    ap.add_argument("--image_tsne_perplexity", type=float, default=35.0)

    args = ap.parse_args()
    args.exts = [e.strip() for e in args.exts.split(",") if e.strip()]

    ensure_dir(args.out)


    tsne_pixel_level(args, args.out)


    tsne_image_level(args, args.out)

    print(f"[Done] t-SNE plots and embeddings saved under: {args.out}")

if __name__ == "__main__":
    main()
