import os, glob, csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

def fg_proportion_by_image(gt_dir, out_dir="analysis_outputs",
                           exts=(".png",".jpg",".jpeg"), show=True, dpi=120):
    os.makedirs(out_dir, exist_ok=True)

    # 1) collect masks
    paths = []
    for e in exts:
        paths.extend(sorted(glob.glob(os.path.join(gt_dir, f"*{e}"))))
    if not paths:
        raise RuntimeError(f"No GT masks found in {gt_dir}")

    # 2) compute per-image proportions
    rows = []  # (image_id, fg_prop)
    for p in paths:
        img_id = os.path.splitext(os.path.basename(p))[0]
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m_bin = (m > 0).astype(np.uint8)
        fg = int(m_bin.sum())
        tot = int(m_bin.size)
        fg_prop = fg / (tot + 1e-9)
        rows.append((img_id, fg_prop))

    if not rows:
        raise RuntimeError("No valid masks loaded.")

    # 3) save CSV
    csv_path = os.path.join(out_dir, "fg_proportion_per_image.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "fg_proportion"])
        w.writerows(rows)
    print(f"[Saved] {csv_path}")

    # 4) plots
    img_ids = [r[0] for r in rows]
    fg_props = np.array([r[1] for r in rows], dtype=np.float32)

    # A) scatter: original order (index ~ dataset order)
    plt.figure(figsize=(10, 4), dpi=dpi)
    plt.scatter(np.arange(len(fg_props)), fg_props, s=12, alpha=0.7)
    plt.xlabel("Image index (dataset order)")
    plt.ylabel("Foreground proportion")
    plt.title("FG proportion per image (unsorted)")
    plt.grid(alpha=0.25)
    if show: plt.show()
    else:
        p1 = os.path.join(out_dir, "fg_prop_scatter_unsorted.png")
        plt.savefig(p1, bbox_inches="tight"); plt.close()
        print(f"[Saved] {p1}")

    # B) scatter: sorted by proportion (reveals clustering streaks)
    order = np.argsort(fg_props)
    plt.figure(figsize=(10, 4), dpi=dpi)
    plt.scatter(np.arange(len(fg_props)), fg_props[order], s=12, alpha=0.7)
    plt.xlabel("Image rank (sorted by FG proportion)")
    plt.ylabel("Foreground proportion")
    plt.title("FG proportion per image (sorted)")
    plt.grid(alpha=0.25)
    if show: plt.show()
    else:
        p2 = os.path.join(out_dir, "fg_prop_scatter_sorted.png")
        plt.savefig(p2, bbox_inches="tight"); plt.close()
        print(f"[Saved] {p2}")

    # Optional: quick clustering in 1D (k=3) to highlight groups
    # (purely diagnostic; comment out if you don’t want it)
    try:
        from sklearn.cluster import KMeans
        k = 3
        km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(fg_props.reshape(-1,1))
        labels = km.labels_
        # color-coded scatter (unsorted)
        plt.figure(figsize=(10, 4), dpi=dpi)
        for lab in range(k):
            idx = np.where(labels == lab)[0]
            plt.scatter(idx, fg_props[idx], s=12, alpha=0.8, label=f"cluster {lab}")
        plt.xlabel("Image index")
        plt.ylabel("Foreground proportion")
        plt.title("FG proportion clusters (KMeans in 1D)")
        plt.legend()
        plt.grid(alpha=0.25)
        if show: plt.show()
        else:
            p3 = os.path.join(out_dir, "fg_prop_scatter_clusters.png")
            plt.savefig(p3, bbox_inches="tight"); plt.close()
            print(f"[Saved] {p3}")
    except Exception as e:
        print(f"[Info] Skipping clustering plot: {e}")

    # Return data for further analysis
    return rows, fg_props


rows, fg_props = fg_proportion_by_image("dataset/train_not_filtered/ground_truth",
                                        out_dir="Data_Analysis/fg_bg_per_image_not_filtered",
                                        exts=(".png",".jpg",".jpeg"),
                                        show=False)  # set True to display instead of saving

