import os, glob
import numpy as np
import cv2

def compute_fg_bg_ratio(gt_dir, exts=(".png",".jpg",".jpeg")):
    gt_paths = []
    for e in exts:
        gt_paths.extend(glob.glob(os.path.join(gt_dir, f"*{e}")))
    if len(gt_paths) == 0:
        raise RuntimeError(f"No GT masks found in {gt_dir}")

    per_image_props = []
    total_fg, total_bg = 0, 0

    for p in gt_paths:
        gt = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt_bin = (gt > 0).astype(np.uint8)  
        fg = int(gt_bin.sum())
        bg = int(gt_bin.size - fg)
        total_fg += fg
        total_bg += bg
        per_image_props.append(fg / (fg + bg + 1e-9))

    avg_prop = float(np.mean(per_image_props))
    global_prop = total_fg / (total_fg + total_bg + 1e-9)

    print(f"[Per-image average proportion] FG: {avg_prop:.4f}, BG: {1-avg_prop:.4f}")
    print(f"[Global proportion over all pixels] FG: {global_prop:.4f}, BG: {1-global_prop:.4f}")
    return per_image_props, avg_prop, global_prop


gt_dir = "dataset/train/ground_truth"
per_image_props, avg_prop, global_prop = compute_fg_bg_ratio(gt_dir)
