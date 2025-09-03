import os
import numpy as np
from PIL import Image

def fix_masks(folder, save_fixed=True, out_dir=None):
    if save_fixed and out_dir is None:
        out_dir = os.path.join(folder, "fixed_masks")
    if save_fixed:
        os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue

        try:
            img = Image.open(fpath)
            arr = np.array(img)

            # Convert {0,255} → {0,1}
            if arr.max() == 255:
                arr = (arr > 0).astype(np.uint8)

                if save_fixed:
                    out_path = os.path.join(out_dir, fname)
                    Image.fromarray(arr * 255).save(out_path)  
                    # saved back as 0/255 for visibility
                else:
                    # overwrite original (saved as 0/1 in uint8)
                    Image.fromarray(arr).save(fpath)

        except Exception as e:
            print(f"Error fixing {fname}: {e}")

    print(f"[Done] Fixed masks saved to: {out_dir if save_fixed else folder}")


if __name__ == "__main__":
    folder_path = "SVM/outputs_svm_balanced/preds" 
    fix_masks(folder_path, save_fixed=True)  
