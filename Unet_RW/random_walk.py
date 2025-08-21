import cv2
import numpy as np
from skimage.segmentation import random_walker


def seeds_from_scribble(scrib):

    seeds = np.full_like(scrib, fill_value=-1, dtype=np.int32)
    seeds[scrib == 0] = 0
    seeds[scrib == 1] = 1
    return seeds

def rw_proba(img, scrib, beta=130, gamma=0.0, mode="cg_mg"):

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab = cv2.GaussianBlur(lab, (0, 0), 0.8)

    seeds = seeds_from_scribble(scrib)  

    prob = random_walker(
        lab,
        seeds,
        beta=beta,
        mode=mode,
        return_full_prob=True,
        channel_axis=-1,
    )

    if prob.ndim != 3:
        raise ValueError(f"Unexpected prob ndim from random_walker: {prob.ndim}")
    if prob.shape[0] in (1, 2, 3) and prob.shape[-1] not in (1, 2, 3):

        prob = np.moveaxis(prob, 0, -1)

    C = prob.shape[-1]
    has_fg = (scrib == 1).any()
    has_bg = (scrib == 0).any()

    if C == 2:
        pfg = prob[..., 1]
    elif C == 1:

        single = prob[..., 0]
        if has_fg and not has_bg:
            pfg = single
        elif has_bg and not has_fg:
            pfg = 1.0 - single
        else:

            pfg = np.full(scrib.shape, 0.5, dtype=np.float32)
    else:
        raise ValueError(f"Unexpected prob shape from random_walker: {prob.shape}")

    pfg = np.squeeze(pfg)
    pfg = np.nan_to_num(pfg, nan=0.5)

    if pfg.shape != scrib.shape:
        pfg = cv2.resize(pfg, (scrib.shape[1], scrib.shape[0]), interpolation=cv2.INTER_LINEAR)

    return pfg