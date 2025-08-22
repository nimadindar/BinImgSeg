def svm_grid_search(
    data_root: str,
    out_dir: str,
    size: int = 384,
    exts: tuple = (".png",".jpg",".jpeg"),
    # --- outer sampling grid ---
    per_image_fg_grid = (10000, 50000, 100000),
    per_image_bg_grid = (10000, 50000, 100000),
    augment_flips: bool = True,
    use_lbp: bool = False,
    # --- inner SVM grid ---
    kernels = ("rbf", "linear", "poly"),
    Cs = (0.5, 1, 2, 4),
    gammas = ("scale", 0.01, 0.005, 0.001),
    degrees = (2, 3),                 # only for poly
    pca_components = (None,),         # None disables PCA; include ints to try PCA
    # --- CV / scoring ---
    scoring: str = "iou",             # "iou" (Jaccard) or "f1"
    cv_folds: int = 3,
    n_jobs: int = -1,
    random_state: int = 42,
    # --- UX ---
    use_progress_bars: bool = True,   # show tqdm bars for outer+inner loops
    print_every: bool = True          # organized console prints
):
    """
    Outer grid over per-image pixel sampling; inner GridSearchCV over SVM params.
    Saves a CSV of ALL trials + best model pipeline.
    Includes tqdm progress bars for both outer and inner loops.
    """
    # -------- imports --------
    import os, time, json
    import joblib
    import numpy as np
    import pandas as pd
    from joblib import parallel_backend
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, GroupKFold, ParameterGrid
    from sklearn.metrics import make_scorer, jaccard_score, f1_score

    # tqdm (optional)
    if use_progress_bars:
        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = None
            use_progress_bars = False
    else:
        tqdm = None

    # ---- helper: joblib <-> tqdm integration for inner CV progress ----
    from contextlib import contextmanager
    @contextmanager
    def tqdm_joblib(tqdm_object):
        """
        Context manager to patch joblib to report into tqdm progress bar.
        Counts completed tasks (batches) from joblib's Parallel.
        """
        import joblib as _joblib
        if tqdm_object is None:
            yield None
            return
        class TqdmBatchCallback(_joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                try:
                    tqdm_object.update(n=self.batch_size)
                finally:
                    return super().__call__(*args, **kwargs)
        old_cb = _joblib.parallel.BatchCompletionCallBack
        _joblib.parallel.BatchCompletionCallBack = TqdmBatchCallback
        try:
            yield tqdm_object
        finally:
            _joblib.parallel.BatchCompletionCallBack = old_cb
            tqdm_object.close()

    # -------- setup --------
    os.makedirs(out_dir, exist_ok=True)

    def _print(msg=""):
        if print_every:
            print(msg, flush=True)

    # nice section headers
    def _banner(text):
        if print_every:
            print("\n" + "=" * 80)
            print(text)
            print("=" * 80, flush=True)

    # scorers
    if scoring == "iou":
        scorer = make_scorer(jaccard_score, average="binary", pos_label=1, zero_division=0)
    elif scoring == "f1":
        scorer = make_scorer(f1_score, average="binary", pos_label=1, zero_division=0)
    else:
        raise ValueError("scoring must be 'iou' or 'f1'")

    # pipeline with optional PCA via grid ("passthrough" disables PCA)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("pca", "passthrough"),
        ("svc", SVC(probability=False, class_weight="balanced", random_state=random_state)),
    ])

    # kernel-aware grid (avoids irrelevant params)
    def _pca_options():
        opts = ["passthrough"]
        for k in pca_components:
            if k:  # skip None / 0
                opts.append(PCA(n_components=int(k), random_state=random_state))
        return opts

    grid = []
    if "linear" in kernels:
        grid.append({
            "pca": _pca_options(),
            "svc__kernel": ["linear"],
            "svc__C": list(Cs),
        })
    if "rbf" in kernels:
        grid.append({
            "pca": _pca_options(),
            "svc__kernel": ["rbf"],
            "svc__C": list(Cs),
            "svc__gamma": list(gammas),
        })
    if "poly" in kernels:
        grid.append({
            "pca": _pca_options(),
            "svc__kernel": ["poly"],
            "svc__C": list(Cs),
            "svc__gamma": list(gammas),
            "svc__degree": list(degrees),
        })

    cv = GroupKFold(n_splits=cv_folds)

    # ---- outer progress setup ----
    outer_total = len(per_image_fg_grid) * len(per_image_bg_grid)
    outer_iter = tqdm(total=outer_total, desc="Outer sampling grid", leave=True) if use_progress_bars else None

    summary_rows = []
    best_global = {
        "score": -1.0,
        "per_image_fg": None,
        "per_image_bg": None,
        "gs_best_params": None,
        "model_path": None,
    }

    _banner("SVM Grid Search: START")
    _print(f"Data root: {data_root}")
    _print(f"Output dir: {out_dir}")
    _print(f"Scoring: {scoring} | CV folds: {cv_folds} | n_jobs: {n_jobs}")
    _print(f"Outer grid sizes | FG: {per_image_fg_grid} | BG: {per_image_bg_grid}")
    _print(f"Kernels: {kernels} | Cs: {Cs} | gammas: {gammas} | degrees: {degrees}")
    _print(f"PCA candidates: {pca_components} (use 'None' to disable)")

    # ===== iterate outer sampling grid =====
    for per_fg in per_image_fg_grid:
        for per_bg in per_image_bg_grid:
            tag = f"fg{per_fg}_bg{per_bg}"
            _banner(f"Outer trial [{tag}]")
            _print("Gathering training data...")

            from .main import gather_training_data  # <- must return groups (image ids)
            X, y, groups = gather_training_data(
                train_img_dir=os.path.join(data_root, "train/images"),
                train_scrib_dir=os.path.join(data_root, "train/scribbles"),
                train_gt_dir=os.path.join(data_root, "train/ground_truth"),
                size=size,
                per_image_fg=per_fg,
                per_image_bg=per_bg,
                exts=exts,
                use_lbp=use_lbp,
                augment_flips=augment_flips,
                return_groups=True
            )

            X = np.asarray(X, dtype=np.float32)
            pos = int(y.sum())
            neg = int((y == 0).sum())
            _print(f"[Sampling] X={tuple(X.shape)} | positives={pos} | negatives={neg}")

            # prepare inner grid size and show ETA-ish info
            n_candidates = sum(len(list(ParameterGrid(g))) for g in grid)
            n_splits = cv.get_n_splits(X, y, groups)
            total_fits = n_candidates * n_splits
            _print(f"[Inner CV] candidates={n_candidates} | folds={n_splits} | total fits={total_fits}")

            # ---- GridSearch with live progress bar ----
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring=scorer,
                cv=cv,
                n_jobs=n_jobs,
                refit=True,
                verbose=0,  # tqdm handles progress instead of sklearn prints
                return_train_score=False,
            )

            t0 = time.time()
            inner_pbar = (tqdm(total=total_fits, desc=f"GridSearch {tag}", leave=False)
                          if use_progress_bars else None)

            with parallel_backend("loky", n_jobs=n_jobs):
                # tie joblib into tqdm so bar advances as fits finish
                with tqdm_joblib(inner_pbar):
                    gs.fit(X, y, groups=groups)

            dt = time.time() - t0
            _print(f"[Result] best_score={gs.best_score_:.5f} | best_params={gs.best_params_} | fit={dt:.2f}s")

            # Save full cv results for this sampling setting
            cv_csv = os.path.join(out_dir, f"svm_grid_cv_{tag}.csv")
            pd.DataFrame(gs.cv_results_).to_csv(cv_csv, index=False)
            _print(f"Saved inner CV results → {cv_csv}")

            # Save best pipeline
            best_path = os.path.join(out_dir, f"svm_best_pipeline_{tag}.joblib")
            joblib.dump(gs.best_estimator_, best_path)
            _print(f"Saved best pipeline → {best_path}")

            # Log outer summary row
            row = {
                "per_image_fg": per_fg,
                "per_image_bg": per_bg,
                "best_score": float(gs.best_score_),
                "best_params": json.dumps(gs.best_params_),
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "fit_seconds": round(dt, 2),
                "cv_csv": cv_csv,
                "model_path": best_path,
            }
            summary_rows.append(row)

            # Track global best
            if gs.best_score_ > best_global["score"] + 1e-12:
                best_global.update({
                    "score": float(gs.best_score_),
                    "per_image_fg": per_fg,
                    "per_image_bg": per_bg,
                    "gs_best_params": gs.best_params_,
                    "model_path": best_path,
                })
                _print(f"[Global best] score={best_global['score']:.5f} @ FG={per_fg}, BG={per_bg}")

            if outer_iter is not None:
                outer_iter.update(1)

    if outer_iter is not None:
        outer_iter.close()

    # write outer summary CSV
    outer_csv = os.path.join(out_dir, "svm_outer_sampling_summary.csv")
    pd.DataFrame(summary_rows).to_csv(outer_csv, index=False)

    # write best summary JSON
    best_json = os.path.join(out_dir, "svm_outer_best.json")
    with open(best_json, "w") as f:
        json.dump(best_global, f, indent=2)

    _banner("SVM Grid Search: COMPLETE")
    _print("=== BEST OVERALL (outer + inner) ===")
    _print(json.dumps(best_global, indent=2))
    _print(f"Saved outer CSV:  {outer_csv}")
    _print(f"Saved best JSON:  {best_json}")
    return best_global

# --- quick, simple run ---
# Goal: fast sanity check (linear SVM only, tiny grid, small sampling, 2-fold CV)

from pathlib import Path

best = svm_grid_search(
    data_root="dataset",        # contains train/images, train/scribbles, train/ground_truth
    out_dir=Path("SVM/svm_grid").as_posix(),
    size=384,
    exts=(".png", ".jpg", ".jpeg"),

    # FAST outer sampling (few pixels per image)
    per_image_fg_grid=(1000,),          # try 8k FG pixels per image
    per_image_bg_grid=(1000,),          # and 8k BG pixels per image
    augment_flips=False,                # turn off to speed up I/O
    use_lbp=False,

    # FAST inner search (linear only, no PCA)
    kernels=("linear",),                # avoid slow RBF/poly
    Cs=(0.5, 1, 2),                     # tiny C grid
    gammas=("scale",),                  # ignored by linear, harmless
    degrees=(2,),                       # unused here
    pca_components=(None,),             # disable PCA

    # FAST(er) CV
    scoring="iou",                      # or "f1"
    cv_folds=2,                         # 2-fold to halve inner CV time
    n_jobs=-1,                          # use all cores
    random_state=42,

    # Tidy console
    use_progress_bars=True,
    print_every=True
)

print("\nQuick run best summary:")
print(best)
