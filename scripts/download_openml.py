# scripts/download_openml.py

from __future__ import annotations

import os, sys, argparse, re, gzip, json, yaml, subprocess, multiprocessing as mp
from pathlib import Path
from typing import Tuple, List

import numpy as np
import openml
from tqdm import tqdm

# Environment – force single‑threaded BLAS to avoid over‑subscribing CPUs
for v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
):
    os.environ.setdefault(v, "1")

REPO_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_DIR / "data"
CONFIG_YML = REPO_DIR / "configs" / "datasets.yaml"


# Helpers
def extract_task_id(tag: str) -> int:
    m = re.match(r".*__(\d+)$", tag)
    if m is None:
        raise ValueError(f"Cannot parse OpenML task ID from tag: {tag}")
    return int(m.group(1))


def _download_tag(args: Tuple[str, str]) -> Tuple[str, bool, str]:
    """
    Worker process.

    Returns
    -------
    tag, ok, msg   (ok==True -> success)
    """
    tag, target_type = args
    out_dir = DATA_DIR / tag
    try:
        # already there?
        if (out_dir / "X.npy.gz").is_file() and (out_dir / "y.npy.gz").is_file():
            return tag, True, "cached"

        task_id = extract_task_id(tag)
        task    = openml.tasks.get_task(task_id)
        ds      = task.get_dataset()
        X_df, y_ser, cat_mask, _ = ds.get_data(
            target=task.target_name,
            dataset_format="dataframe"
        )
        X = X_df.to_numpy()
        y = y_ser.to_numpy()

        # write to disk atomically
        out_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_dir / "X.npy.gz", "wb") as f:
            np.save(f, X)
        with gzip.open(out_dir / "y.npy.gz", "wb") as f:
            np.save(f, y)

        meta = {
            "name":               tag,
            "target_type":        target_type,
            "num_instances":      int(X.shape[0]),
            "num_features":       int(X.shape[1]),
            "num_unique_labels":  int(np.unique(y).size),
            "cat_idx":            np.where(np.asarray(cat_mask, bool))[0].tolist(),
            "feature_names":      X_df.columns.tolist(),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return tag, True, "done"
    except Exception as e:                
        return tag, False, str(e)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download OpenML tasks (parallel).")
    p.add_argument("group", nargs="?", help="Group name in datasets.yaml.")
    p.add_argument("-c", "--config", default=str(CONFIG_YML),
                   help=f"Path to datasets.yaml (default: {CONFIG_YML})")
    p.add_argument("--one-tag", dest="one_tag", help="Download a single tag")
    p.add_argument("--target-type", "-t", choices=["classification", "regression"])
    p.add_argument("-j", "--jobs", type=int, default=min(8, mp.cpu_count()),
                   help="Parallel workers (default: min(8, n_cores))")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))

    # One‑tag mode
    if args.one_tag:
        if args.target_type is None:
            raise ValueError("--target-type required with --one-tag")
        tag, ok, msg = _download_tag((args.one_tag, args.target_type))
        print(f"[{'OK' if ok else 'FAIL'}] {tag}: {msg}")
        sys.exit(0 if ok else 1)

    # Group mode 
    if args.group not in cfg:
        raise KeyError(f"Group '{args.group}' not found in {args.config}")

    tags: List[str] = cfg[args.group]
    target_type     = "classification" if args.group.startswith("cls") else "regression"
    print(f"Downloading {len(tags)} tasks for group '{args.group}' "
          f"({target_type}) → {DATA_DIR}")

    # Submit to process pool
    with mp.Pool(processes=args.jobs) as pool:
        results = list(
            tqdm(pool.imap_unordered(_download_tag,
                                     [(t, target_type) for t in tags]),
                 total=len(tags))
        )

    # Summary
    n_ok  = sum(ok for _, ok, _ in results)
    n_err = len(results) - n_ok
    print(f"\n✔ {n_ok} done   ✗ {n_err} failed\n")
    for tag, ok, msg in sorted(results):
        if not ok:
            print(f"   {tag}  →  {msg}")

    sys.exit(0 if n_err == 0 else 1)


if __name__ == "__main__":
    main()