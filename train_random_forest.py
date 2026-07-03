"""
Train the random forest pixel-wise spectral baseline.

This baseline uses the same prepared model inputs as the deep models:
reflectance bands plus spectral indices with the same channel normalization.
It samples valid labelled pixels from the training tiles and uses LCI as
sample_weight.
"""

import argparse
import os

import config
from random_forest_utils import train_random_forest_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default=os.path.join(config.SPLIT_DIR, "train_manifest.csv"),
        help="Training split manifest. Random forest refuses to train without it.",
    )
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--max_pixels_per_class", type=int, default=150000)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    if not args.manifest_csv or not os.path.exists(args.manifest_csv):
        raise FileNotFoundError(
            f"Random forest training manifest not found: {args.manifest_csv}. "
            "Refusing to fall back to all tiles."
        )

    config.ensure_runtime_dirs()
    config.print_config_summary()
    train_random_forest_classifier(
        manifest_csv=args.manifest_csv,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        max_pixels_per_class=args.max_pixels_per_class,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
