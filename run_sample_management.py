"""
Run the full sample-management pipeline in one command.

Pipeline:
1. build_sample_manifest.py
2. summarize_samples.py
3. assign_background_type.py
4. split_samples.py
"""

import argparse
import os

import config
from assign_background_type import assign_backgrounds
from build_sample_manifest import (
    DEFAULT_CENTER_SOURCE,
    DEFAULT_CLIMATE_SHP,
    DEFAULT_TRUTH_DIR,
    build_manifest,
)
from split_samples import build_splits
from summarize_samples import summarize_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--center_source", default=DEFAULT_CENTER_SOURCE)
    parser.add_argument("--center_csv", dest="center_source")
    parser.add_argument("--climate_shp", dest="climate_shp", default=DEFAULT_CLIMATE_SHP)
    parser.add_argument("--thermal_shp", dest="climate_shp")
    parser.add_argument("--center_crs", default=config.SAMPLE_CENTER_CRS)
    parser.add_argument("--koppen_raster", default=None)
    parser.add_argument("--koppen_conf_raster", default=None)
    parser.add_argument(
        "--allow_legacy_climate",
        action="store_true",
        help="Allow fallback to old climate fields if Koppen raster is unavailable",
    )
    parser.add_argument(
        "--allow_extra_truth",
        action="store_true",
        help="Allow truth SHP files that have no matching sample-center row",
    )
    parser.add_argument("--truth_dir", default=DEFAULT_TRUTH_DIR)
    parser.add_argument("--image_dir", default=getattr(config, "SAMPLE_IMAGE_DIR", config.RAW_TIF_DIR))
    parser.add_argument(
        "--allow_missing_images",
        action="store_true",
        help="Allow labeled-area fallback when sample PlanetScope clips are missing",
    )
    parser.add_argument(
        "--allow_geometry_fallback",
        action="store_true",
        help="Allow fallback geometry area calculation if union/repair fails",
    )
    parser.add_argument(
        "--manifest_csv",
        default=os.path.join(config.MANIFEST_DIR, "sample_manifest.csv"),
    )
    parser.add_argument(
        "--summary_csv",
        default=os.path.join(config.SAMPLE_STATS_DIR, "sample_summary.csv"),
    )
    parser.add_argument(
        "--cluster_csv",
        default=os.path.join(config.SAMPLE_STATS_DIR, "cluster_summary.csv"),
    )
    parser.add_argument(
        "--labeled_summary_csv",
        default=os.path.join(config.SAMPLE_STATS_DIR, "sample_summary_labeled.csv"),
    )
    parser.add_argument(
        "--split_balance_csv",
        default=os.path.join(config.SPLIT_DIR, "split_balance_report.csv"),
    )
    parser.add_argument("--cluster_distance_m", type=float, default=3.0)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--scale_ratio", type=float, default=0.25)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--split_search_iterations", type=int, default=5000)
    args = parser.parse_args()
    if args.koppen_raster:
        config.KOPPEN_KC_RASTER = args.koppen_raster
    if args.koppen_conf_raster:
        config.KOPPEN_CONF_RASTER = args.koppen_conf_raster

    manifest_csv = build_manifest(
        center_source=args.center_source,
        climate_shp=args.climate_shp,
        truth_dir=args.truth_dir,
        output_csv=args.manifest_csv,
        center_crs=args.center_crs,
        image_dir=args.image_dir,
        require_koppen=not args.allow_legacy_climate,
        allow_extra_truth=args.allow_extra_truth,
    )

    summarize_manifest(
        manifest_csv=manifest_csv,
        output_csv=args.summary_csv,
        cluster_csv=args.cluster_csv,
        cluster_distance_m=args.cluster_distance_m,
        image_dir=args.image_dir,
        require_images=not args.allow_missing_images,
        allow_geometry_fallback=args.allow_geometry_fallback,
    )

    assign_backgrounds(
        summary_csv=args.summary_csv,
        output_csv=args.labeled_summary_csv,
    )

    build_splits(
        labeled_summary_csv=args.labeled_summary_csv,
        output_dir=config.SPLIT_DIR,
        train_ratio=args.train_ratio,
        scale_ratio=args.scale_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        balance_report_csv=args.split_balance_csv,
        search_iterations=args.split_search_iterations,
    )


if __name__ == "__main__":
    main()
