"""
Add a missing Class_Int field to shapefiles by copying the existing `class` field.

This script edits shapefiles in place.
"""

import argparse
import glob
import os
import config


def fix_one(shp_path: str) -> str:
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    cols = list(gdf.columns)

    if "Class_Int" in cols:
        return "already_ok"
    if "class" not in cols:
        return "missing_class"

    gdf["Class_Int"] = gdf["class"].astype("Int64")
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    return "fixed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--shp_dir",
    default=config.SAMPLE_TRUTH_DIR,
    help="Directory containing truth shapefiles. Defaults to config.SAMPLE_TRUTH_DIR.",
    )

    parser.add_argument(
        "--pattern",
        default="*.shp",
    )
    args = parser.parse_args()

    fixed = []
    already_ok = []
    missing_class = []

    for shp_path in sorted(glob.glob(os.path.join(args.shp_dir, args.pattern))):
        status = fix_one(shp_path)
        name = os.path.basename(shp_path)
        if status == "fixed":
            fixed.append(name)
        elif status == "already_ok":
            already_ok.append(name)
        else:
            missing_class.append(name)

    print(f"[fix_class_int] fixed: {len(fixed)}")
    for name in fixed:
        print(name)
    print(f"[fix_class_int] already_ok: {len(already_ok)}")
    print(f"[fix_class_int] missing_class: {len(missing_class)}")
    for name in missing_class:
        print(name)


if __name__ == "__main__":
    main()
