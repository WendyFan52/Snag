"""
Read-only audit for sample coverage and shapefile schema quality.

This script does not modify any shapefile.
It helps explain why a sample appears in manifest but not in summary.
"""

import argparse
import csv
import json
import os
from collections import Counter


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def audit_one_shp(shp_path):
    try:
        import geopandas as gpd
    except ImportError:
        raise RuntimeError("geopandas is required for audit")

    gdf = gpd.read_file(shp_path)
    cols = list(gdf.columns)
    has_class_int = "Class_Int" in cols
    has_class = "class" in cols
    geom_ok = 0
    if "geometry" in cols:
        geom_ok = int((gdf.geometry.notnull() & ~gdf.geometry.is_empty).sum())
    unique_vals = []
    if has_class_int:
        unique_vals = sorted(str(v) for v in gdf["Class_Int"].dropna().unique().tolist())
    return {
        "rows": len(gdf),
        "geom_ok": geom_ok,
        "has_Class_Int": has_class_int,
        "has_class": has_class,
        "columns": cols,
        "class_int_values": unique_vals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    manifest_rows = load_csv(args.manifest_csv)
    summary_rows = load_csv(args.summary_csv) if os.path.exists(args.summary_csv) else []
    summarized_names = {r.get("layer_name", "") for r in summary_rows}

    missing = [r for r in manifest_rows if r.get("layer_name", "") not in summarized_names]
    report = {
        "manifest_count": len(manifest_rows),
        "summary_count": len(summary_rows),
        "missing_count": len(missing),
        "missing_reasons": [],
        "schema_counter": Counter(),
    }

    for row in missing:
        shp_path = row.get("truth_shp_path", "")
        reason = {
            "layer_name": row.get("layer_name", ""),
            "truth_shp_path": shp_path,
            "exists": os.path.exists(shp_path),
        }
        if not reason["exists"]:
            reason["why"] = "missing_shp_file"
        else:
            try:
                info = audit_one_shp(shp_path)
                reason.update(info)
                if not info["has_Class_Int"]:
                    reason["why"] = "no_Class_Int"
                elif info["geom_ok"] == 0:
                    reason["why"] = "empty_or_invalid_geometry"
                else:
                    reason["why"] = "other_pipeline_reason"
                schema_key = f"Class_Int={info['has_Class_Int']};class={info['has_class']}"
                report["schema_counter"][schema_key] += 1
            except Exception as exc:
                reason["why"] = "read_error"
                reason["error"] = str(exc)
        report["missing_reasons"].append(reason)

    report["schema_counter"] = dict(report["schema_counter"])
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(json.dumps(
        {
            "manifest_count": report["manifest_count"],
            "summary_count": report["summary_count"],
            "missing_count": report["missing_count"],
            "schema_counter": report["schema_counter"],
            "output_json": args.output_json,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

