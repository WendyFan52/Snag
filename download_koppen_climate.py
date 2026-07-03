"""
Download KGClim 1984-2013 Koppen rasters.

By default this downloads only the two files needed by the framework:
kc_1984_2013.tif and kc_conf_1984_2013.tif. The full Zenodo archive remains
available through --archive, but it is much larger.
"""

import argparse
import os
import zipfile
from urllib.request import urlopen, Request

import config


def download(url: str, out_path: str, chunk_size: int = 1024 * 1024) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".part"
    existing = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    headers = {"User-Agent": "TreeDead-KGClim-downloader/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = Request(url, headers=headers)

    with urlopen(request) as response:
        resumed = existing and getattr(response, "status", 200) == 206
        mode = "ab" if resumed else "wb"
        downloaded = existing if resumed else 0
        total = response.headers.get("Content-Length")
        total = int(total) + downloaded if total else 0
        with open(tmp_path, mode) as fh:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r[download] {downloaded / total:6.2%}  {downloaded / (1024 ** 3):.2f}/{total / (1024 ** 3):.2f} GB", end="")
                else:
                    print(f"\r[download] {downloaded / (1024 ** 3):.2f} GB", end="")
    print()
    os.replace(tmp_path, out_path)
    return out_path


def extract(zip_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="store_true", help="Download the full Zenodo zip archive instead of only kc rasters")
    parser.add_argument("--url", default=config.KOPPEN_DOWNLOAD_URL)
    parser.add_argument("--zip_path", default=config.KOPPEN_ZIP_PATH)
    parser.add_argument("--extract_dir", default=config.KOPPEN_EXTRACT_DIR)
    parser.add_argument("--kc_url", default=config.KOPPEN_KC_URL)
    parser.add_argument("--kc_path", default=config.KOPPEN_KC_RASTER)
    parser.add_argument("--conf_url", default=config.KOPPEN_CONF_URL)
    parser.add_argument("--conf_path", default=config.KOPPEN_CONF_RASTER)
    parser.add_argument("--no_extract", action="store_true")
    args = parser.parse_args()

    if not args.archive:
        for url, path in ((args.kc_url, args.kc_path), (args.conf_url, args.conf_path)):
            if os.path.exists(path):
                print(f"[download] exists -> {path}")
            else:
                print(f"[download] {url}")
                print(f"[download] -> {path}")
                download(url, path)
        print("[done]")
        return

    if os.path.exists(args.zip_path):
        print(f"[download] exists -> {args.zip_path}")
    else:
        print(f"[download] {args.url}")
        print(f"[download] -> {args.zip_path}")
        download(args.url, args.zip_path)

    if not args.no_extract:
        print(f"[extract] -> {args.extract_dir}")
        extract(args.zip_path, args.extract_dir)
    print("[done]")


if __name__ == "__main__":
    main()
