"""
01_organize_raw_downloads.py

Helper for the KSA Shield aeromagnetic data acquisition step.

The SGS National Geological Database Portal (https://ngp.sgs.gov.sa/, RGP - GPAS layer) has no public
bulk-download API, so the actual download is a manual step, see README.md. This script takes whatever .zip
archives were manually downloaded into an incoming folder and:

  1. Unzips every archive into a scratch folder.
  2. Sorts the resulting .tif/.tiff files into per-block subfolders under --output, based on the survey
     block code found in the filename (GEOPH1, GEOPH2, GEOPH3). Files that don't match any known code are
     placed in an UNSORTED/ folder for manual review.
  3. Prints a short inventory (CRS, pixel size, band count, dimensions) per file, so the download can be
     checked before running 03_snap_raster.py.

Usage
-----
    uv run python scripts/build_ksa_dataset/00_ksa_preprocessing/01_organize_raw_downloads.py \\
        --incoming data/raw/ksa_shield/_incoming --output data/raw/ksa_shield
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import rasterio

BLOCK_CODES = ["GEOPH1", "GEOPH2", "GEOPH3"]


def unzip_all(incoming_dir: Path, scratch_dir: Path) -> None:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(incoming_dir.glob("*.zip"))
    if not zips:
        print(f"[*] No .zip archives found in {incoming_dir} (skipping unzip step)")
        return
    for z in zips:
        print(f"[*] Extracting {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(scratch_dir)


def detect_block(filename: str) -> str:
    upper = filename.upper()
    for code in BLOCK_CODES:
        if code in upper:
            return code
    return "UNSORTED"


def sort_tiles(scratch_dir: Path, output_dir: Path) -> list[Path]:
    tif_files = sorted(scratch_dir.rglob("*.tif")) + sorted(scratch_dir.rglob("*.tiff"))
    sorted_paths = []
    for f in tif_files:
        block = detect_block(f.name)
        dest_dir = output_dir / block
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f.name
        if dest.exists():
            print(f"    [!] {dest} already exists — skipping to avoid overwrite")
            continue
        shutil.move(str(f), str(dest))
        sorted_paths.append(dest)
    return sorted_paths


def print_inventory(paths: list[Path]) -> None:
    if not paths:
        print("\n[!] No .tif/.tiff files found to sort — check the incoming folder.")
        return

    print(f"\n[*] Inventory — {len(paths)} tile(s) sorted\n")
    header = f"{'File':45s} {'Block':8s} {'EPSG':8s} {'Px size':10s} {'Bands':6s} {'Size (px)'}"
    print(header)
    print("-" * len(header))
    for p in paths:
        try:
            with rasterio.open(p) as ds:
                epsg = ds.crs.to_epsg() if ds.crs else "?"
                print(
                    f"{p.name:45s} {p.parent.name:8s} {str(epsg):8s} "
                    f"{ds.res[0]:<10.2f} {ds.count:<6d} {ds.width}x{ds.height}"
                )
        except rasterio.errors.RasterioIOError:
            print(f"{p.name:45s} [!] could not open")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incoming",
        required=True,
        type=Path,
        help="Folder containing manually downloaded .zip archives (or loose .tif files)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination root; tiles are sorted into <output>/<BLOCK>/ subfolders",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep the intermediate extraction folder instead of deleting it",
    )
    args = parser.parse_args()

    if not args.incoming.exists():
        print(f"[!] Incoming folder not found: {args.incoming}")
        return

    scratch_dir = args.incoming / "_extracted_scratch"
    unzip_all(args.incoming, scratch_dir)

    # Also pick up loose .tif/.tiff files sitting directly in --incoming
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for f in list(args.incoming.glob("*.tif")) + list(args.incoming.glob("*.tiff")):
        shutil.copy(str(f), str(scratch_dir / f.name))

    sorted_paths = sort_tiles(scratch_dir, args.output)

    if not args.keep_scratch:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    print_inventory(sorted_paths)
    print(f"\n[+] Done. Tiles sorted under: {args.output}")


if __name__ == "__main__":
    main()
