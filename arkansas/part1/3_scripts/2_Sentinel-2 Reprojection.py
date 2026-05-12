"""
FAST REPROJECTION PIPELINE (MULTIPROCESSING VERSION)
- Reprojects Sentinel-2 shards to EPSG:5070
- Runs multiple shards in parallel for speed
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS
from pathlib import Path
import time
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(r"C:\projects\resneur\projectcrops\project_crop_classiffication")

INPUT_DIR = BASE_DIR / "arkansas" / "part1" / "1_data_sentinel"
OUTPUT_DIR = BASE_DIR / "arkansas" / "part1" / "4_results" / "2_Sentinel-2 Reprojection_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CDL_CRS = CRS.from_epsg(5070)
S2_CRS = CRS.from_epsg(4326)

SHARDS = [
    "MCTNet_14GB_Arkansas_North-0000000000-0000000000.tif",
    "MCTNet_14GB_Arkansas_North-0000000000-0000001792.tif",
    "MCTNet_14GB_Arkansas_North-0000001792-0000000000.tif",
    "MCTNet_14GB_Arkansas_North-0000001792-0000001792.tif",
    "MCTNet_14GB_Arkansas_South-0000000000-0000000000.tif",
    "MCTNet_14GB_Arkansas_South-0000000000-0000001792.tif",
    "MCTNet_14GB_Arkansas_South-0000001792-0000000000.tif",
    "MCTNet_14GB_Arkansas_South-0000001792-0000001792.tif",
]

# ============================================================
# CORE FUNCTION (RUNS IN PARALLEL)
# ============================================================

def reproject_shard(shard_name):
    # FIXED: Use INPUT_DIR instead of BASE_DIR
    shard_path = INPUT_DIR / shard_name
    output_path = OUTPUT_DIR / f"{Path(shard_name).stem}_5070.tif"

    if not shard_path.exists():
        return f"❌ Missing {shard_name} from {INPUT_DIR}"

    if output_path.exists():
        return f"✅ Already exists {shard_name}"

    try:
        with rasterio.open(shard_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs,
                CDL_CRS,
                src.width,
                src.height,
                *src.bounds,
                resolution=30
            )

            profile = src.profile.copy()
            profile.update({
                "crs": CDL_CRS,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256
            })

            with rasterio.open(output_path, "w", **profile) as dst:
                for band in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=CDL_CRS,
                        resampling=Resampling.nearest
                    )

        return f"✅ Done {shard_name}"

    except Exception as e:
        return f"❌ Error {shard_name}: {e}"


# ============================================================
# MAIN FUNCTION (PARALLEL EXECUTION)
# ============================================================

def main():
    print("=" * 70)
    print("FAST REPROJECTION PIPELINE (PARALLEL VERSION)")
    print("=" * 70)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Shards: {len(SHARDS)}")
    print("=" * 70)

    start = time.time()

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(reproject_shard, s): s for s in SHARDS}
        for future in as_completed(futures):
            result = future.result()
            print(result)

    info_path = OUTPUT_DIR / "reprojection_info.json"
    with open(info_path, "w") as f:
        json.dump({"status": "completed", "num_shards": len(SHARDS)}, f, indent=2)

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(f"TOTAL TIME: {elapsed/60:.2f} minutes")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()