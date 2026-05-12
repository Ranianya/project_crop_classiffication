import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, mapping
from shapely.ops import transform
import pyproj
from pathlib import Path

# =============================
# CONFIG
# =============================

BASE_DIR = Path(r"C:\projects\resneur\projectcrops\project_crop_classiffication")

CDL_PATH = BASE_DIR / "arkansas" / "part1" / "2_data_cdl" / "CDL_2021_05.tif"

OUTPUT_NORTH = BASE_DIR / "arkansas" / "part1" / "4_results" / "1_CDL Extraction_result" / "CDL_Arkansas_North.tif"
OUTPUT_SOUTH = BASE_DIR / "arkansas" / "part1" / "4_results" / "1_CDL Extraction_result" / "CDL_Arkansas_South.tif"

# Create output directory if it doesn't exist
OUTPUT_NORTH.parent.mkdir(parents=True, exist_ok=True)

POINTS = {
    "North": (-90.80, 35.10),
    "South": (-91.50, 34.50)
}

BUFFER_METERS = 20000  # 20 km

# =============================
# SAFE BUFFER IN RASTER CRS
# =============================
def create_buffer_in_raster_crs(lon, lat, src_crs, buffer_m):

    # 1. WGS84 point
    point = Point(lon, lat)

    # 2. Transform to raster CRS
    project_to_raster = pyproj.Transformer.from_crs(
        "EPSG:4326",
        src_crs,
        always_xy=True
    ).transform

    point_raster = transform(project_to_raster, point)

    # 3. Buffer (ONLY valid if CRS is meters)
    buffer_geom = point_raster.buffer(buffer_m)

    return buffer_geom


# =============================
# CLIP FUNCTION
# =============================
def clip_cdl(output_path, lon, lat):

    with rasterio.open(CDL_PATH) as src:

        print(f"\n📂 Processing {output_path.name}")
        print("Raster CRS:", src.crs)
        print("Raster bounds:", src.bounds)

        # Create buffer in raster CRS
        geom = create_buffer_in_raster_crs(lon, lat, src.crs, BUFFER_METERS)
        geojson = [mapping(geom)]

        # ⚠️ Check bounds BEFORE masking (prevents crash)
        raster_bbox = src.bounds
        point_check = geom.bounds

        if (
            point_check[2] < raster_bbox.left or
            point_check[0] > raster_bbox.right or
            point_check[3] < raster_bbox.bottom or
            point_check[1] > raster_bbox.top
        ):
            print("❌ SKIPPED: No overlap with raster!")
            return

        # Clip safely
        try:
            out_image, out_transform = mask(src, geojson, crop=True)

        except ValueError as e:
            print("❌ Mask failed:", e)
            return

        # Save output
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"✅ Saved: {output_path.name}")
        print(f"   ➜ Shape: {out_image.shape}")


# =============================
# MAIN
# =============================
def main():

    print("🚀 START CDL EXTRACTION")

    for name, (lon, lat) in POINTS.items():
        output = BASE_DIR / f"CDL_Arkansas_{name}.tif"
        clip_cdl(output, lon, lat)

    print("\n🎉 DONE")


if __name__ == "__main__":
    main()