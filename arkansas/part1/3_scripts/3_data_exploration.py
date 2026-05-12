"""
MCTNet IMPROVED DATA EXPLORATION (FINAL FIXED VERSION)
=====================================================

Paper-style EDA directly from:

- Sentinel-2 reprojected shards
- CDL crop rasters

This version fixes:
    - broken NDVI curves
    - missing temporal separability values
    - empty correlation matrix
    - noisy temporal signals

OUTPUT:
    processed/eda_raw/

Generated plots:
    - class_distribution.png
    - missing_rate.png
    - ndvi_temporal_profiles.png
    - spectral_profiles.png
    - temporal_separability.png
    - band_correlation.png
"""

import numpy as np
import rasterio

from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS

from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\projects\resneur\projectcrops\project_crop_classiffication")

# Paths to your actual data directories
CDL_PROJ_DIR = BASE_DIR / "arkansas" / "part1" / "4_results" / "1_CDL Extraction_result"
REPROJ_DIR = BASE_DIR / "arkansas" / "part1" / "4_results" / "2_Sentinel-2 Reprojection_result"
OUTPUT_DIR = BASE_DIR / "arkansas" / "part1" / "4_results" / "3_data_exploration_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CDL_CRS = CRS.from_epsg(5070)

REGIONS = {
    "Arkansas_North": {
        "cdl": CDL_PROJ_DIR / "CDL_Arkansas_North.tif",
        "reproj_shards": [
            REPROJ_DIR / "MCTNet_14GB_Arkansas_North-0000000000-0000000000_5070.tif",
            REPROJ_DIR / "MCTNet_14GB_Arkansas_North-0000000000-0000001792_5070.tif",
            REPROJ_DIR / "MCTNet_14GB_Arkansas_North-0000001792-0000000000_5070.tif",
            REPROJ_DIR / "MCTNet_14GB_Arkansas_North-0000001792-0000001792_5070.tif",
        ],
    },
    "Arkansas_South": {
        "cdl": CDL_PROJ_DIR / "CDL_Arkansas_South.tif",
        "reproj_shards": [
            REPROJ_DIR / "MCTNet_14GB_Arkansas_South-0000000000-0000000000_5070.tif",
            REPROJ_DIR / "MCTNet_14GB_Arkansas_South-0000000000-0000001792_5070.tif",
            REPROJ_DIR / "MCTNet_14GB_Arkansas_South-0000001792-0000000000_5070.tif",
            REPROJ_DIR / "MCTNet_14GB_Arkansas_South-0000001792-0000001792_5070.tif",
        ],
    },
}

# ============================================================
# PAPER PARAMETERS
# ============================================================

NUM_TIMESTEPS = 36
NUM_BANDS = 10

TIMESTEPS_IN_TILE = 37
BANDS_PER_TS = 11

SPECTRAL_IDX = list(range(10))

BLOCK_ROWS = 64

MAX_SAMPLES = 12000

MIN_VALID_TS = 5

# ============================================================
# LABELS
# ============================================================

CDL_CODES = {

    1: "Corn",
    2: "Cotton",
    5: "Soybeans",
    24: "Rice",
}

# ============================================================
# VISUAL CONFIG
# ============================================================

DOY = [10 * i + 5 for i in range(36)]

BAND_NAMES = [

    "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B11", "B12"
]

COLORS = [

    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628"
]

np.random.seed(42)

# ============================================================
# SMOOTH FUNCTION
# ============================================================

def smooth_curve(arr, window=2):

    out = np.copy(arr)

    for i in range(len(arr)):

        start = max(0, i - window)
        end = min(len(arr), i + window + 1)

        out[i] = np.nanmean(arr[start:end])

    return out

# ============================================================
# REPROJECT CDL TO SHARD GRID
# ============================================================

def crop_cdl_to_shard(cdl_path, shard_path):

    with rasterio.open(shard_path) as ref:

        H = ref.height
        W = ref.width

        dst_transform = ref.transform
        dst_crs = ref.crs

    cdl_out = np.zeros((H, W), dtype=np.uint8)

    with rasterio.open(cdl_path) as src:

        reproject(
            source=rasterio.band(src, 1),
            destination=cdl_out,

            src_transform=src.transform,
            src_crs=src.crs,

            dst_transform=dst_transform,
            dst_crs=dst_crs,

            resampling=Resampling.nearest,
        )

    return cdl_out

# ============================================================
# SAMPLE EXTRACTION
# ============================================================

def extract_samples(region_name, cfg):

    print("\n" + "=" * 70)
    print(f"REGION: {region_name}")
    print("=" * 70)

    samples = []

    for shard_path in cfg["reproj_shards"]:

        if not shard_path.exists():
            continue

        print(f"\nProcessing:")
        print(shard_path.name)

        cdl = crop_cdl_to_shard(
            cfg["cdl"],
            shard_path
        )

        with rasterio.open(shard_path) as src:

            H = src.height
            W = src.width

            for row_start in range(0, H, BLOCK_ROWS):

                row_end = min(row_start + BLOCK_ROWS, H)

                block_h = row_end - row_start

                window = Window(
                    0,
                    row_start,
                    W,
                    block_h
                )

                block = src.read(window=window)

                cdl_block = cdl[row_start:row_end]

                block = block.astype(np.float32)

                # ------------------------------------------------
                # reshape
                # ------------------------------------------------

                flat = block.reshape(407, -1).T

                cdl_flat = cdl_block.reshape(-1)

                valid_idx = np.where(cdl_flat > 0)[0]

                if len(valid_idx) == 0:
                    continue

                flat = flat[valid_idx]
                labels = cdl_flat[valid_idx]

                ts = flat.reshape(
                    -1,
                    TIMESTEPS_IN_TILE,
                    BANDS_PER_TS
                )

                ts = ts[:, :NUM_TIMESTEPS, :]

                spectral = ts[:, :, SPECTRAL_IDX]

                # ------------------------------------------------
                # mask
                # ------------------------------------------------

                all_zero = (spectral == 0).all(axis=2)

                mask = (~all_zero).astype(np.float32)

                valid_ts = mask.sum(axis=1)

                keep = valid_ts >= MIN_VALID_TS

                spectral = spectral[keep]
                mask = mask[keep]
                labels = labels[keep]

                # ------------------------------------------------
                # save
                # ------------------------------------------------

                for i in range(len(labels)):

                    label = CDL_CODES.get(
                        int(labels[i]),
                        "Others"
                    )

                    samples.append({

                        "X": spectral[i],
                        "mask": mask[i],
                        "label": label,
                    })

                print(
                    f"rows {row_start:4d}-{row_end:4d} "
                    f"| samples={len(samples):,}",
                    end="\r"
                )

                if len(samples) >= MAX_SAMPLES:
                    break

        if len(samples) >= MAX_SAMPLES:
            break

    print(f"\nCollected {len(samples):,} samples")

    return samples

# ============================================================
# LOAD REGIONS
# ============================================================

all_samples = []

for region_name, cfg in REGIONS.items():

    region_samples = extract_samples(
        region_name,
        cfg
    )

    all_samples.extend(region_samples)

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Total samples: {len(all_samples):,}")

# ============================================================
# BUILD ARRAYS
# ============================================================

X = np.stack([
    s["X"]
    for s in all_samples
])

mask = np.stack([
    s["mask"]
    for s in all_samples
])

labels = [
    s["label"]
    for s in all_samples
]

classes = sorted(list(set(labels)))

class_to_id = {
    c: i for i, c in enumerate(classes)
}

y = np.array([
    class_to_id[c]
    for c in labels
])

print("X shape:", X.shape)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nCLASS DISTRIBUTION")

counter = Counter(labels)

for cls, count in sorted(counter.items()):

    pct = 100 * count / len(labels)

    print(f"{cls:15s} {count:6d} ({pct:.2f}%)")

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(
    counter.keys(),
    counter.values(),
    color=COLORS[:len(counter)],
    edgecolor="black",
    alpha=0.85
)

ax.bar_label(bars)

ax.set_title("Crop Class Distribution")
ax.set_xlabel("Crop Type")
ax.set_ylabel("Samples")

ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "class_distribution.png",
    dpi=300
)

plt.close(fig)

# ============================================================
# MISSING DATA
# ============================================================

print("\nMISSING DATA ANALYSIS")

valid_ts = mask.sum(axis=1)

missing_rate = 1 - (valid_ts / NUM_TIMESTEPS)

print(f"Average missing rate: {missing_rate.mean()*100:.2f}%")

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    missing_rate * 100,
    bins=36,
    color="#377eb8",
    edgecolor="white"
)

ax.set_title("Missing Timestep Distribution")
ax.set_xlabel("Missing Rate (%)")
ax.set_ylabel("Samples")

ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "missing_rate.png",
    dpi=300
)

plt.close(fig)

# ============================================================
# NDVI ANALYSIS
# ============================================================

print("\nNDVI ANALYSIS")

nir = X[:, :, 6]
red = X[:, :, 2]

denom = nir + red

ndvi = np.where(
    denom > 0,
    (nir - red) / (denom + 1e-8),
    np.nan
)

ndvi[mask == 0] = np.nan

fig, ax = plt.subplots(figsize=(14, 7))

for i, cls in enumerate(classes):

    cls_ndvi = ndvi[y == i]

    if len(cls_ndvi) < 10:
        continue

    mean_curve = np.nanmean(
        cls_ndvi,
        axis=0
    )

    std_curve = np.nanstd(
        cls_ndvi,
        axis=0
    )

    # --------------------------------------------
    # interpolate NaNs
    # --------------------------------------------

    valid = ~np.isnan(mean_curve)

    if valid.sum() < 2:
        continue

    mean_curve = np.interp(
        np.arange(len(mean_curve)),
        np.where(valid)[0],
        mean_curve[valid]
    )

    std_curve = np.interp(
        np.arange(len(std_curve)),
        np.where(valid)[0],
        std_curve[valid]
    )

    # --------------------------------------------
    # smooth
    # --------------------------------------------

    mean_curve = smooth_curve(
        mean_curve,
        window=2
    )

    std_curve = smooth_curve(
        std_curve,
        window=2
    )

    ax.plot(
        DOY,
        mean_curve,
        label=cls,
        linewidth=3,
        marker="o",
        markersize=4,
        color=COLORS[i % len(COLORS)]
    )

    ax.fill_between(
        DOY,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.15,
        color=COLORS[i % len(COLORS)]
    )

ax.set_title(
    "Smoothed NDVI Temporal Profiles",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Day of Year")
ax.set_ylabel("NDVI")

ax.set_xlim(0, 365)
ax.set_ylim(0, 1)

ax.grid(alpha=0.3)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "ndvi_temporal_profiles.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# SPECTRAL PROFILE ANALYSIS
# ============================================================

print("\nSPECTRAL PROFILE ANALYSIS")

timestep = 17

fig, ax = plt.subplots(figsize=(10, 5))

x_pos = np.arange(len(BAND_NAMES))

for i, cls in enumerate(classes):

    valid_pixels = X[y == i, timestep]

    if len(valid_pixels) == 0:
        continue

    mean_spectrum = np.nanmean(
        valid_pixels,
        axis=0
    )

    ax.plot(
        x_pos,
        mean_spectrum,
        label=cls,
        linewidth=2,
        marker="o",
        color=COLORS[i % len(COLORS)]
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(BAND_NAMES)

ax.set_title(
    f"Spectral Profiles at DOY ≈ {DOY[timestep]}"
)

ax.set_xlabel("Band")
ax.set_ylabel("Reflectance")

ax.grid(alpha=0.3)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "spectral_profiles.png",
    dpi=300
)

plt.close(fig)

# ============================================================
# TEMPORAL SEPARABILITY
# ============================================================

print("\nTEMPORAL SEPARABILITY")

mean_ndvi_per_class = []

for i in range(len(classes)):

    cls_ndvi = ndvi[y == i]

    if len(cls_ndvi) == 0:
        continue

    mean_curve = np.nanmean(
        cls_ndvi,
        axis=0
    )

    valid = ~np.isnan(mean_curve)

    if valid.sum() < 2:
        continue

    mean_curve = np.interp(
        np.arange(len(mean_curve)),
        np.where(valid)[0],
        mean_curve[valid]
    )

    mean_curve = smooth_curve(
        mean_curve,
        window=2
    )

    mean_ndvi_per_class.append(mean_curve)

mean_ndvi_per_class = np.array(
    mean_ndvi_per_class
)

temporal_variance = np.std(
    mean_ndvi_per_class,
    axis=0
)

temporal_variance = smooth_curve(
    temporal_variance,
    window=2
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    DOY,
    temporal_variance,
    linewidth=3
)

ax.fill_between(
    DOY,
    temporal_variance,
    alpha=0.2
)

ax.set_title(
    "Temporal Crop Separability",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Day of Year")
ax.set_ylabel("Inter-Class NDVI Variance")

ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "temporal_separability.png",
    dpi=300
)

plt.close(fig)

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\nCORRELATION ANALYSIS")

masked_X = np.where(
    mask[:, :, None] == 1,
    X,
    np.nan
)

mean_features = np.nanmean(
    masked_X,
    axis=1
)

valid_rows = ~np.isnan(
    mean_features
).any(axis=1)

mean_features = mean_features[
    valid_rows
]

print(
    "Valid samples for correlation:",
    len(mean_features)
)

corr = np.corrcoef(
    mean_features.T
)

corr = np.nan_to_num(corr)

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    corr,
    xticklabels=BAND_NAMES,
    yticklabels=BAND_NAMES,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    ax=ax
)

ax.set_title(
    "Spectral Band Correlation",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "band_correlation.png",
    dpi=300
)

plt.close(fig)

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("EDA FINISHED")
print("=" * 70)

print(f"Outputs saved to:\n{OUTPUT_DIR}")