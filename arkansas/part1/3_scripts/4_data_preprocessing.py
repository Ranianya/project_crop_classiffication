"""
MCTNet FAST PREPROCESSING - Block Read Version (FIXED)
=======================================================
Changes vs original:
  - block_to_samples() now also returns pixel (row, col) positions
  - sample_from_shard() passes shard transform to block_to_samples
  - save_splits() additionally saves coords_<split>.npy  shape (N, 2)
    with columns [x_epsg5070, y_epsg5070]
  - metadata.json updated to document the coords files
  - coords are in EPSG:5070 (same CRS as everything else)

Why coords are needed:
  Part 2 requires joining environmental covariates (climate, soil,
  topography rasters) to each pixel sample. Without knowing WHERE
  each pixel is geographically, that join is impossible.

Usage for Part 2:
  coords = np.load("coords_train.npy")   # shape (N, 2)  [x, y] EPSG:5070
  # Then sample any raster at those coordinates with rasterio.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import xy as transform_xy
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import json, time, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(r"C:\projects\resneur\projectcrops\project_crop_classiffication")

# Paths to your actual data directories
CDL_PROJ_DIR = BASE_DIR / "arkansas" / "part1" / "4_results" / "1_CDL Extraction_result"
REPROJ_DIR = BASE_DIR / "arkansas" / "part1" / "4_results" / "2_Sentinel-2 Reprojection_result"
OUTPUT_DIR = BASE_DIR / "arkansas" / "part1" / "4_results" / "4_data_preprocessing_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = OUTPUT_DIR / "_cdl_crops"
TMP_DIR.mkdir(exist_ok=True)

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

SAMPLES_PER_REGION = 6_000
BLOCK_ROWS         = 64

# ─────────────────────────────────────────────────────────────────────────────
# PAPER CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NUM_TIMESTEPS      = 36
NUM_BANDS          = 10
BANDS_PER_TS       = 11
TIMESTEPS_IN_TILE  = 37
SPECTRAL_IDX       = list(range(10))

MIN_VALID_TS       = 5
MIN_CLASS_FRAC     = 0.05
TRAINVAL_PER_CLASS = 300
TRAIN_FRAC         = 0.8

CDL_CODES  = {1: "Corn", 2: "Cotton", 5: "Soybeans", 24: "Rice"}
BAND_NAMES = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
DOY        = [10 * i + 5 for i in range(36)]
COLORS     = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628"]
MARKERS    = ["o","s","^","D","v","P"]

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# CDL CROP TO SHARD GRID  (tiny array, cached)
# ─────────────────────────────────────────────────────────────────────────────

def crop_cdl_to_shard(cdl_path: Path, shard_path: Path) -> np.ndarray:
    """
    Reproject CDL to match reprojected shard's grid.
    Both are in EPSG:5070 → just a crop + nearest resample.
    Result is a (H, W) uint8 array, ~few MB. Cached on disk.
    """
    cache = TMP_DIR / f"{shard_path.stem}_cdl.tif"
    if cache.exists():
        with rasterio.open(cache) as f:
            return f.read(1)

    with rasterio.open(shard_path) as ref:
        dst_tf   = ref.transform
        dst_h, dst_w = ref.shape

    cdl_arr = np.zeros((dst_h, dst_w), dtype=np.uint8)
    with rasterio.open(cdl_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=cdl_arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_tf, dst_crs=CDL_CRS,
            resampling=Resampling.nearest,
        )

    # Cache
    with rasterio.open(shard_path) as ref:
        profile = ref.profile.copy()
    profile.update({"count": 1, "dtype": "uint8", "nodata": 0, "compress": "lzw"})
    with rasterio.open(cache, "w", **profile) as dst:
        dst.write(cdl_arr, 1)

    return cdl_arr


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION FROM A BLOCK  ← FIXED: now saves pixel coordinates
# ─────────────────────────────────────────────────────────────────────────────

def block_to_samples(block: np.ndarray,
                     cdl_block: np.ndarray,
                     row_offset: int,
                     label_map: dict,
                     shard_transform) -> list:           # ← NEW PARAMETER
    """
    block           : (407, BLOCK_ROWS, W) float32
    cdl_block       : (BLOCK_ROWS, W)      uint8
    row_offset      : absolute row index of block start in the shard grid
    label_map       : CDL code → class name
    shard_transform : rasterio Affine transform of the reprojected shard
                      used to convert (row, col) → (x, y) in EPSG:5070

    Returns list of sample dicts, each with keys:
        X     : (36, 10) float32  spectral time-series
        mask  : (36,)    float32  1=valid timestep, 0=missing
        label : str               crop class name
        x     : float             EPSG:5070 easting  of the pixel centre
        y     : float             EPSG:5070 northing of the pixel centre
    """
    _, nrows, ncols = block.shape

    # Flatten spatial dims: (407, nrows*ncols) → transpose → (N, 407)
    flat     = block.reshape(407, -1).T          # (N, 407)
    cdl_flat = cdl_block.reshape(-1)             # (N,)

    # Keep only CDL-labelled pixels
    valid_idx = np.where(cdl_flat > 0)[0]
    if len(valid_idx) == 0:
        return []

    flat_valid = flat[valid_idx]                 # (Nv, 407)

    # ── spectral reshape ────────────────────────────────────────────────────
    ts_valid = flat_valid.reshape(-1, TIMESTEPS_IN_TILE,
                                  BANDS_PER_TS)[:, :NUM_TIMESTEPS, :]   # (Nv,36,11)
    spectral = ts_valid[:, :, SPECTRAL_IDX].astype(np.float32)          # (Nv,36,10)

    # Missing mask: timestep is missing when ALL 10 bands == 0
    all_zero = (spectral == 0).all(axis=2)       # (Nv,36) True=missing
    mask_arr = (~all_zero).astype(np.float32)    # (Nv,36) 1=valid
    spectral[all_zero] = 0.0

    # Drop pixels with fewer than MIN_VALID_TS valid timesteps
    keep = mask_arr.sum(axis=1) >= MIN_VALID_TS
    if keep.sum() == 0:
        return []

    spectral  = spectral[keep]
    mask_arr  = mask_arr[keep]
    cdl_keep  = cdl_flat[valid_idx[keep]]

    # ── pixel coordinates (NEW) ─────────────────────────────────────────────
    # Recover (row, col) in shard coordinates for every kept pixel
    kept_flat_idx = valid_idx[keep]              # indices into the flattened block
    rows_in_block = kept_flat_idx // ncols       # row within this block
    cols_in_block = kept_flat_idx  % ncols       # col within this block

    abs_rows = rows_in_block + row_offset        # absolute row in the full shard
    abs_cols = cols_in_block                     # col is unchanged

    # rasterio.transform.xy returns (xs, ys) as lists
    xs, ys = transform_xy(shard_transform,
                          abs_rows.tolist(),
                          abs_cols.tolist())     # EPSG:5070 coordinates

    # ── build sample dicts ──────────────────────────────────────────────────
    samples = []
    for j in range(len(cdl_keep)):
        code  = int(cdl_keep[j])
        label = label_map.get(code, "Others")
        samples.append({
            "X"    : spectral[j],      # (36, 10)
            "mask" : mask_arr[j],      # (36,)
            "label": label,
            "x"    : float(xs[j]),     # EPSG:5070 easting  ← NEW
            "y"    : float(ys[j]),     # EPSG:5070 northing ← NEW
        })

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE FROM ONE SHARD  (block reads)  ← FIXED: passes transform
# ─────────────────────────────────────────────────────────────────────────────

def sample_from_shard(shard_path: Path,
                      cdl_path: Path,
                      n_samples: int) -> list:
    name = shard_path.stem
    print(f"\n    Shard: {name}")

    if not shard_path.exists():
        print(f"      NOT FOUND — skipping")
        return []

    cdl = crop_cdl_to_shard(cdl_path, shard_path)
    H, W = cdl.shape
    total_valid = (cdl > 0).sum()
    print(f"      Grid: {H}×{W}  |  CDL valid: {total_valid:,}  "
          f"|  requesting: {n_samples:,}")

    if total_valid == 0:
        print(f"      WARNING: 0 CDL valid pixels — no overlap")
        return []

    all_candidate_samples = []
    t0 = time.time()

    with rasterio.open(shard_path) as src:
        shard_transform = src.transform            # ← read transform once here

        for r_start in range(0, H, BLOCK_ROWS):
            r_end   = min(r_start + BLOCK_ROWS, H)
            block_h = r_end - r_start

            win       = rasterio.windows.Window(0, r_start, W, block_h)
            block     = src.read(window=win).astype(np.float32)
            cdl_block = cdl[r_start:r_end, :]

            candidates = block_to_samples(
                block, cdl_block, r_start, CDL_CODES,
                shard_transform                    # ← pass transform
            )
            all_candidate_samples.extend(candidates)

            elapsed = time.time() - t0
            rate    = len(all_candidate_samples) / elapsed if elapsed > 0 else 0
            print(f"      rows {r_start:4d}-{r_end:4d}  "
                  f"candidates={len(all_candidate_samples):,}  "
                  f"{rate:.0f}px/s   ",
                  end="\r")

            if len(all_candidate_samples) >= n_samples * 3:
                break

    elapsed = time.time() - t0
    print(f"      Collected {len(all_candidate_samples):,} candidates "
          f"in {elapsed:.1f}s  ({len(all_candidate_samples)/max(elapsed,1):.0f}px/s)   ")

    if len(all_candidate_samples) > n_samples:
        idx = np.random.choice(len(all_candidate_samples), n_samples, replace=False)
        all_candidate_samples = [all_candidate_samples[i] for i in idx]

    return all_candidate_samples


# ─────────────────────────────────────────────────────────────────────────────
# REGION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_region(region_name: str, cfg: dict, n_samples: int) -> list:
    print(f"\n{'─'*60}")
    print(f"  Region: {region_name}")
    print(f"{'─'*60}")

    existing = [p for p in cfg["reproj_shards"] if p.exists()]
    if not existing:
        print(f"  ERROR: no reprojected shards found in {REPROJ_DIR}")
        print(f"  Run reproject_sentinel.py first!")
        return []

    n_per_shard = int(np.ceil(n_samples / len(existing)))
    print(f"  {len(existing)} shards  |  ~{n_per_shard:,}/shard  "
          f"|  target={n_samples:,}")

    all_samples = []
    for shard_path in existing:
        all_samples.extend(
            sample_from_shard(shard_path, cfg["cdl"], n_per_shard)
        )

    if not all_samples:
        return []

    # Merge classes that represent < 5% of samples into "Others"
    total    = len(all_samples)
    by_class = defaultdict(int)
    for s in all_samples:
        by_class[s["label"]] += 1

    print(f"\n  Class distribution ({total:,} samples):")
    to_merge = set()
    for cname in sorted(by_class):
        cnt  = by_class[cname]
        frac = cnt / total
        flag = "  ← merge" if frac < MIN_CLASS_FRAC and cname != "Others" else ""
        print(f"    {cname:15s}: {cnt:6,}  ({frac*100:.1f}%){flag}")
        if frac < MIN_CLASS_FRAC and cname != "Others":
            to_merge.add(cname)

    if to_merge:
        print(f"  Merging {to_merge} → 'Others'")
        for s in all_samples:
            if s["label"] in to_merge:
                s["label"] = "Others"

    if len(all_samples) > n_samples:
        idx = np.random.choice(len(all_samples), n_samples, replace=False)
        all_samples = [all_samples[i] for i in idx]

    print(f"  Final: {len(all_samples):,} samples")
    return all_samples


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_paper_style(samples: list) -> tuple:
    by_class = defaultdict(list)
    for s in samples:
        by_class[s["label"]].append(s)

    train_all, val_all, test_all = [], [], []
    print(f"\n  {'Class':15s} {'Total':>7} {'Train':>6} {'Val':>5} {'Test':>7}")
    print(f"  {'─'*48}")

    for cname in sorted(by_class):
        items = by_class[cname].copy()
        np.random.shuffle(items)
        n_total    = len(items)
        n_trainval = min(TRAINVAL_PER_CLASS, n_total)
        n_train    = int(TRAIN_FRAC * n_trainval)
        n_val      = n_trainval - n_train

        train_all.extend(items[:n_train])
        val_all.extend(items[n_train:n_trainval])
        test_all.extend(items[n_trainval:])
        print(f"  {cname:15s} {n_total:>7,} {n_train:>6,} "
              f"{n_val:>5,} {n_total-n_trainval:>7,}")

    print(f"\n  Totals  train={len(train_all):,}  "
          f"val={len(val_all):,}  test={len(test_all):,}")
    for lst in [train_all, val_all, test_all]:
        np.random.shuffle(lst)
    return train_all, val_all, test_all


# ─────────────────────────────────────────────────────────────────────────────
# SAVE  ← FIXED: now also saves coords_<split>.npy
# ─────────────────────────────────────────────────────────────────────────────

def save_splits(train, val, test, out_dir: Path) -> LabelEncoder:
    le = LabelEncoder()
    le.fit([s["label"] for s in train])
    known = set(le.classes_)
    stats = {}

    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        if not data:
            continue

        X      = np.stack([s["X"]    for s in data])          # (N, 36, 10)
        mask   = np.stack([s["mask"] for s in data])          # (N, 36)
        lbls   = np.array(["Others" if s["label"] not in known
                            else s["label"] for s in data])
        y      = le.transform(lbls)                           # (N,)

        # ── NEW: geographic coordinates ────────────────────────────────────
        # Shape (N, 2)  columns: [x_epsg5070, y_epsg5070]
        # Your friend samples covariate rasters at these positions in Part 2.
        coords = np.array([[s["x"], s["y"]] for s in data],
                          dtype=np.float64)                   # (N, 2)

        np.save(out_dir / f"X_{split_name}.npy",      X)
        np.save(out_dir / f"mask_{split_name}.npy",   mask)
        np.save(out_dir / f"y_{split_name}.npy",      y)
        np.save(out_dir / f"coords_{split_name}.npy", coords) # ← NEW

        dist = {c: int((y == i).sum()) for i, c in enumerate(le.classes_)}
        stats[split_name] = {"n_samples": len(data), "class_distribution": dist}

        print(f"\n  [{split_name}]  X={X.shape}  mask={mask.shape}  "
              f"y={y.shape}  coords={coords.shape}")
        for c, cnt in dist.items():
            print(f"    {c:15s}: {cnt:,}")

    meta = {
        "classes"          : le.classes_.tolist(),
        "num_classes"      : len(le.classes_),
        "num_timesteps"    : NUM_TIMESTEPS,
        "num_bands"        : NUM_BANDS,
        "input_shape"      : [NUM_TIMESTEPS, NUM_BANDS],
        "mask_convention"  : "1=valid, 0=missing",
        "normalization"    : "none (raw L2A reflectance)",
        "target_crs"       : "EPSG:5070",
        # ── NEW ──────────────────────────────────────────────────────────
        "coords_files"     : "coords_train/val/test.npy",
        "coords_columns"   : ["x_epsg5070", "y_epsg5070"],
        "coords_dtype"     : "float64",
        "coords_usage"     : (
            "Use these coordinates in Part 2 to sample climate/soil/"
            "topography rasters. Example: rasterio.sample.sample_gen("
            "raster, coords)"
        ),
        # ─────────────────────────────────────────────────────────────────
        "splits"           : stats,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
    print(f"\n  Saved: metadata.json")
    return le


# ─────────────────────────────────────────────────────────────────────────────
# EDA PLOTS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ndvi_timeseries(all_samples, le, out_dir, title="Arkansas"):
    X    = np.stack([s["X"]    for s in all_samples])
    mask = np.stack([s["mask"] for s in all_samples])
    y    = le.transform([
        "Others" if s["label"] not in set(le.classes_) else s["label"]
        for s in all_samples
    ])
    nir   = X[:, :, 6].astype(np.float32)
    red   = X[:, :, 2].astype(np.float32)
    denom = nir + red
    ndvi  = np.where(denom > 0, (nir - red) / (denom + 1e-8), np.nan)
    ndvi[mask == 0] = np.nan

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, cname in enumerate(le.classes_):
        cls_ndvi  = ndvi[y == i]
        mean_ndvi = np.nanmean(cls_ndvi, axis=0)
        std_ndvi  = np.nanstd(cls_ndvi,  axis=0)
        valid_t   = ~np.isnan(mean_ndvi)
        xv        = np.array(DOY)[valid_t]
        ax.plot(xv, mean_ndvi[valid_t], label=cname,
                color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                markersize=4, linewidth=2)
        ax.fill_between(xv,
                        (mean_ndvi - std_ndvi)[valid_t],
                        (mean_ndvi + std_ndvi)[valid_t],
                        alpha=0.15, color=COLORS[i % len(COLORS)])

    ax.set_xlabel("Day of Year", fontsize=12)
    ax.set_ylabel("NDVI", fontsize=12)
    ax.set_title(f"NDVI Time-Series by Crop Type — {title}", fontsize=14)
    ax.set_xlim(0, 365); ax.set_ylim(0, 1)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"ndvi_timeseries_{title}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  NDVI time-series      → {p.name}")


def plot_class_distribution(all_samples, le, out_dir, title="Arkansas"):
    by_class = defaultdict(int)
    for s in all_samples:
        by_class[s["label"]] += 1
    classes = sorted(by_class.keys())
    counts  = [by_class[c] for c in classes]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(classes, counts, color=COLORS[:len(classes)],
                  edgecolor="black", alpha=0.85)
    ax.bar_label(bars, fmt="%d", fontsize=9)
    ax.set_xlabel("Crop Class", fontsize=12)
    ax.set_ylabel("Samples", fontsize=12)
    ax.set_title(f"Class Distribution — {title}", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"class_distribution_{title}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Class distribution    → {p.name}")


def plot_missing_rate(all_samples, out_dir, title="Arkansas"):
    masks      = np.stack([s["mask"] for s in all_samples])
    miss_rates = 1.0 - masks.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(miss_rates * 100, bins=36, color="#377eb8",
            edgecolor="white", linewidth=0.5)
    ax.axvline(miss_rates.mean() * 100, color="red", linestyle="--",
               label=f"Mean = {miss_rates.mean()*100:.1f}%")
    ax.set_xlabel("Missing Rate per Pixel (%)", fontsize=12)
    ax.set_ylabel("Samples", fontsize=12)
    ax.set_title(f"Missing Timestep Distribution — {title}", fontsize=14)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"missing_rate_{title}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Missing rate          → {p.name}")


def plot_spectral_profiles(all_samples, le, out_dir,
                           timestep=17, title="Arkansas"):
    X    = np.stack([s["X"]    for s in all_samples])
    mask = np.stack([s["mask"] for s in all_samples])
    y    = le.transform([
        "Others" if s["label"] not in set(le.classes_) else s["label"]
        for s in all_samples
    ])

    if mask.ndim == 3:
        mask = mask.squeeze()

    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(BAND_NAMES))
    for i, cname in enumerate(le.classes_):
        timestep_int = int(timestep)
        mask_t = mask[:, timestep_int].astype(bool)
        valid  = X[(y == i) & mask_t, timestep_int, :]
        if len(valid) == 0:
            continue
        ax.plot(x_pos, valid.mean(axis=0), label=cname,
                color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                markersize=5, linewidth=1.5)
    ax.set_xticks(x_pos); ax.set_xticklabels(BAND_NAMES, fontsize=10)
    ax.set_xlabel("Spectral Band", fontsize=12)
    ax.set_ylabel("Mean Reflectance", fontsize=12)
    ax.set_title(f"Spectral Profiles at DOY≈{DOY[timestep_int]} — {title}", fontsize=14)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"spectral_profiles_{title}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Spectral profiles     → {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("MCTNet FAST PREPROCESSING  —  Block Read Version (FIXED)")
    print("=" * 70)
    print(f"Block size: {BLOCK_ROWS} rows  |  Expected speed: ~10,000-50,000 px/s")
    print(f"Reprojected shards dir: {REPROJ_DIR}")

    if not REPROJ_DIR.exists():
        print(f"\nERROR: {REPROJ_DIR} not found.")
        print("Run reproject_sentinel.py first, then rerun this script.")
        return

    all_samples = []
    for region_name, cfg in REGIONS.items():
        all_samples.extend(
            process_region(region_name, cfg, SAMPLES_PER_REGION)
        )

    total = len(all_samples)
    print(f"\n{'─'*60}")
    print(f"Grand total: {total:,} samples")
    if total == 0:
        print("No samples. Check that reprojected shards overlap with CDL.")
        return

    print(f"\n>>> Splitting (paper style)")
    train, val, test = split_paper_style(all_samples)

    print(f"\n>>> Saving to {OUTPUT_DIR}")
    le = save_splits(train, val, test, OUTPUT_DIR)

    print(f"\n>>> EDA plots")
    plot_ndvi_timeseries(all_samples, le, OUTPUT_DIR, "Arkansas")
    plot_class_distribution(all_samples, le, OUTPUT_DIR, "Arkansas")
    plot_missing_rate(all_samples, OUTPUT_DIR, "Arkansas")
    plot_spectral_profiles(all_samples, le, OUTPUT_DIR, 17, "Arkansas")

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"DONE  ({elapsed:.1f} min)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Data:   X / mask / y / coords  _train/_val/_test.npy  +  metadata.json")
    print(f"Plots:  ndvi_timeseries.png  class_distribution.png")
    print(f"        missing_rate.png     spectral_profiles.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# HOW USES coords_*.npy IN PART 2
# ─────────────────────────────────────────────────────────────────────────────
#
#   import numpy as np
#   import rasterio
#   from rasterio.sample import sample_gen
#
#   coords_train = np.load("coords_train.npy")   # (N, 2)  x, y  EPSG:5070
#
#   # Example: sample a DEM (topography) raster at every pixel location
#   with rasterio.open("DEM_EPSG5070.tif") as dem:
#       elev_vals = np.array(list(sample_gen(dem, coords_train)))  # (N, 1)
#
#   # Example: sample a soil raster
#   with rasterio.open("soil_EPSG5070.tif") as soil:
#       soil_vals = np.array(list(sample_gen(soil, coords_train)))  # (N, n_bands)
#
#   # Then concatenate with X_train for the covariate-enhanced model:
#   # X_train shape (N, 36, 10)  +  extra_train shape (N, n_covariates)
#   # Feed both into your extended model.
#
# NOTE: all covariate rasters must be reprojected to EPSG:5070 first,
# same as the Sentinel-2 shards.  Use the same reproject_sentinel.py
# approach (calculate_default_transform + rasterio.warp.reproject).