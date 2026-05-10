# ============================================================
# ARKANSAS CROP CLASSIFICATION PROJECT
# PART 2 — DATA PREPARATION PIPELINE
# FINAL CORRECT VERSION
# ============================================================

# This script:
# 1. Loads Sentinel-2 temporal features
# 2. Loads environmental covariates
# 3. Cleans covariates
# 4. Normalizes covariates
# 5. Aligns sample counts
# 6. Expands covariates across timesteps
# 7. Builds ablation datasets
# 8. Saves datasets for training

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# 2. DEFINE INPUT AND OUTPUT FOLDERS
# ============================================================

# Folder structure:
#
# project/
# ├── inputs/
# │   ├── X_train (2).npy
# │   ├── X_val (2).npy
# │   ├── X_test (2).npy
# │   ├── y_train (2).npy
# │   ├── y_val (2).npy
# │   ├── y_test (2).npy
# │   ├── train_covariates.csv
# │   ├── val_covariates.csv
# │   └── test_covariates.csv
# │
# └── outputs/
#     ├── datasets/
#     └── labels/

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

DATASET_DIR = os.path.join(OUTPUT_DIR, "datasets")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")

# Create output folders if they do not exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD SENTINEL-2 FEATURES
# ============================================================

# Temporal Sentinel-2 features
X_train_s2 = np.load(os.path.join(INPUT_DIR, "X_train (2).npy"))
X_val_s2   = np.load(os.path.join(INPUT_DIR, "X_val (2).npy"))
X_test_s2  = np.load(os.path.join(INPUT_DIR, "X_test (2).npy"))

# Labels
y_train = np.load(os.path.join(INPUT_DIR, "y_train (2).npy"))
y_val   = np.load(os.path.join(INPUT_DIR, "y_val (2).npy"))
y_test  = np.load(os.path.join(INPUT_DIR, "y_test (2).npy"))


# ============================================================
# 4. CHECK SHAPES
# ============================================================

print("\n=== SENTINEL-2 FEATURES ===")

print("X_train_s2:", X_train_s2.shape)
print("X_val_s2:", X_val_s2.shape)
print("X_test_s2:", X_test_s2.shape)

print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)


# ============================================================
# 5. LOAD ENVIRONMENTAL COVARIATES
# ============================================================

train_cov = pd.read_csv(os.path.join(INPUT_DIR, "train_covariates.csv"))
val_cov   = pd.read_csv(os.path.join(INPUT_DIR, "val_covariates.csv"))
test_cov  = pd.read_csv(os.path.join(INPUT_DIR, "test_covariates.csv"))


# ============================================================
# 6. INSPECT COVARIATES
# ============================================================

print("\n=== RAW COVARIATES ===")

print(train_cov.head())

print("\nColumns:")
print(train_cov.columns)

print("\nShape:")
print(train_cov.shape)


# ============================================================
# 7. REMOVE GEE EXTRA COLUMNS
# ============================================================

drop_cols = [
    "system:index",
    ".geo"
]

train_cov = train_cov.drop(
    columns=[c for c in drop_cols if c in train_cov.columns],
    errors="ignore"
)

val_cov = val_cov.drop(
    columns=[c for c in drop_cols if c in val_cov.columns],
    errors="ignore"
)

test_cov = test_cov.drop(
    columns=[c for c in drop_cols if c in test_cov.columns],
    errors="ignore"
)


# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================

train_cov = train_cov.fillna(train_cov.mean())
val_cov   = val_cov.fillna(val_cov.mean())
test_cov  = test_cov.fillna(test_cov.mean())


# ============================================================
# 9. CONVERT TO NUMPY
# ============================================================

X_train_cov = train_cov.values
X_val_cov   = val_cov.values
X_test_cov  = test_cov.values


# ============================================================
# 10. CHECK COVARIATE SHAPES
# ============================================================

print("\n=== COVARIATE SHAPES ===")

print("X_train_cov:", X_train_cov.shape)
print("X_val_cov:", X_val_cov.shape)
print("X_test_cov:", X_test_cov.shape)


# ============================================================
# 11. NORMALIZE COVARIATES
# ============================================================

scaler = StandardScaler()

X_train_cov = scaler.fit_transform(X_train_cov)
X_val_cov   = scaler.transform(X_val_cov)
X_test_cov  = scaler.transform(X_test_cov)


# ============================================================
# 12. ALIGN SAMPLE COUNTS
# ============================================================

train_size = min(len(X_train_s2), len(X_train_cov))
val_size   = min(len(X_val_s2), len(X_val_cov))
test_size  = min(len(X_test_s2), len(X_test_cov))


# ------------------------------------------------------------
# Trim Sentinel data
# ------------------------------------------------------------

X_train_s2 = X_train_s2[:train_size]
X_val_s2   = X_val_s2[:val_size]
X_test_s2  = X_test_s2[:test_size]


# ------------------------------------------------------------
# Trim covariates
# ------------------------------------------------------------

X_train_cov = X_train_cov[:train_size]
X_val_cov   = X_val_cov[:val_size]
X_test_cov  = X_test_cov[:test_size]


# ------------------------------------------------------------
# Trim labels
# ------------------------------------------------------------

y_train = y_train[:train_size]
y_val   = y_val[:val_size]
y_test  = y_test[:test_size]


print("\n=== AFTER ALIGNMENT ===")

print("Train:", X_train_s2.shape, X_train_cov.shape)
print("Val  :", X_val_s2.shape, X_val_cov.shape)
print("Test :", X_test_s2.shape, X_test_cov.shape)


# ============================================================
# 13. EXPAND COVARIATES ACROSS TIMESTEPS
# ============================================================

# Sentinel shape:
# (samples, timesteps, bands)

TIMESTEPS = X_train_s2.shape[1]


def repeat_covariates(covariates, timesteps):
    return np.repeat(
        covariates[:, np.newaxis, :],
        timesteps,
        axis=1
    )


X_train_cov_3d = repeat_covariates(X_train_cov, TIMESTEPS)
X_val_cov_3d   = repeat_covariates(X_val_cov, TIMESTEPS)
X_test_cov_3d  = repeat_covariates(X_test_cov, TIMESTEPS)


print("\n=== EXPANDED COVARIATES ===")

print("Train:", X_train_cov_3d.shape)
print("Val  :", X_val_cov_3d.shape)
print("Test :", X_test_cov_3d.shape)


# ============================================================
# 14. COVARIATE ORDER
# ============================================================

# 0 temperature
# 1 precipitation
# 2 solar_radiation
# 3 soil_ph
# 4 organic_carbon
# 5 soil_texture
# 6 elevation
# 7 landforms


# ============================================================
# 15. BUILD ABLATION DATASETS
# ============================================================


# ------------------------------------------------------------
# A. BASELINE — Sentinel only
# ------------------------------------------------------------

X_train_base = X_train_s2
X_val_base   = X_val_s2
X_test_base  = X_test_s2


# ------------------------------------------------------------
# B. CLIMATE
# ------------------------------------------------------------

X_train_climate = np.concatenate(
    [X_train_s2, X_train_cov_3d[:, :, 0:3]],
    axis=2
)

X_val_climate = np.concatenate(
    [X_val_s2, X_val_cov_3d[:, :, 0:3]],
    axis=2
)

X_test_climate = np.concatenate(
    [X_test_s2, X_test_cov_3d[:, :, 0:3]],
    axis=2
)


# ------------------------------------------------------------
# C. SOIL
# ------------------------------------------------------------

X_train_soil = np.concatenate(
    [X_train_s2, X_train_cov_3d[:, :, 3:6]],
    axis=2
)

X_val_soil = np.concatenate(
    [X_val_s2, X_val_cov_3d[:, :, 3:6]],
    axis=2
)

X_test_soil = np.concatenate(
    [X_test_s2, X_test_cov_3d[:, :, 3:6]],
    axis=2
)


# ------------------------------------------------------------
# D. TOPOGRAPHY
# ------------------------------------------------------------

X_train_topo = np.concatenate(
    [X_train_s2, X_train_cov_3d[:, :, 6:8]],
    axis=2
)

X_val_topo = np.concatenate(
    [X_val_s2, X_val_cov_3d[:, :, 6:8]],
    axis=2
)

X_test_topo = np.concatenate(
    [X_test_s2, X_test_cov_3d[:, :, 6:8]],
    axis=2
)


# ------------------------------------------------------------
# E. FULL MODEL
# ------------------------------------------------------------

X_train_full = np.concatenate(
    [X_train_s2, X_train_cov_3d],
    axis=2
)

X_val_full = np.concatenate(
    [X_val_s2, X_val_cov_3d],
    axis=2
)

X_test_full = np.concatenate(
    [X_test_s2, X_test_cov_3d],
    axis=2
)


# ============================================================
# 16. FINAL SHAPE CHECK
# ============================================================

print("\n=== FINAL DATASET SHAPES ===")

print("Baseline :", X_train_base.shape)
print("Climate  :", X_train_climate.shape)
print("Soil     :", X_train_soil.shape)
print("Topo     :", X_train_topo.shape)
print("Full     :", X_train_full.shape)


# ============================================================
# 17. SAVE FINAL DATASETS
# ============================================================

# BASELINE
np.save(os.path.join(DATASET_DIR, "X_train_base.npy"), X_train_base)
np.save(os.path.join(DATASET_DIR, "X_val_base.npy"), X_val_base)
np.save(os.path.join(DATASET_DIR, "X_test_base.npy"), X_test_base)

# CLIMATE
np.save(os.path.join(DATASET_DIR, "X_train_climate.npy"), X_train_climate)
np.save(os.path.join(DATASET_DIR, "X_val_climate.npy"), X_val_climate)
np.save(os.path.join(DATASET_DIR, "X_test_climate.npy"), X_test_climate)

# SOIL
np.save(os.path.join(DATASET_DIR, "X_train_soil.npy"), X_train_soil)
np.save(os.path.join(DATASET_DIR, "X_val_soil.npy"), X_val_soil)
np.save(os.path.join(DATASET_DIR, "X_test_soil.npy"), X_test_soil)

# TOPOGRAPHY
np.save(os.path.join(DATASET_DIR, "X_train_topo.npy"), X_train_topo)
np.save(os.path.join(DATASET_DIR, "X_val_topo.npy"), X_val_topo)
np.save(os.path.join(DATASET_DIR, "X_test_topo.npy"), X_test_topo)

# FULL
np.save(os.path.join(DATASET_DIR, "X_train_full.npy"), X_train_full)
np.save(os.path.join(DATASET_DIR, "X_val_full.npy"), X_val_full)
np.save(os.path.join(DATASET_DIR, "X_test_full.npy"), X_test_full)


# ============================================================
# 18. SAVE LABELS
# ============================================================

np.save(os.path.join(LABEL_DIR, "y_train.npy"), y_train)
np.save(os.path.join(LABEL_DIR, "y_val.npy"), y_val)
np.save(os.path.join(LABEL_DIR, "y_test.npy"), y_test)


# ============================================================
# 19. DONE
# ============================================================

print("\n================================================")
print("PART 2 DATA PREPARATION COMPLETED SUCCESSFULLY")
print("================================================")

print("\nSaved datasets in:", DATASET_DIR)
print("Saved labels in:", LABEL_DIR)

print("\nSaved datasets:")
print("- X_train_base.npy")
print("- X_train_climate.npy")
print("- X_train_soil.npy")
print("- X_train_topo.npy")
print("- X_train_full.npy")

print("\nReady for model training.")