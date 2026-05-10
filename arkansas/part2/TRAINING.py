# TRAINING.py

# PART 2 — ABLATION EXPERIMENTS

# FINAL IMPROVED VERSION + DASHBOARD

# ============================================================


import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import sys
import os


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.append(PROJECT_ROOT)

from model.mctnet import MCTNet


# ============================================================
# INPUT / OUTPUT FOLDERS
# ============================================================

# Expected structure:
#
# project/
# ├── inputs/
# │   ├── datasets/
# │   │   ├── X_train_base.npy
# │   │   ├── X_val_base.npy
# │   │   ├── X_test_base.npy
# │   │   ├── X_train_climate.npy
# │   │   ├── X_val_climate.npy
# │   │   ├── X_test_climate.npy
# │   │   ├── X_train_soil.npy
# │   │   ├── X_val_soil.npy
# │   │   ├── X_test_soil.npy
# │   │   ├── X_train_topo.npy
# │   │   ├── X_val_topo.npy
# │   │   ├── X_test_topo.npy
# │   │   ├── X_train_full.npy
# │   │   ├── X_val_full.npy
# │   │   └── X_test_full.npy
# │   │
# │   └── labels/
# │       ├── y_train.npy
# │       ├── y_val.npy
# │       └── y_test.npy
# │
# ├── outputs/
# │   ├── models/
# │   ├── plots/
# │   └── results/
# │
# └── model.py

INPUT_DIR = "outputs"
DATASET_INPUT_DIR = os.path.join(INPUT_DIR, "datasets")
LABEL_INPUT_DIR = os.path.join(INPUT_DIR, "labels")

OUTPUT_DIR = "final_outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 20
MIN_LR = 1e-6
D_MODEL = 64
NHEAD = 4
NSTAGES = 3
DROPOUT = 0.5
NUM_CLASSES = 5


# ============================================================
# DATASETS
# ============================================================

DATASETS = {

    "baseline": {
        "train": os.path.join(DATASET_INPUT_DIR, "X_train_base.npy"),
        "val": os.path.join(DATASET_INPUT_DIR, "X_val_base.npy"),
        "test": os.path.join(DATASET_INPUT_DIR, "X_test_base.npy")
    },

    "climate": {
        "train": os.path.join(DATASET_INPUT_DIR, "X_train_climate.npy"),
        "val": os.path.join(DATASET_INPUT_DIR, "X_val_climate.npy"),
        "test": os.path.join(DATASET_INPUT_DIR, "X_test_climate.npy")
    },

    "soil": {
        "train": os.path.join(DATASET_INPUT_DIR, "X_train_soil.npy"),
        "val": os.path.join(DATASET_INPUT_DIR, "X_val_soil.npy"),
        "test": os.path.join(DATASET_INPUT_DIR, "X_test_soil.npy")
    },

    "topography": {
        "train": os.path.join(DATASET_INPUT_DIR, "X_train_topo.npy"),
        "val": os.path.join(DATASET_INPUT_DIR, "X_val_topo.npy"),
        "test": os.path.join(DATASET_INPUT_DIR, "X_test_topo.npy")
    },

    "full": {
        "train": os.path.join(DATASET_INPUT_DIR, "X_train_full.npy"),
        "val": os.path.join(DATASET_INPUT_DIR, "X_val_full.npy"),
        "test": os.path.join(DATASET_INPUT_DIR, "X_test_full.npy")
    }
}


# ============================================================
# LOAD LABELS
# ============================================================

y_train = np.load(os.path.join(LABEL_INPUT_DIR, "y_train.npy"))
y_val = np.load(os.path.join(LABEL_INPUT_DIR, "y_val.npy"))
y_test = np.load(os.path.join(LABEL_INPUT_DIR, "y_test.npy"))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n================================================")
print("CLASS DISTRIBUTION")
print("================================================")

unique, counts = np.unique(y_train, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Class {u}: {c}")


# ============================================================
# CREATE MASK
# ============================================================

def create_mask(x):
    mask = np.ones_like(x).astype(np.float32)
    return mask


# ============================================================
# DATALOADER
# ============================================================

def build_loader(X, y, shuffle=True):

    mask = create_mask(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.float32)

    dataset = TensorDataset(X, mask, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

    return loader


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_model(model, train_loader, val_loader, name):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=MIN_LR
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_oa": [],
        "val_oa": [],
        "lr": []
    }

    best_val_oa = 0
    early_stop_counter = 0

    for epoch in range(EPOCHS):

        model.train()

        train_losses = []
        train_preds = []
        train_targets = []

        for X, mask, y in train_loader:

            X = X.to(device)
            mask = mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, _ = model(X, mask)

            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)

            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(y.cpu().numpy())

        train_loss = np.mean(train_losses)

        train_oa = accuracy_score(
            train_targets,
            train_preds
        )

        model.eval()

        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():

            for X, mask, y in val_loader:

                X = X.to(device)
                mask = mask.to(device)
                y = y.to(device)

                logits, _ = model(X, mask)

                loss = criterion(logits, y)

                val_losses.append(loss.item())

                preds = torch.argmax(logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_loss = np.mean(val_losses)

        val_oa = accuracy_score(
            val_targets,
            val_preds
        )

        scheduler.step(val_oa)

        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_oa"].append(train_oa)
        history["val_oa"].append(val_oa)
        history["lr"].append(current_lr)

        if val_oa > best_val_oa:

            best_val_oa = val_oa
            early_stop_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(MODEL_DIR, f"best_{name}.pth")
            )

            print(
                f"\nBest model updated "
                f"({best_val_oa:.4f})"
            )

        else:
            early_stop_counter += 1

        print(
            f"[{name}] "
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train OA: {train_oa:.4f} | "
            f"Val OA: {val_oa:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if early_stop_counter >= PATIENCE:

            print("\n========================================")
            print(f"EARLY STOPPING FOR {name.upper()}")
            print("========================================")

            break

    return history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_loader):

    model.eval()

    preds_all = []
    targets_all = []

    with torch.no_grad():

        for X, mask, y in test_loader:

            X = X.to(device)
            mask = mask.to(device)

            logits, _ = model(X, mask)

            preds = torch.argmax(logits, dim=1)

            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(y.numpy())

    oa = accuracy_score(targets_all, preds_all)

    f1 = f1_score(
        targets_all,
        preds_all,
        average="macro"
    )

    kappa = cohen_kappa_score(
        targets_all,
        preds_all
    )

    cm = confusion_matrix(
        targets_all,
        preds_all
    )

    return oa, f1, kappa, cm


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

all_results = {}
all_histories = {}

for name, paths in DATASETS.items():

    print("\n================================================")
    print("TRAINING:", name.upper())
    print("================================================")

    X_train = np.load(paths["train"])
    X_val = np.load(paths["val"])
    X_test = np.load(paths["test"])

    print("\nTrain shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    train_loader = build_loader(X_train, y_train)

    val_loader = build_loader(
        X_val,
        y_val,
        shuffle=False
    )

    test_loader = build_loader(
        X_test,
        y_test,
        shuffle=False
    )

    input_channels = X_train.shape[2]

    model = MCTNet(
        input_channels=input_channels,
        time_steps=X_train.shape[1],
        d_model=D_MODEL,
        nhead=NHEAD,
        n_stages=NSTAGES,
        n_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(device)

    history = train_model(
        model,
        train_loader,
        val_loader,
        name
    )

    all_histories[name] = history

    with open(
        os.path.join(RESULT_DIR, f"{name}_history.json"),
        "w"
    ) as f:
        json.dump(history, f)

    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, f"best_{name}.pth"),
            map_location=device
        )
    )

    oa, f1, kappa, cm = evaluate_model(
        model,
        test_loader
    )

    all_results[name] = {
        "OA": oa,
        "F1": f1,
        "Kappa": kappa
    }

    print("\n================================================")
    print(f"FINAL TEST RESULTS — {name.upper()}")
    print("================================================")
    print("OA     :", round(oa, 4))
    print("F1     :", round(f1, 4))
    print("Kappa  :", round(kappa, 4))

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()

    plt.title(f"{name} Confusion Matrix")

    plt.savefig(
        os.path.join(PLOT_DIR, f"{name}_cm.png"),
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


# ============================================================
# FINAL RESULTS TABLE
# ============================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")

for name, metrics in all_results.items():

    print(f"\n{name.upper()}")
    print("OA     :", round(metrics["OA"], 4))
    print("F1     :", round(metrics["F1"], 4))
    print("Kappa  :", round(metrics["Kappa"], 4))


# ============================================================
# TRAINING HISTORY VISUALIZATION
# ============================================================

plt.figure(figsize=(14, 8))

for name, history in all_histories.items():

    plt.plot(
        history["train_oa"],
        label=f"{name}_train"
    )

    plt.plot(
        history["val_oa"],
        linestyle="--",
        label=f"{name}_val"
    )

plt.xlabel("Epoch")
plt.ylabel("Overall Accuracy")
plt.title("Training vs Validation OA")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(PLOT_DIR, "all_histories_oa.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# LOSS VISUALIZATION
# ============================================================

plt.figure(figsize=(14, 8))

for name, history in all_histories.items():

    plt.plot(
        history["train_loss"],
        label=f"{name}_train_loss"
    )

    plt.plot(
        history["val_loss"],
        linestyle="--",
        label=f"{name}_val_loss"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(PLOT_DIR, "all_histories_loss.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# LEARNING RATE VISUALIZATION
# ============================================================

plt.figure(figsize=(14, 8))

for name, history in all_histories.items():

    plt.plot(
        history["lr"],
        label=f"{name}"
    )

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Scheduling")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(PLOT_DIR, "learning_rates.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# FINAL OA COMPARISON
# ============================================================

names = list(all_results.keys())

OA_scores = [
    all_results[n]["OA"]
    for n in names
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
    names,
    OA_scores
)

plt.ylabel("Overall Accuracy")
plt.title("Ablation Study Results")

for bar, score in zip(bars, OA_scores):

    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{score:.3f}",
        ha='center',
        va='bottom'
    )

plt.savefig(
    os.path.join(PLOT_DIR, "ablation_oa.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# DASHBOARD
# ============================================================

fig, axs = plt.subplots(
    2,
    2,
    figsize=(18, 12)
)

for name, history in all_histories.items():
    axs[0, 0].plot(history["val_oa"], label=name)

axs[0, 0].set_title("Validation OA")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("OA")
axs[0, 0].legend()
axs[0, 0].grid(True)

for name, history in all_histories.items():
    axs[0, 1].plot(history["val_loss"], label=name)

axs[0, 1].set_title("Validation Loss")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Loss")
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 0].bar(names, OA_scores)
axs[1, 0].set_title("Final OA Comparison")
axs[1, 0].set_ylabel("OA")

f1_scores = [
    all_results[n]["F1"]
    for n in names
]

axs[1, 1].bar(names, f1_scores)
axs[1, 1].set_title("Final F1 Comparison")
axs[1, 1].set_ylabel("F1")

plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "dashboard.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# SAVE RESULTS TXT
# ============================================================

with open(
    os.path.join(RESULT_DIR, "final_results.txt"),
    "w"
) as f:

    f.write("FINAL RESULTS\n\n")

    for name, metrics in all_results.items():

        f.write(f"{name.upper()}\n")

        f.write(
            f"OA     : {metrics['OA']:.4f}\n"
        )

        f.write(
            f"F1     : {metrics['F1']:.4f}\n"
        )

        f.write(
            f"Kappa  : {metrics['Kappa']:.4f}\n\n"
        )


# ============================================================
# SAVE RESULTS CSV
# ============================================================

with open(
    os.path.join(RESULT_DIR, "final_results.csv"),
    "w"
) as f:

    f.write("Dataset,OA,F1,Kappa\n")

    for name, metrics in all_results.items():

        f.write(
            f"{name},"
            f"{metrics['OA']:.4f},"
            f"{metrics['F1']:.4f},"
            f"{metrics['Kappa']:.4f}\n"
        )


# ============================================================
# DONE
# ============================================================

print("\n================================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================================")

print("\nInput folders:")
print("-", DATASET_INPUT_DIR)
print("-", LABEL_INPUT_DIR)

print("\nOutput folders:")
print("-", MODEL_DIR)
print("-", PLOT_DIR)
print("-", RESULT_DIR)

print("\nGenerated files:")
print("- best_baseline.pth")
print("- best_climate.pth")
print("- best_soil.pth")
print("- best_topography.pth")
print("- best_full.pth")
print("- dashboard.png")
print("- all_histories_oa.png")
print("- all_histories_loss.png")
print("- learning_rates.png")
print("- ablation_oa.png")
print("- baseline_cm.png")
print("- climate_cm.png")
print("- soil_cm.png")
print("- topography_cm.png")
print("- full_cm.png")
print("- final_results.txt")
print("- final_results.csv")

