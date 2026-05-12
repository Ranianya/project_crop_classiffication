# TRAINING_PART3_FINAL.py



import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)




PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model.mctnet_improved import MCTNetImproved


# ============================================================
# PATHS
# ============================================================


BASE_DIR = os.path.dirname(__file__)

INPUT_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "part2",
        "outputs"
    )
)
DATASET_INPUT_DIR = os.path.join(INPUT_DIR, "datasets")
LABEL_INPUT_DIR = os.path.join(INPUT_DIR, "labels")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_outputs"
)

MODELS_DIR = os.path.join(OUTPUT_DIR, "models2")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots2")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results2")


# ============================================================
# OUTPUT FOLDERS
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n================================================")
print("DEVICE")
print("================================================")
print(device)


# ============================================================
# RANDOM SEED
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 20
MIN_LR = 1e-6

D_MODEL = 64
NHEAD = 4
NSTAGES = 3
DROPOUT = 0.4
NUM_CLASSES = 4

LABEL_SMOOTHING = 0.05
GRAD_CLIP = 1.0
EMA_DECAY = 0.995
USE_AMP = torch.cuda.is_available()
TTA_ROUNDS = 5
MIN_DELTA = 0.002
TRAIN_NOISE_STD = 0.01


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
# CLASS WEIGHTS
# ============================================================
def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


CLASS_WEIGHTS = compute_class_weights(y_train, NUM_CLASSES).to(device)


# ============================================================
# MASK
# ============================================================
def create_mask(x):
    return np.ones_like(x, dtype=np.float32)


# ============================================================
# DATALOADER
# ============================================================
def build_loader(X, y, shuffle=True):
    mask = create_mask(X)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ============================================================
# EMA
# ============================================================
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, loader, criterion=None, tta=False):
    model.eval()

    preds_all = []
    targets_all = []
    losses = []

    with torch.no_grad():
        for X, mask, y in loader:
            X = X.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=USE_AMP):
                if tta:
                    logits_sum = 0
                    for _ in range(TTA_ROUNDS):
                        noise = torch.randn_like(X) * 0.005
                        logits, _ = model(X + noise, mask)
                        logits_sum += logits
                    logits = logits_sum / TTA_ROUNDS
                else:
                    logits, _ = model(X, mask)

                if criterion is not None:
                    loss = criterion(logits, y)
                    losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(y.cpu().numpy())

    oa = accuracy_score(targets_all, preds_all)
    f1 = f1_score(targets_all, preds_all, average="macro")
    kappa = cohen_kappa_score(targets_all, preds_all)
    cm = confusion_matrix(targets_all, preds_all)
    avg_loss = np.mean(losses) if losses else None

    return avg_loss, oa, f1, kappa, cm


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, val_loader, name):
    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=MIN_LR,
    )

    scaler = GradScaler("cuda", enabled=USE_AMP)
    ema = EMA(model, decay=EMA_DECAY)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_oa": [],
        "val_oa": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_val_oa = 0.0
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        model.train()

        train_losses = []
        train_preds = []
        train_targets = []

        for X, mask, y in train_loader:
            X = X.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            X = X + torch.randn_like(X) * TRAIN_NOISE_STD

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=USE_AMP):
                logits, _ = model(X, mask)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            train_losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(y.detach().cpu().numpy())

        train_loss = float(np.mean(train_losses))
        train_oa = accuracy_score(train_targets, train_preds)

        ema.apply_shadow(model)
        val_loss, val_oa, _, _, _ = evaluate_model(
            model,
            val_loader,
            criterion=criterion,
            tta=False,
        )
        ema.restore(model)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_oa"].append(train_oa)
        history["val_oa"].append(val_oa)
        history["lr"].append(current_lr)

        improved = False

        if val_loss < best_val_loss - MIN_DELTA:
            improved = True

        if val_oa > best_val_oa + MIN_DELTA:
            improved = True

        if improved:
            best_val_loss = min(best_val_loss, val_loss)
            best_val_oa = max(best_val_oa, val_oa)
            early_stop_counter = 0

            ema.apply_shadow(model)
            torch.save(
                model.state_dict(),
                os.path.join(MODELS_DIR, f"best_{name}.pth")
            )
            ema.restore(model)

            print(
                f"\n✅ Best model updated "
                f"(Val OA={val_oa:.4f}, Val Loss={val_loss:.4f})"
            )
        else:
            early_stop_counter += 1

        print(
            f"[{name}] Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train OA: {train_oa:.4f} | "
            f"Val OA: {val_oa:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if (train_oa - val_oa) > 0.15 and epoch > 20:
            early_stop_counter += 2

        if early_stop_counter >= PATIENCE:
            print("\n========================================")
            print(f"EARLY STOPPING FOR {name.upper()}")
            print("========================================")
            break

    return history


# ============================================================
# MAIN LOOP
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

    train_loader = build_loader(X_train, y_train, shuffle=True)
    val_loader = build_loader(X_val, y_val, shuffle=False)
    test_loader = build_loader(X_test, y_test, shuffle=False)

    model = MCTNetImproved(
        input_channels=X_train.shape[2],
        time_steps=X_train.shape[1],
        d_model=D_MODEL,
        nhead=NHEAD,
        n_stages=NSTAGES,
        n_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    history = train_model(model, train_loader, val_loader, name)
    all_histories[name] = history

    with open(
        os.path.join(RESULTS_DIR, f"{name}_history.json"),
        "w"
    ) as f:
        json.dump(history, f)

    model.load_state_dict(
        torch.load(
            os.path.join(MODELS_DIR, f"best_{name}.pth"),
            map_location=device
        )
    )

    _, oa, f1, kappa, cm = evaluate_model(
        model,
        test_loader,
        criterion=None,
        tta=True,
    )

    all_results[name] = {
        "OA": oa,
        "F1": f1,
        "Kappa": kappa,
    }

    print("\n================================================")
    print(f"FINAL TEST RESULTS — {name.upper()}")
    print("================================================")
    print("OA    :", round(oa, 4))
    print("F1    :", round(f1, 4))
    print("Kappa :", round(kappa, 4))

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{name}_cm.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# SAVE FINAL RESULTS
# ============================================================
print("\n================================================")
print("FINAL RESULTS")
print("================================================")

for name, metrics in all_results.items():
    print(f"\n{name.upper()}")
    print("OA    :", round(metrics["OA"], 4))
    print("F1    :", round(metrics["F1"], 4))
    print("Kappa :", round(metrics["Kappa"], 4))

with open(os.path.join(RESULTS_DIR, "final_results.csv"), "w") as f:
    f.write("Dataset,OA,F1,Kappa\n")
    for name, metrics in all_results.items():
        f.write(
            f"{name},"
            f"{metrics['OA']:.4f},"
            f"{metrics['F1']:.4f},"
            f"{metrics['Kappa']:.4f}\n"
        )


# ============================================================
# PLOT ACCURACY
# ============================================================
plt.figure(figsize=(14, 8))
for name, history in all_histories.items():
    plt.plot(history["train_oa"], label=f"{name}_train")
    plt.plot(history["val_oa"], "--", label=f"{name}_val")
plt.xlabel("Epoch")
plt.ylabel("Overall Accuracy")
plt.title("Improved MCTNet Training vs Validation OA")
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join(PLOTS_DIR, "all_histories_oa.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# PLOT LOSS
# ============================================================
plt.figure(figsize=(14, 8))
for name, history in all_histories.items():
    plt.plot(history["train_loss"], label=f"{name}_train_loss")
    plt.plot(history["val_loss"], "--", label=f"{name}_val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Improved MCTNet Loss Curves")
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join(PLOTS_DIR, "all_histories_loss.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\n================================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================================")
print("Results saved in:")
print(MODELS_DIR)
print(PLOTS_DIR)
print(RESULTS_DIR)


