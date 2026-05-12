import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# =============================================================================
# FIX THE IMPORT - Add project root to Python path
# =============================================================================
import sys
PROJECT_ROOT = Path(r"C:\projects\resneur\projectcrops\project_crop_classiffication")
sys.path.insert(0, str(PROJECT_ROOT))

# Now import from the model folder
from model.mctnet_improved import MCTNetImproved

warnings.filterwarnings("ignore")


# =============================================================================
# SEED
# =============================================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


set_seed(42)


# =============================================================================
# CONFIG - OVERFITTING FIXES APPLIED (FIXED PATH)
# =============================================================================
DATA_DIR = PROJECT_ROOT / "arkansas" / "part1" / "4_results" / "4_data_preprocessing_result"
OUTPUT_DIR = PROJECT_ROOT / "arkansas" / "part3" / "part1_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Using device: {DEVICE}\n")

# === ARCHITECTURE — scaled down for 960-sample dataset ===
D_MODEL  = 32       # ↓ from 64
N_HEAD   = 4        # ↓ from 8
N_STAGE  = 2        # ↓ from 3
DROPOUT  = 0.4      # ↑ — more regularization for small data

# === TRAINING ===
BATCH_SIZE   = 16       # ↓ from 64 — 60 steps/epoch vs 15, smoother gradients
NUM_EPOCHS   = 200
LR           = 1e-4     # ↓ from 5e-4 — PRIMARY instability fix
WEIGHT_DECAY = 1e-4

# === REGULARIZATION ===
GRAD_CLIP        = 0.5   # ↓ from 1.0 — tighter for stability
LABEL_SMOOTHING  = 0.1

# === DATA AUGMENTATION ===
USE_AUGMENTATION   = True
NOISE_LEVEL        = 0.01
AUGMENTATION_PROB  = 0.5
MIXUP_ALPHA        = 0.0    # DISABLED — too few samples for stable mixup

# === WARMUP — longer ramp ===
WARMUP_EPOCHS   = 10    # ↑ from 5
WARMUP_START_LR = 1e-6

# === EARLY STOPPING ===
EARLY_STOP_PATIENCE  = 20
EARLY_STOP_MIN_DELTA = 0.001

# === ReduceLROnPlateau ===
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR   = 0.5
MIN_LR             = 1e-7

# === GRADIENT ACCUMULATION — not needed with small batch ===
GRADIENT_ACCUMULATION_STEPS = 1

# === MIXED PRECISION ===
USE_AMP = True

# === DATA INFO ===
NUM_TIMESTEPS = 36
NUM_BANDS     = 10


# =============================================================================
# LOSSES
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss   = nn.CrossEntropyLoss(reduction="none")(pred, target)
        pt        = torch.exp(-ce_loss)
        focal     = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


class CombinedLoss(nn.Module):
    """CE + Focal + Label Smoothing.

    Weights kept identical to original so behaviour is unchanged;
    label smoothing value is now 0.1 (set via config above).
    """

    def __init__(self, ce_weight=0.5, focal_weight=0.3, smooth_weight=0.2,
                 smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.ce_weight     = ce_weight
        self.focal_weight  = focal_weight
        self.smooth_weight = smooth_weight
        self.smoothing     = smoothing

        self.ce_loss    = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, pred, target):
        ce    = self.ce_loss(pred, target)
        focal = self.focal_loss(pred, target)

        n_classes = pred.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        smooth = torch.mean(
            torch.sum(-true_dist * torch.log_softmax(pred, dim=1), dim=1)
        )

        return self.ce_weight * ce + self.focal_weight * focal + self.smooth_weight * smooth


# =============================================================================
# NORMALIZATION
# =============================================================================
class NormalizationParams:
    def __init__(self):
        self.means = None
        self.stds  = None

    def fit(self, X, mask):
        bands      = X.shape[-1]
        self.means = np.zeros(bands)
        self.stds  = np.zeros(bands)
        for b in range(bands):
            valid          = mask > 0
            self.means[b]  = X[:, :, b][valid].mean()
            self.stds[b]   = X[:, :, b][valid].std() + 1e-6

    def transform(self, X):
        out = X.copy()
        for b in range(X.shape[-1]):
            out[:, :, b] = (out[:, :, b] - self.means[b]) / self.stds[b]
        return out


# =============================================================================
# DATASET
# =============================================================================
class CropDataset(Dataset):
    def __init__(self, data_dir, split, norm=None):
        X    = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
        mask = np.load(data_dir / f"mask_{split}.npy").astype(np.float32)
        y    = np.load(data_dir / f"y_{split}.npy").astype(np.int64)

        X = X * np.expand_dims(mask, axis=-1)
        X = np.nan_to_num(X, nan=0.0)

        if split == "train":
            self.norm = NormalizationParams()
            self.norm.fit(X, mask)
            X = self.norm.transform(X)
        else:
            self.norm = norm
            X = self.norm.transform(X)

        self.X     = X
        self.mask  = mask
        self.y     = y
        self.split = split

        print(f"Loaded {split}: {len(y)} samples")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        X    = self.X[i].copy()
        mask = self.mask[i]
        y    = self.y[i]

        if self.split == "train" and USE_AUGMENTATION:
            if np.random.rand() < AUGMENTATION_PROB:
                # Gaussian noise
                X += np.random.normal(0, NOISE_LEVEL, X.shape)

            if np.random.rand() < 0.3:
                # Gentle amplitude scaling
                X = X * np.random.uniform(0.95, 1.05)

            if np.random.rand() < 0.2:
                # Random temporal shift (roll along time axis)
                shift = np.random.randint(-3, 4)
                X     = np.roll(X, shift, axis=0)

        return (
            torch.tensor(X),
            torch.tensor(mask),
            torch.tensor(y),
        )


# =============================================================================
# MIXUP COLLATE
# =============================================================================
def mixup_collate(batch):
    """Apply mixup in feature space during collation."""
    Xs, masks, ys = zip(*batch)
    Xs    = torch.stack(Xs)
    masks = torch.stack(masks)
    ys    = torch.stack(ys)

    if MIXUP_ALPHA > 0 and np.random.rand() < 0.5:
        lam   = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
        idx   = torch.randperm(Xs.size(0))
        Xs    = lam * Xs + (1 - lam) * Xs[idx]
        masks = torch.maximum(masks, masks[idx])   # union of valid timesteps
        # Return mixed inputs; labels stay as original (soft mixup on features only)
        # For hard-label mixup return a tuple — handled in train_epoch below
        return Xs, masks, ys, ys[idx], lam

    return Xs, masks, ys, ys, 1.0   # lam=1 → pure original labels


# =============================================================================
# METRICS
# =============================================================================
def metrics(y_true, y_pred):
    return {
        "OA":    accuracy_score(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "F1":    f1_score(y_true, y_pred, average="macro"),
    }


# =============================================================================
# TRAINING
# =============================================================================
def train_epoch(model, loader, loss_fn, opt, scaler=None):
    """Single training epoch.

    Key changes vs original:
    - Cosine LR schedule removed from here; ReduceLROnPlateau handles it.
    - Mixup support via the collate_fn output signature.
    """
    model.train()
    losses, preds, labels = 0.0, [], []
    opt.zero_grad()

    for batch_idx, batch in enumerate(loader):
        # Unpack — mixup_collate always returns 5 items
        X, mask, y, y2, lam = batch
        X, mask, y, y2 = (
            X.to(DEVICE), mask.to(DEVICE),
            y.to(DEVICE), y2.to(DEVICE),
        )

        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand(-1, -1, X.shape[-1])

        if scaler is not None:
            with autocast():
                out, _ = model(X, mask)
                loss   = lam * loss_fn(out, y) + (1 - lam) * loss_fn(out, y2)
                loss   = loss / GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
        else:
            out, _ = model(X, mask)
            loss   = lam * loss_fn(out, y) + (1 - lam) * loss_fn(out, y2)
            loss   = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                opt.zero_grad()

        if torch.isnan(loss):
            continue

        losses += loss.item() * X.size(0) * GRADIENT_ACCUMULATION_STEPS
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    return losses / len(loader.dataset), metrics(labels, preds)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    model.eval()
    losses, preds, labels = 0.0, [], []

    for batch in loader:
        # eval loader uses default collate, but mixup_collate also works (lam=1)
        if len(batch) == 5:
            X, mask, y, _, _ = batch
        else:
            X, mask, y = batch

        X, mask, y = X.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)

        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand(-1, -1, X.shape[-1])

        out, _ = model(X, mask)
        loss   = loss_fn(out, y)

        losses += loss.item() * X.size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    return losses / len(loader.dataset), metrics(labels, preds), preds, labels


# =============================================================================
# PLOTTING
# =============================================================================
def plot_history(hist):
    e = range(len(hist["train_loss"]))

    plt.figure(figsize=(16, 10))

    ax = plt.subplot(2, 3, 1)
    ax.plot(e, hist["train_loss"], label="train", linewidth=2)
    ax.plot(e, hist["val_loss"],   label="val",   linewidth=2)
    ax.set_title("Loss (Lower is Better)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = plt.subplot(2, 3, 2)
    ax.plot(e, hist["train_oa"], label="train", linewidth=2)
    ax.plot(e, hist["val_oa"],   label="val",   linewidth=2)
    ax.set_title("Overall Accuracy (OA)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = plt.subplot(2, 3, 3)
    ax.plot(e, hist["train_kappa"], label="train", linewidth=2)
    ax.plot(e, hist["val_kappa"],   label="val",   linewidth=2)
    ax.set_title("Kappa Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Kappa")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = plt.subplot(2, 3, 4)
    ax.plot(e, hist["train_f1"], label="train", linewidth=2)
    ax.plot(e, hist["val_f1"],   label="val",   linewidth=2)
    ax.set_title("F1 Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = plt.subplot(2, 3, 5)
    gap = np.array(hist["val_loss"]) - np.array(hist["train_loss"])
    ax.plot(e, gap, color="red", linewidth=2)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Overfitting Gap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss − Train Loss")
    ax.grid(True, alpha=0.3)

    ax = plt.subplot(2, 3, 6)
    ax.plot(e, hist["lr"], color="green", linewidth=2)
    ax.set_title("Learning Rate", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📈 history.png saved")


def plot_cm(cm, names):
    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=names, yticklabels=names,
        cmap="Blues", cbar=True, square=True,
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class",      fontsize=12)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 confusion_matrix.png saved")

    with open(OUTPUT_DIR / "confusion_matrix.txt", "w") as f:
        f.write("Confusion Matrix:\n" + "=" * 50 + "\n")
        f.write(f"{'':15s}")
        for name in names:
            f.write(f"{name:15s}")
        f.write("\n")
        for i, name in enumerate(names):
            f.write(f"{name:15s}")
            for j in range(len(names)):
                f.write(f"{cm[i,j]:<15d}")
            f.write("\n")
        f.write("\nNormalized (%):\n")
        for i, name in enumerate(names):
            f.write(f"{name:15s}")
            for j in range(len(names)):
                f.write(f"{cm_pct[i,j]:<15.1f}")
            f.write("\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("MCTNet — OVERFITTING-FIXED VERSION")
    print("=" * 70)

    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)

    print(f"\n📊 Dataset  →  {meta['num_classes']} classes: {meta['classes']}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = MCTNetImproved(
        input_channels=NUM_BANDS,
        time_steps=NUM_TIMESTEPS,
        d_model=D_MODEL,
        nhead=N_HEAD,
        n_stages=N_STAGE,
        n_classes=meta["num_classes"],
        dropout=DROPOUT,
    ).to(DEVICE)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️  Model  →  {total:,} params ({trainable:,} trainable)")
    print(f"   D={D_MODEL}, H={N_HEAD}, stages={N_STAGE}, dropout={DROPOUT}")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    print(f"\n📁 Loading datasets …")
    train_ds = CropDataset(DATA_DIR, "train")
    val_ds   = CropDataset(DATA_DIR, "val",  train_ds.norm)
    test_ds  = CropDataset(DATA_DIR, "test", train_ds.norm)

    pin = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=pin, drop_last=True,
        collate_fn=mixup_collate,   # ← mixup applied here
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=pin,
        collate_fn=mixup_collate,   # lam=1 at eval (no mixing)
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=pin,
        collate_fn=mixup_collate,
    )

    # ── Optimiser & loss ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999),
    )

    loss_fn = CombinedLoss(
        ce_weight=0.5, focal_weight=0.3, smooth_weight=0.2,
        smoothing=LABEL_SMOOTHING,
    )

    # Warmup scheduler (linear ramp for first WARMUP_EPOCHS)
    def lambda_lr(epoch):
        return (epoch + 1) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1.0

    scheduler_warmup   = torch.optim.lr_scheduler.LambdaLR(opt, lambda_lr)

    # ReduceLROnPlateau — kicks in after warmup, reacts to val kappa stagnation
    scheduler_plateau  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=MIN_LR,
    )

    scaler = GradScaler() if USE_AMP and DEVICE.type == "cuda" else None
    if scaler:
        print("🔧 Mixed precision training enabled")

    # ── Training state ───────────────────────────────────────────────────────
    best_kappa       = -1.0
    best_val_loss    = float("inf")
    best_state       = None
    best_epoch       = 0
    no_improve_count = 0

    hist = {k: [] for k in [
        "train_loss", "val_loss",
        "train_oa",   "val_oa",
        "train_kappa","val_kappa",
        "train_f1",   "val_f1",
        "lr",
    ]}

    print(f"\n🚀 Training config:")
    print(f"   Batch={BATCH_SIZE}, GradAccum={GRADIENT_ACCUMULATION_STEPS}")
    print(f"   MaxEpochs={NUM_EPOCHS}, LR={LR}, WD={WEIGHT_DECAY}")
    print(f"   Dropout={DROPOUT}, LabelSmoothing={LABEL_SMOOTHING}")
    print(f"   Mixup α={MIXUP_ALPHA}, Noise={NOISE_LEVEL}, AugProb={AUGMENTATION_PROB}")
    print(f"   EarlyStop patience={EARLY_STOP_PATIENCE} (monitored: val Kappa)")
    print("=" * 70)

    start = time.time()

    for epoch in range(NUM_EPOCHS):

        # Warmup phase: step the warmup scheduler each epoch
        if epoch < WARMUP_EPOCHS:
            scheduler_warmup.step()

        # ── Train ──
        train_loss, tr = train_epoch(model, train_loader, loss_fn, opt, scaler)

        # ── Validate ──
        val_loss, va, _, _ = eval_epoch(model, val_loader, loss_fn)

        # ── After warmup: ReduceLROnPlateau on val Kappa ──
        if epoch >= WARMUP_EPOCHS:
            scheduler_plateau.step(va["Kappa"])

        # ── Record ──
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_oa"].append(tr["OA"])
        hist["val_oa"].append(va["OA"])
        hist["train_kappa"].append(tr["Kappa"])
        hist["val_kappa"].append(va["Kappa"])
        hist["train_f1"].append(tr["F1"])
        hist["val_f1"].append(va["F1"])
        hist["lr"].append(opt.param_groups[0]["lr"])

        # ── Early stopping on val Kappa (↑ = better) ──
        improved = va["Kappa"] > best_kappa + EARLY_STOP_MIN_DELTA

        if improved:
            best_kappa    = va["Kappa"]
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            no_improve_count = 0
            print(
                f"🌟 Epoch {epoch:3d} | ✨ NEW BEST  "
                f"Loss {train_loss:.4f}→{val_loss:.4f} | "
                f"Kappa {va['Kappa']:.4f} | OA {va['OA']:.4f}"
            )
        else:
            no_improve_count += 1

        if epoch % 5 == 0 or epoch == best_epoch:
            print(
                f"📊 Epoch {epoch:3d} | "
                f"Loss {train_loss:.4f}/{val_loss:.4f} | "
                f"OA {va['OA']:.4f} | Kappa {va['Kappa']:.4f} | "
                f"F1 {va['F1']:.4f} | LR {opt.param_groups[0]['lr']:.2e}"
            )

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch}  "
                  f"(best Kappa {best_kappa:.4f} at epoch {best_epoch})")
            break

    # ── Restore best weights ──────────────────────────────────────────────────
    training_time = (time.time() - start) / 60
    print("\n" + "=" * 70)
    print(f"✅ Training done in {training_time:.2f} min")
    print(f"   Best epoch: {best_epoch}  |  Best val Kappa: {best_kappa:.4f}")

    if best_state:
        model.load_state_dict(best_state)
        print(f"📦 Restored weights from epoch {best_epoch}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n🔍 Evaluating on test set …")
    test_loss, test_m, predictions, true_labels = eval_epoch(model, test_loader, loss_fn)
    cm = confusion_matrix(true_labels, predictions)

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n📈 Generating plots …")
    plot_history(hist)
    plot_cm(cm, meta["classes"])

    # ── Save model ───────────────────────────────────────────────────────────
    torch.save(
        {"epoch": best_epoch, "state_dict": best_state, "kappa": best_kappa},
        OUTPUT_DIR / "best_model.pt",
    )
    print(f"   💾 best_model.pt saved")

    # ── Save JSON results ────────────────────────────────────────────────────
    results = {
        "best_epoch":           best_epoch,
        "best_val_kappa":       float(best_kappa),
        "best_val_loss":        float(best_val_loss),
        "training_time_minutes":training_time,
        "test_metrics": {
            "loss":  float(test_loss),
            "OA":    float(test_m["OA"]),
            "Kappa": float(test_m["Kappa"]),
            "F1":    float(test_m["F1"]),
        },
        "confusion_matrix": cm.tolist(),
        "classes":          meta["classes"],
        "model_config": {
            "d_model":        D_MODEL,
            "n_head":         N_HEAD,
            "n_stage":        N_STAGE,
            "dropout":        DROPOUT,
            "batch_size":     BATCH_SIZE,
            "learning_rate":  LR,
            "weight_decay":   WEIGHT_DECAY,
            "label_smoothing":LABEL_SMOOTHING,
            "mixup_alpha":    MIXUP_ALPHA,
        },
        "training_history": {
            k: hist[k] for k in [
                "train_loss", "val_loss",
                "train_oa",   "val_oa",
                "train_kappa","val_kappa",
                "train_f1",   "val_f1",
            ]
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🎯 FINAL TEST RESULTS")
    print("=" * 70)
    print(f"   Test Loss  : {test_loss:.4f}")
    print(f"   Test OA    : {test_m['OA']:.4f}  ({test_m['OA']*100:.2f}%)")
    print(f"   Test Kappa : {test_m['Kappa']:.4f}")
    print(f"   Test F1    : {test_m['F1']:.4f}")

    class_acc = cm.diagonal() / cm.sum(axis=1)
    print(f"\n📋 Per-class Accuracy:")
    for i, name in enumerate(meta["classes"]):
        print(f"   {name:15s}: {class_acc[i]:.4f}  ({class_acc[i]*100:.1f}%)")

    print(f"\n📉 Loss Summary:")
    print(f"   Best val loss : {best_val_loss:.4f}")
    print(f"   Test loss     : {test_loss:.4f}")

    print(f"\n💾 All outputs → {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()