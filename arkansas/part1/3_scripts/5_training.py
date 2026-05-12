# train_arkansas_optimized.py
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
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from torch.utils.data import Dataset, DataLoader

# IMPORTANT: Fix the import - define BASE_DIR directly
BASE_DIR = Path(r"C:\projects\resneur\projectcrops\project_crop_classiffication")

# Now import the model (make sure the path is correct)
import sys
sys.path.insert(0, str(BASE_DIR))

# Try importing from the correct file names
try:
    from model.mctnet import MCTNet
    print("✓ Imported MCTNet from model.mctnet")
except ImportError:
    try:
        from model.mctnet_improved import MCTNet
        print("✓ Imported MCTNet from model.mctnet_improved")
    except ImportError:
        raise ImportError("Could not find MCTNet in model/ folder. Files available: mctnet.py and mctnet_improved.py")

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# SEEDS FOR REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

set_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION - OPTIMIZED
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "arkansas" / "part1" / "4_results" / "4_data_preprocessing_result"
OUTPUT_DIR = BASE_DIR / "arkansas" / "part1" / "4_results" / "5_training_results" 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Training hyperparameters - UPDATED
BATCH_SIZE = 64                      # Changed from 32 to 64 (smoother gradients)
NUM_EPOCHS = 200
LR = 5e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.5                        # Changed from 0.65 back to 0.5 (better for batch size 64)
PATIENCE = 15
GRAD_CLIP = 1.0

# Data augmentation
USE_AUGMENTATION = True
NOISE_LEVEL = 0.02                   # Reduced slightly
AUGMENTATION_PROB = 0.3              # Reduced slightly

# Model architecture
N_STAGE = 3
N_HEAD = 4
NUM_TIMESTEPS = 36
NUM_BANDS = 10

# Learning rate scheduling
WARMUP_EPOCHS = 5
WARMUP_START_LR = 1e-5
SCHEDULER_PATIENCE = 8
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION PARAMETERS STORAGE
# ─────────────────────────────────────────────────────────────────────────────
class NormalizationParams:
    """Store normalization parameters computed from training set"""
    def __init__(self):
        self.means = None
        self.stds = None
        
    def fit(self, X, mask):
        """Compute means and stds from training data"""
        num_bands = X.shape[-1]
        self.means = np.zeros(num_bands)
        self.stds = np.zeros(num_bands)
        
        for band in range(num_bands):
            band_data = X[:, :, band]
            valid_mask = mask > 0
            if valid_mask.any():
                self.means[band] = band_data[valid_mask].mean()
                self.stds[band] = band_data[valid_mask].std() + 1e-6
            else:
                self.means[band] = 0
                self.stds[band] = 1
                
        print(f"Normalization params computed:")
        for band in range(num_bands):
            print(f"  Band {band}: mean={self.means[band]:.4f}, std={self.stds[band]:.4f}")
                
    def transform(self, X, mask):
        """Apply training set normalization to any data"""
        X_normalized = X.copy()
        for band in range(X.shape[-1]):
            X_normalized[:, :, band] = (X[:, :, band] - self.means[band]) / self.stds[band]
        return X_normalized


# ─────────────────────────────────────────────────────────────────────────────
# DATASET WITH CORRECT NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, norm_params=None):
        # Load data
        X = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
        mask = np.load(data_dir / f"mask_{split}.npy").astype(np.float32)
        y = np.load(data_dir / f"y_{split}.npy").astype(np.int64)
        
        # Apply mask - set missing values to 0
        X = X * np.expand_dims(mask, axis=-1)
        
        # Replace any NaN with 0
        X = np.nan_to_num(X, nan=0.0)
        
        # Apply normalization
        if split == "train":
            # Compute and store normalization params from training set
            self.norm_params = NormalizationParams()
            self.norm_params.fit(X, mask)
            X = self.norm_params.transform(X, mask)
        else:
            # Use training set normalization params
            if norm_params is None:
                raise ValueError(f"For {split} set, norm_params must be provided!")
            self.norm_params = norm_params
            X = self.norm_params.transform(X, mask)
        
        self.X = X
        self.mask = mask
        self.y = y
        self.split = split
        
        print(f"Loaded {split} set: {len(self.y)} samples")
        print(f"  X shape: {self.X.shape}, mask shape: {self.mask.shape}")
        print(f"  X range: [{self.X.min():.4f}, {self.X.max():.4f}]")
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X = self.X[idx].copy()
        mask = self.mask[idx]
        y = self.y[idx]
        
        # Data augmentation for training only
        if self.split == "train" and USE_AUGMENTATION:
            # Random noise
            if np.random.random() < AUGMENTATION_PROB:
                noise = np.random.normal(0, NOISE_LEVEL, X.shape).astype(np.float32)
                X = X + noise
            
            # Random scaling (small perturbations)
            if np.random.random() < 0.3:
                scale_factor = np.random.uniform(0.95, 1.05)
                X = X * scale_factor
        
        return (
            torch.from_numpy(X),
            torch.from_numpy(mask),
            torch.tensor(y, dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────────────────
# WARMUP SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
class WarmupReduceLROnPlateau:
    def __init__(self, optimizer, warmup_epochs, warmup_start_lr, target_lr, 
                 mode='max', factor=0.5, patience=8, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.target_lr = target_lr
        self.warmup_step = (target_lr - warmup_start_lr) / warmup_epochs
        self.is_warmup = True
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, 
            patience=patience, min_lr=min_lr
        )
        
    def step(self, epoch, metric=None):
        if epoch < self.warmup_epochs:
            lr = self.warmup_start_lr + self.warmup_step * epoch
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        elif self.is_warmup:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.target_lr
            self.is_warmup = False
            return self.target_lr
        else:
            if metric is not None:
                self.scheduler.step(metric)
            return self.optimizer.param_groups[0]['lr']


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    return {
        "OA": accuracy_score(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN EPOCH
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    preds = []
    labels = []
    
    for batch_idx, (X, mask, y) in enumerate(loader):
        X = X.to(DEVICE)
        mask = mask.to(DEVICE)
        y = y.to(DEVICE)
        
        if mask.dim() == 2:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, X.shape[-1])
        else:
            mask_expanded = mask
        
        optimizer.zero_grad()
        
        logits, _ = model(X, mask_expanded)
        loss = criterion(logits, y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Warning: NaN/Inf loss at epoch {epoch}, batch {batch_idx}")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    
    epoch_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(labels, preds)
    
    return epoch_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION EPOCH
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    
    total_loss = 0.0
    preds = []
    labels = []
    
    for X, mask, y in loader:
        X = X.to(DEVICE)
        mask = mask.to(DEVICE)
        y = y.to(DEVICE)
        
        if mask.dim() == 2:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, X.shape[-1])
        else:
            mask_expanded = mask
        
        logits, _ = model(X, mask_expanded)
        loss = criterion(logits, y)
        
        total_loss += loss.item() * X.size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    
    epoch_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(labels, preds)
    
    return epoch_loss, metrics, np.array(preds), np.array(labels)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_history(history, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_title("Loss", fontsize=12)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # OA
    axes[0, 1].plot(epochs, history["train_oa"], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history["val_oa"], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_title("Overall Accuracy", fontsize=12)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("OA")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Kappa
    axes[0, 2].plot(epochs, history["train_kappa"], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, history["val_kappa"], 'r-', label='Val', linewidth=2)
    axes[0, 2].set_title("Kappa", fontsize=12)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Kappa")
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # F1
    axes[1, 0].plot(epochs, history["train_f1"], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history["val_f1"], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_title("F1 Score", fontsize=12)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Overfitting Gap
    gap_oa = np.array(history["train_oa"]) - np.array(history["val_oa"])
    axes[1, 1].fill_between(epochs, 0, gap_oa, alpha=0.3, color='red', label='Gap')
    axes[1, 1].plot(epochs, gap_oa, 'r-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=0.10, color='g', linestyle='--', alpha=0.5, label='Target (<10%)')
    axes[1, 1].set_title(f"Train-Val Gap (Final: {gap_oa[-1]:.2%})", fontsize=12)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("OA Gap")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Learning rate
    axes[1, 2].plot(epochs, history["lr"], 'g-', linewidth=2)
    axes[1, 2].set_title("Learning Rate", fontsize=12)
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("LR")
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("MCTNet Training - CORRECTED NORMALIZATION")
    print("=" * 80)
    print(f"Device:           {DEVICE}")
    print(f"Batch Size:       {BATCH_SIZE} (smoother gradients)")
    print(f"Dropout:          {DROPOUT}")
    print(f"Learning Rate:    {LR}")
    print(f"Weight Decay:     {WEIGHT_DECAY}")
    print(f"Data Augmentation: {USE_AUGMENTATION} (noise={NOISE_LEVEL}, prob={AUGMENTATION_PROB})")
    print("=" * 80)
    
    # Load metadata
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    
    num_classes = meta["num_classes"]
    class_names = meta["classes"]
    print(f"\nClasses: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # Create train dataset first (to get normalization params)
    train_ds = CropDataset(DATA_DIR, "train", norm_params=None)
    
    # Get normalization params from training set
    norm_params = train_ds.norm_params
    
    # Create val and test datasets using training normalization
    val_ds = CropDataset(DATA_DIR, "val", norm_params=norm_params)
    test_ds = CropDataset(DATA_DIR, "test", norm_params=norm_params)
    
    # DataLoaders
    pin_memory = torch.cuda.is_available()
    num_workers = 2 if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Initialize model
    model = MCTNet(
        input_channels=NUM_BANDS,
        time_steps=NUM_TIMESTEPS,
        d_model=64,
        nhead=N_HEAD,
        n_stages=N_STAGE,
        n_classes=num_classes,
        kernel_sizes=[3, 5, 7],
        dropout=DROPOUT,
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=WARMUP_START_LR, weight_decay=WEIGHT_DECAY)
    
    scheduler = WarmupReduceLROnPlateau(
        optimizer, warmup_epochs=WARMUP_EPOCHS,
        warmup_start_lr=WARMUP_START_LR, target_lr=LR,
        mode='max', factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=MIN_LR
    )
    
    # Training state
    best_kappa = -1
    best_epoch = 0
    best_state = None
    patience_counter = 0
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_oa": [], "val_oa": [],
        "train_kappa": [], "val_kappa": [],
        "train_f1": [], "val_f1": [],
        "lr": []
    }
    
    start_time = time.time()
    
    print("\nStarting Training...")
    print("=" * 80)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_metrics, _, _ = eval_epoch(model, val_loader, criterion)
        
        current_lr = scheduler.step(epoch - 1, val_metrics["Kappa"])
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_oa"].append(train_metrics["OA"])
        history["val_oa"].append(val_metrics["OA"])
        history["train_kappa"].append(train_metrics["Kappa"])
        history["val_kappa"].append(val_metrics["Kappa"])
        history["train_f1"].append(train_metrics["F1"])
        history["val_f1"].append(val_metrics["F1"])
        history["lr"].append(current_lr)
        
        gap = train_metrics["OA"] - val_metrics["OA"]
        
        # Best model
        if val_metrics["Kappa"] > best_kappa:
            best_kappa = val_metrics["Kappa"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"✨ Epoch {epoch:03d} | Best Kappa: {best_kappa:.4f} | Gap: {gap:.2%}")
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 5 == 0 or epoch == 1:
            warmup_status = " (Warmup)" if epoch <= WARMUP_EPOCHS else ""
            print(
                f"Epoch {epoch:03d}{warmup_status} | "
                f"Loss {train_loss:.4f}/{val_loss:.4f} | "
                f"OA {train_metrics['OA']:.4f}/{val_metrics['OA']:.4f} | "
                f"Gap {gap:+.2%} | "
                f"Kappa {val_metrics['Kappa']:.4f}"
            )
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    elapsed_time = time.time() - start_time
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    test_loss, test_metrics, test_preds, test_labels = eval_epoch(model, test_loader, criterion)
    final_gap = history["train_oa"][best_epoch-1] - history["val_oa"][best_epoch-1] if best_epoch > 0 else 0
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Train OA:  {history['train_oa'][best_epoch-1]:.4f}")
    print(f"Val OA:    {history['val_oa'][best_epoch-1]:.4f}")
    print(f"Test OA:   {test_metrics['OA']:.4f}")
    print(f"Gap:       {final_gap:.2%}")
    print(f"Kappa:     {test_metrics['Kappa']:.4f}")
    print(f"F1:        {test_metrics['F1']:.4f}")
    print(f"Time:      {elapsed_time / 60:.2f} minutes")
    
    # Save results
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, class_names, OUTPUT_DIR / "confusion_matrix.png")
    plot_training_history(history, OUTPUT_DIR / "training_history.png")
    
    results = {
        "test_oa": float(test_metrics["OA"]),
        "test_kappa": float(test_metrics["Kappa"]),
        "test_f1": float(test_metrics["F1"]),
        "overfitting_gap": float(final_gap),
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "dropout": DROPOUT,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
        }
    }
    
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()