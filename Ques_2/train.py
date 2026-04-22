import os, json, glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
SAVE_DIR   = "Question2"         # plots saved here (as required)
MODEL_PATH = "model.pth"
N_CLASSES   = 23
IMG_H, IMG_W = 96, 128          # resize target (H, W)
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 5e-4
SEED        = 42
MAX_SAMPLES = None              # use full dataset

# single GPU only
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Dataset (preloads all images into RAM so GPU is never idle) ───────────────
class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        self.augment = augment
        self.imgs, self.masks = [], []
        for ip, mp in zip(image_paths, mask_paths):
            img  = cv2.imread(ip); img  = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
            img  = cv2.resize(img,  (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
            img  = img.astype(np.float32) / 255.0
            mask = cv2.imread(mp); mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
            mask = np.max(mask, axis=-1)
            self.imgs.append(img)
            self.masks.append(mask)
        print(f"  Preloaded {len(self.imgs)} samples into RAM")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img  = self.imgs[idx].copy()
        mask = self.masks[idx].copy()
        if self.augment and np.random.rand() > 0.5:
            img  = img[:, ::-1, :]
            mask = mask[:, ::-1]
        img  = torch.from_numpy(img.copy()).permute(2, 0, 1)
        mask = torch.from_numpy(mask.copy()).long()
        mask = torch.clamp(mask, 0, N_CLASSES - 1)
        return img, mask

# ── UNet Architecture ─────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        # Encoder
        self.enc1 = DoubleConv(3,   64)
        self.enc2 = DoubleConv(64,  128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        # Bottleneck
        self.bot  = DoubleConv(512, 1024)
        # Decoder
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512,  256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256,  128)
        self.up1  = nn.ConvTranspose2d(128,  64, 2, stride=2)
        self.dec1 = DoubleConv(128,   64)
        self.out  = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # 96x128 → 48x64 → 24x32 → 12x16 → 6x8 (bottleneck)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds, masks, n_classes=N_CLASSES):
    """Per-class IoU and Dice, then mean over present classes."""
    preds = preds.cpu().numpy().flatten()
    masks = masks.cpu().numpy().flatten()
    iou_list, dice_list = [], []
    for cls in range(n_classes):
        p = preds == cls
        m = masks == cls
        inter = int((p & m).sum())
        union = int((p | m).sum())
        denom = int(p.sum() + m.sum())
        if union > 0:
            iou_list.append(inter / union)
        if denom > 0:
            dice_list.append(2 * inter / denom)
    miou  = float(np.mean(iou_list))  if iou_list  else 0.0
    mdice = float(np.mean(dice_list)) if dice_list else 0.0
    return miou, mdice

# ── Training Loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, max_batches=None):
    model.eval()
    miou_sum, mdice_sum, n = 0.0, 0.0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        m, d  = compute_metrics(preds, masks)
        miou_sum += m; mdice_sum += d; n += 1
        if max_batches and n >= max_batches:
            break
    return miou_sum / n, mdice_sum / n

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # collect and split paths
    all_imgs  = sorted(glob.glob(os.path.join(DATA_DIR, "CameraRGB",  "*.png")))
    all_masks = sorted(glob.glob(os.path.join(DATA_DIR, "CameraMask", "*.png")))
    assert len(all_imgs) == len(all_masks) and len(all_imgs) > 0, \
        "Images/masks not found or count mismatch — check DATA_DIR paths."

    # cap to MAX_SAMPLES with fixed seed for reproducibility
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(all_imgs), size=min(MAX_SAMPLES, len(all_imgs)), replace=False)
    idx = sorted(idx)
    all_imgs  = [all_imgs[i]  for i in idx]
    all_masks = [all_masks[i] for i in idx]

    tr_imgs, te_imgs, tr_masks, te_masks = train_test_split(
        all_imgs, all_masks, test_size=0.2, random_state=SEED
    )
    print(f"Train: {len(tr_imgs)}  |  Test: {len(te_imgs)}")

    train_dl = DataLoader(
        CityscapesDataset(tr_imgs, tr_masks, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    test_dl = DataLoader(
        CityscapesDataset(te_imgs, te_masks, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    model     = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "miou": [], "mdice": []}

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_dl, criterion, optimizer)
        miou, mdice = evaluate(model, train_dl, max_batches=5)  # fast sample for curves
        scheduler.step()

        history["loss"].append(loss)
        history["miou"].append(miou)
        history["mdice"].append(mdice)
        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f}  mIOU={miou:.4f}  mDice={mdice:.4f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_miou, test_mdice = evaluate(model, test_dl)
    print(f"\n{'='*50}")
    print(f"Test mIOU : {test_miou:.4f}")
    print(f"Test mDice: {test_mdice:.4f}")
    print(f"{'='*50}\n")

    # ── Save artifacts ────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)
    with open("training_metrics.json", "w") as f:
        json.dump(history, f, indent=2)
    with open("test_metrics.json", "w") as f:
        json.dump({"miou": test_miou, "mdice": test_mdice}, f, indent=2)
    # store test split paths so app.py can reference masks by filename
    with open("test_split.json", "w") as f:
        json.dump({"images": te_imgs, "masks": te_masks}, f, indent=2)

    # ── Update README ─────────────────────────────────────────────────────────
    readme_path = "README.md"
    tag_line    = f"Question2: mIOU: {test_miou:.4f} and mDICE: {test_mdice:.4f}\n"
    if os.path.exists(readme_path):
        with open(readme_path) as f:
            content = f.read()
        if "Question2: mIOU:" in content:
            lines = content.splitlines(keepends=True)
            lines = [tag_line if "Question2: mIOU:" in l else l for l in lines]
            content = "".join(lines)
        else:
            content += "\n" + tag_line
        with open(readme_path, "w") as f:
            f.write(content)
    else:
        with open(readme_path, "w") as f:
            f.write(tag_line)
    print(f"README.md updated with test metrics.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["loss"], "b-o", linewidth=2, markersize=5)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["miou"], "g-o", linewidth=2, markersize=5)
    plt.axhline(0.48, color="r", linestyle="--", label="Threshold 0.48")
    plt.axhline(test_miou, color="orange", linestyle="-.", label=f"Test mIOU={test_miou:.4f}")
    plt.title("Training mIOU Curve")
    plt.xlabel("Epoch"); plt.ylabel("mIOU")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "miou_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["mdice"], "r-o", linewidth=2, markersize=5)
    plt.axhline(0.48, color="b", linestyle="--", label="Threshold 0.48")
    plt.axhline(test_mdice, color="purple", linestyle="-.", label=f"Test mDice={test_mdice:.4f}")
    plt.title("Training mDice Curve")
    plt.xlabel("Epoch"); plt.ylabel("mDice")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "mdice_curve.png"), dpi=150)
    plt.close()

    print(f"Plots saved to '{SAVE_DIR}/'")
    print(f"Model saved to '{MODEL_PATH}'")

if __name__ == "__main__":
    main()
