"""Vision Transformer branch: chart images → directional probability.

No internet access to HuggingFace weights in this environment, so we train a
compact ViT (Dosovitskiy-style: patch embedding + CLS token + Transformer
encoder) from scratch on candlestick windows — architecture-equivalent to
ViT-Tiny at 128×128 with patch 16 (64 patches).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg")


class ChartDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_size: int = 128,
                 labels: pd.DataFrame | None = None, augment: bool = False):
        df = manifest.copy()
        if labels is not None:
            df = df.merge(labels, on=["symbol", "date"], how="inner")
            self.has_labels = True
        else:
            self.has_labels = False
        self.df = df.reset_index(drop=True)
        self.size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load(self, path: str) -> torch.Tensor:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != (self.size, self.size):
                im = im.resize((self.size, self.size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        # ImageNet-ish normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        x = self._load(row["path"])
        if self.augment:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[2])          # horizontal flip ≈ time reverse (risky?) keep off
            if torch.rand(()) < 0.3:
                x = x + 0.03 * torch.randn_like(x)
        if self.has_labels:
            return x, torch.tensor(int(row["label"]), dtype=torch.long)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch=16, in_ch=3, dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                 # B, D, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, N, D
        return x


class ViT(nn.Module):
    """Compact Vision Transformer (ViT-Tiny class)."""

    def __init__(self, img_size: int = 128, patch: int = 16, dim: int = 128,
                 depth: int = 4, heads: int = 4, mlp_ratio: float = 4.0,
                 num_classes: int = 2, drop: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, 3, dim)
        n_patches = (img_size // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=drop, activation="gelu", batch_first=True, norm_first=True)
        try:
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth,
                                                 enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def _device() -> torch.device:
    return torch.device("cpu")


def train_vit(cfg: dict, manifest: pd.DataFrame, labels: pd.DataFrame,
              train_dates: set, val_dates: set,
              save_path: str | Path) -> tuple[ViT, dict]:
    """Train ViT on charts whose (symbol,date) is in train_dates; early-stop on val."""
    v = cfg["vision"]
    torch.set_num_threads(max(1, (os_cpu := __import__("os").cpu_count()) or 1))
    tr = manifest[manifest.apply(lambda r: (r["symbol"], str(r["date"])[:10]) in
                                 {(s, str(d)[:10]) for s, d in train_dates}, axis=1)]
    va = manifest[manifest.apply(lambda r: (r["symbol"], str(r["date"])[:10]) in
                                 {(s, str(d)[:10]) for s, d in val_dates}, axis=1)]
    # faster membership: rebuild via merge instead of apply
    tr_keys = pd.DataFrame(list(train_dates), columns=["symbol", "date"])
    tr_keys["date"] = tr_keys["date"].astype(str).str[:10]
    va_keys = pd.DataFrame(list(val_dates), columns=["symbol", "date"])
    va_keys["date"] = va_keys["date"].astype(str).str[:10]
    m = manifest.copy()
    m["date"] = m["date"].astype(str).str[:10]
    tr = m.merge(tr_keys, on=["symbol", "date"], how="inner")
    va = m.merge(va_keys, on=["symbol", "date"], how="inner")

    train_ds = ChartDataset(tr, v["image_size"], labels=labels_assign(labels, tr), augment=True)
    val_ds = ChartDataset(va, v["image_size"], labels=labels_assign(labels, va), augment=False)
    if len(train_ds) == 0:
        raise RuntimeError("ViT training set is empty — check chart manifest / dates")
    pin = False
    train_dl = DataLoader(train_ds, batch_size=v["batch_size"], shuffle=True,
                          num_workers=0, drop_last=len(train_ds) > v["batch_size"])
    val_dl = DataLoader(val_ds, batch_size=v["batch_size"], shuffle=False, num_workers=0) \
        if len(val_ds) else None

    model = ViT(img_size=v["image_size"], patch=v["patch_size"], dim=v["vit_dim"],
                depth=v["vit_depth"], heads=v["vit_heads"], mlp_ratio=v["vit_mlp_ratio"])
    model.to(_device())
    opt = torch.optim.AdamW(model.parameters(), lr=v["lr"], weight_decay=v["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=v["epochs"])
    crit = nn.CrossEntropyLoss(weight=class_weights(train_ds))

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(v["epochs"]):
        model.train()
        tot, corr, loss_sum = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(_device()), yb.to(_device())
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss) * len(yb)
            corr += int((logits.argmax(1) == yb).sum())
            tot += len(yb)
        sched.step()
        tr_acc = corr / max(tot, 1)
        rec = {"epoch": epoch, "train_acc": tr_acc, "train_loss": loss_sum / max(tot, 1)}
        if val_dl is not None:
            model.eval()
            vcorr, vtot, vloss = 0, 0, 0.0
            preds, gts = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(_device()), yb.to(_device())
                    logits = model(xb)
                    loss = crit(logits, yb)
                    vloss += float(loss) * len(yb)
                    vcorr += int((logits.argmax(1) == yb).sum())
                    vtot += len(yb)
                    preds.append(logits.argmax(1).cpu().numpy())
                    gts.append(yb.cpu().numpy())
            rec["val_acc"] = vcorr / max(vtot, 1)
            rec["val_loss"] = vloss / max(vtot, 1)
            from sklearn.metrics import f1_score
            if vtot:
                f1 = f1_score(np.concatenate(gts), np.concatenate(preds), zero_division=0)
                rec["val_f1"] = float(f1)
                if f1 >= best_f1:
                    best_f1 = f1
                    best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
        history.append(rec)
        print(f"    [vit] epoch {epoch}: {rec}")

    if best_state is not None:
        model.load_state_dict(best_state)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": v, "history": history}, save_path)
    return model, {"history": history, "best_val_f1": best_f1}


def labels_assign(labels: pd.DataFrame, manifest_sub: pd.DataFrame) -> pd.DataFrame:
    """labels df has symbol,date,label — align to manifest subset (date as str)."""
    lab = labels.copy()
    lab["date"] = lab["date"].astype(str).str[:10]
    sub = manifest_sub.copy()
    sub["date"] = sub["date"].astype(str).str[:10]
    merged = sub[["symbol", "date"]].merge(lab, on=["symbol", "date"], how="left")
    return merged


def class_weights(ds: ChartDataset) -> torch.Tensor | None:
    if not ds.has_labels or len(ds) == 0:
        return None
    y = ds.df["label"].astype(int).to_numpy()
    n = len(y)
    up = max(y.sum(), 1)
    down = max(n - y.sum(), 1)
    w = torch.tensor([n / (2.0 * down), n / (2.0 * up)], dtype=torch.float32)
    return w


@torch.no_grad()
def predict_proba(model: ViT, manifest: pd.DataFrame, image_size: int,
                  batch_size: int = 32) -> pd.DataFrame:
    """Return symbol, date, p_up for every row of manifest."""
    model.eval()
    m = manifest.copy()
    m["date"] = m["date"].astype(str).str[:10]
    ds = ChartDataset(m, image_size, labels=None)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    for xb in dl:
        logits = model(xb)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p.numpy())
    p_up = np.concatenate(probs) if probs else np.array([])
    out = m[["symbol", "date"]].copy()
    out["p_up_vit"] = p_up
    return out


def load_vit(cfg: dict, path: str | Path) -> ViT:
    v = cfg["vision"]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ViT(img_size=v["image_size"], patch=v["patch_size"], dim=v["vit_dim"],
                depth=v["vit_depth"], heads=v["vit_heads"], mlp_ratio=v["vit_mlp_ratio"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
