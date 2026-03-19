"""
src/train.py
============
Loop de entrenamiento para LSMTransformer con:
  - Pérdida de clasificación de glosa (CrossEntropy ponderada)
  - Pérdida de end-trigger (BCEWithLogits)
  - Pérdida AimCLR / D3M (divergencia KL simétrica entre vistas contrastivas)
  - Early stopping por Val Macro-F1
  - Checkpointing del mejor modelo
  - Logging compatible con TensorBoard y/o W&B

Uso rápido:
    python -m src.train \
        --parquet corpus_LSM_esp/lsm_dataset.parquet \
        --output  runs/exp01 \
        --epochs  100 \
        --batch   32 \
        --device  cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Compatibles con ejecución desde src/ o desde raíz
try:
    from src.dataset      import LSMDataset, collate_fn, split_dataset, TMAX, INPUT_DIM, N_LANDMARKS
    from src.augmentations import LandmarkAugmenter, AimCLRViewGenerator
    from src.model         import LSMTransformer
except ModuleNotFoundError:
    from dataset       import LSMDataset, collate_fn, split_dataset, TMAX, INPUT_DIM, N_LANDMARKS
    from augmentations import LandmarkAugmenter, AimCLRViewGenerator
    from model         import LSMTransformer

import warnings
warnings.filterwarnings("ignore", message="The number of unique classes")


# ─────────────────────────────────────────────────────────────────────────────
# Pérdida D3M (AimCLR)
# ─────────────────────────────────────────────────────────────────────────────

class D3MLoss(nn.Module):
    """
    KL simétrica entre distribuciones de vista original y aumentada.
        L = KL(p ‖ q) + KL(q ‖ p),  p/q = softmax(logits / τ)
    """

    def __init__(self, tau: float = 0.5):
        super().__init__()
        self.tau = tau

    def forward(self, logits_orig: Tensor, logits_aug: Tensor) -> Tensor:
        p = F.softmax(logits_orig / self.tau, dim=-1).clamp(min=1e-8)
        q = F.softmax(logits_aug  / self.tau, dim=-1).clamp(min=1e-8)
        return (p * (p.log() - q.log())).sum(-1).mean() + \
               (q * (q.log() - p.log())).sum(-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._preds:  List[int]   = []
        self._labels: List[int]   = []
        self._losses: List[float] = []

    def update(self, preds: Tensor, labels: Tensor, loss: float):
        self._preds.extend(preds.cpu().tolist())
        self._labels.extend(labels.cpu().tolist())
        self._losses.append(loss)

    def compute(self) -> Dict[str, float]:
        preds  = np.array(self._preds)
        labels = np.array(self._labels)
        return {
            "top1":     float((preds == labels).mean()),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "loss":     float(np.mean(self._losses)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best        = -math.inf
        self.counter     = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Collate con vistas AimCLR
# ─────────────────────────────────────────────────────────────────────────────

def make_aimclr_collate(view_gen: AimCLRViewGenerator, tmax: int = TMAX):
    """
    Extiende collate_fn estándar añadiendo view1 y view2 para la pérdida D3M.
    Las vistas se generan desde los datos crudos (post-preprocess, solo xy)
    añadiendo un score dummy=1.0 para que view_gen funcione con (T,133,3).
    """

    def _collate(batch):
        base    = collate_fn(batch)
        kpts_np = base["keypoints"].numpy()        # (B, T, 266)
        B, T, _ = kpts_np.shape
        kpts_3d = kpts_np.reshape(B, T, N_LANDMARKS, 2)

        views1, views2 = [], []
        for i in range(B):
            kpts_score = np.concatenate(
                [kpts_3d[i], np.ones((T, N_LANDMARKS, 1), dtype=np.float32)],
                axis=-1,
            )  # (T, 133, 3)
            v1, v2 = view_gen(kpts_score)
            views1.append(torch.from_numpy(v1[:, :, :2].reshape(T, INPUT_DIM)))
            views2.append(torch.from_numpy(v2[:, :, :2].reshape(T, INPUT_DIM)))

        base["view1"] = torch.stack(views1)
        base["view2"] = torch.stack(views2)
        return base

    return _collate


# ─────────────────────────────────────────────────────────────────────────────
# Paso de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def train_step(
    model:     LSMTransformer,
    batch:     Dict,
    optimizer: torch.optim.Optimizer,
    ce_loss:   nn.CrossEntropyLoss,
    bce_loss:  nn.BCEWithLogitsLoss,
    d3m_loss:  D3MLoss,
    alpha:     float,
    beta:      float,
    lam:       float,
    device:    torch.device,
    scaler:    Optional[torch.cuda.amp.GradScaler] = None,
    clip_norm: float = 1.0,
) -> Tuple[float, Tensor]:

    kpts   = batch["keypoints"].to(device, non_blocking=True)
    mask   = batch["valid_mask"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    has_views  = "view1" in batch
    trigger_gt = torch.ones(kpts.size(0), 1, device=device)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
        glosa_logits, trigger_logits = model(kpts, mask)

        loss = alpha * ce_loss(glosa_logits, labels) + \
               beta  * bce_loss(trigger_logits, trigger_gt)

        if has_views and lam > 0:
            v1 = batch["view1"].to(device, non_blocking=True)
            v2 = batch["view2"].to(device, non_blocking=True)
            g1, _ = model(v1, mask)
            g2, _ = model(v2, mask)
            loss = loss + lam * d3m_loss(g1, g2)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

    return loss.item(), glosa_logits.argmax(dim=-1).detach()


# ─────────────────────────────────────────────────────────────────────────────
# Epoch de validación
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(
    model:   LSMTransformer,
    loader:  DataLoader,
    ce_loss: nn.CrossEntropyLoss,
    device:  torch.device,
) -> Dict[str, float]:
    model.eval()
    tracker = MetricTracker()
    for batch in loader:
        kpts   = batch["keypoints"].to(device, non_blocking=True)
        mask   = batch["valid_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits, _ = model(kpts, mask)
        tracker.update(logits.argmax(-1), labels, ce_loss(logits, labels).item())
    model.train()
    return tracker.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: argparse.Namespace):
    out_dir   = Path(cfg.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"
    log_path  = out_dir / "train_log.jsonl"

    device = torch.device(
        cfg.device if (torch.cuda.is_available() or cfg.device == "cpu") else "cpu"
    )
    print(f"[Train] Dispositivo: {device}")

    # ── Aumentaciones ─────────────────────────────────────────────────────────
    augmenter = LandmarkAugmenter(
        p_speed=0.9,          # antes 0.8
        p_scale=0.7,          # antes 0.6
        p_wrist_noise=0.8,    # antes 0.7
        p_blur=0.6,           # antes 0.5
        p_crop=0.6,           # antes 0.5
        p_rotation=0.5,       # antes 0.4
        speed_range=(0.55, 1.45),     # antes (0.6, 1.4)
        scale_range=(0.75, 1.25),     # antes (0.80, 1.20)
        wrist_noise_range=(0.03, 0.12), # antes (0.02, 0.10)
        blur_sigma_range=(0.5, 3.0),  # antes (0.5, 2.5)
        crop_ratio_range=(0.08, 0.25), # antes (0.05, 0.20)
        rotation_range_deg=(-12.0, 12.0), # antes (-10, 10)
    )
    view_gen = AimCLRViewGenerator(
        p_flip=0.5,
        p_group_dropout=0.5,
        p_blur=0.6,
        blur_sigma=2.5,
    )

    # ── Dataset & splits ──────────────────────────────────────────────────────
    print("[Train] Cargando dataset…")
    train_ds, val_ds, test_ds = split_dataset(
        cfg.parquet,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=cfg.seed,
        augment_train=augmenter,
        score_thresh=0.3,
        tmax=TMAX,
        split_by_speaker=cfg.split_by_speaker
    )
    num_classes = train_ds.num_classes
    print(
        f"[Train] {len(train_ds)} train | {len(val_ds)} val | "
        f"{len(test_ds)} test | {num_classes} clases"
    )

    train_collate = make_aimclr_collate(view_gen, tmax=TMAX)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=train_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch * 2,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = LSMTransformer(
        num_classes=num_classes,
        input_dim=INPUT_DIM,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_ff,
        dropout=cfg.dropout,
        tmax=TMAX,
        trigger_window=cfg.trigger_window,
        use_eadm=cfg.use_eadm,
    ).to(device)
    print(f"[Train] {model.param_count()}")

    # ── Pérdidas ──────────────────────────────────────────────────────────────
    class_weights = train_ds.get_label_weights().to(device)
    ce_loss  = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    bce_loss = nn.BCEWithLogitsLoss()
    d3m_loss = D3MLoss(tau=cfg.tau)

    # ── Optimizador ───────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(), lr=cfg.lr,
        betas=(0.9, 0.999), weight_decay=1e-4,
    )

    # Warmup lineal → cosine decay
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── AMP ───────────────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and cfg.amp) else None

    # ── Loop ──────────────────────────────────────────────────────────────────
    es      = EarlyStopping(patience=cfg.patience)
    best_f1 = 0.0

    print("[Train] Iniciando entrenamiento…")
    for epoch in range(1, cfg.epochs + 1):

        # λ de AimCLR crece linealmente en los primeros aimclr_warmup epochs
        lam_eff = cfg.lam * min(1.0, epoch / max(1, cfg.aimclr_warmup))

        model.train()
        tracker = MetricTracker()
        t0 = time.time()

        for batch in train_loader:
            loss_val, preds = train_step(
                model, batch, optimizer,
                ce_loss, bce_loss, d3m_loss,
                alpha=cfg.alpha, beta=cfg.beta, lam=lam_eff,
                device=device, scaler=scaler, clip_norm=cfg.clip_norm,
            )
            tracker.update(preds, batch["label"].to(device), loss_val)

        train_m = tracker.compute()
        val_m   = val_epoch(model, val_loader, ce_loss, device)
        scheduler.step()

        elapsed  = time.time() - t0
        improved = es.step(val_m["macro_f1"])

        if improved:
            best_f1 = val_m["macro_f1"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_f1":      best_f1,
                "cfg":         vars(cfg),
                "glosa2idx":   train_ds.glosa2idx,
            }, ckpt_path)

        lr_now = scheduler.get_last_lr()[0]
        log_entry = {
            "epoch": epoch, "lr": lr_now, "lam": round(lam_eff, 4),
            "time":  round(elapsed, 2),
            "train": {k: round(v, 4) for k, v in train_m.items()},
            "val":   {k: round(v, 4) for k, v in val_m.items()},
            "best_val_f1": round(best_f1, 4),
            "improved": improved,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(
            f"Ep {epoch:03d}/{cfg.epochs} | lr {lr_now:.2e} | λ {lam_eff:.3f} | "
            f"loss {train_m['loss']:.4f} | "
            f"train {train_m['top1']:.3f} | "
            f"val {val_m['top1']:.3f} | F1 {val_m['macro_f1']:.3f} | "
            f"{'✓' if improved else ''} [{elapsed:.1f}s]"
        )

        if es.should_stop:
            print(f"[Train] Early stopping en epoch {epoch}.")
            break

    print(f"[Train] Mejor val Macro-F1: {best_f1:.4f}  →  {ckpt_path}")
    return str(ckpt_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Entrenar LSMTransformer")

    # Datos
    p.add_argument("--parquet",  required=True)
    p.add_argument("--output",   default="runs/exp01")
    p.add_argument("--workers",  type=int,   default=1)
    p.add_argument("--seed",     type=int,   default=42)

    # Modelo
    p.add_argument("--d_model",        type=int,   default=256)
    p.add_argument("--nhead",          type=int,   default=4)
    p.add_argument("--num_layers",     type=int,   default=3)
    p.add_argument("--dim_ff",         type=int,   default=512)
    p.add_argument("--dropout",        type=float, default=0.2)
    p.add_argument("--trigger_window", type=int,   default=30)
    p.add_argument("--use_eadm",       action="store_true", default=True)

    # Entrenamiento
    p.add_argument("--epochs",        type=int,   default=150)
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--clip_norm",     type=float, default=1.0)
    p.add_argument("--patience",      type=int,   default=30)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--aimclr_warmup", type=int,   default=20)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--amp",           action="store_true", default=True)

    # Pesos de pérdida
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta",  type=float, default=0.1)
    p.add_argument("--lam",   type=float, default=0.1)
    p.add_argument("--tau",   type=float, default=0.5)

    # reemplaza la línea de split_by_speaker si existe, o agrégala
    p.add_argument("--no_split_by_speaker", action="store_false", dest="split_by_speaker", default=True)
    return p.parse_args()


if __name__ == "__main__":
    cfg = _parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train(cfg)