"""
src/hparam_search.py
====================
Búsqueda de hiperparámetros con Optuna para LSMTransformer.

Uso:
    python -m src.hparam_search \
        --parquet corpus_LSM_esp/lsm_dataset.parquet \
        --output  runs/hparam \
        --trials  40 \
        --epochs  40 \
        --device  cuda

Instalar Optuna si no está:
    pip install optuna
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    raise ImportError("Instala optuna: pip install optuna")

try:
    from src.dataset import collate_fn, split_dataset, TMAX, INPUT_DIM, N_LANDMARKS
    from src.augmentations import LandmarkAugmenter, AimCLRViewGenerator
    from src.model import LSMTransformer
    from src.train import D3MLoss, MetricTracker, EarlyStopping, make_aimclr_collate, train_step, val_epoch
except ModuleNotFoundError:
    from dataset import collate_fn, split_dataset, TMAX, INPUT_DIM, N_LANDMARKS
    from augmentations import LandmarkAugmenter, AimCLRViewGenerator
    from model import LSMTransformer
    from train import D3MLoss, MetricTracker, EarlyStopping, make_aimclr_collate, train_step, val_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Objetivo Optuna
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(cfg: argparse.Namespace):
    """
    Fabrica la función objetivo para Optuna.
    Pre-carga el dataset una sola vez para no re-leerlo en cada trial.
    """

    print("[HPSearch] Pre-cargando dataset…")
    augmenter = LandmarkAugmenter(p_build=0.5, p_hand=0.5, p_noise=0.7, p_jitter=0.3, p_speed=0.4)
    train_ds, val_ds, _ = split_dataset(
        cfg.parquet,
        train_ratio=0.70, val_ratio=0.15,
        seed=cfg.seed, augment_train=augmenter,
    )
    num_classes = train_ds.num_classes
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    print(f"[HPSearch] {len(train_ds)} train | {len(val_ds)} val | {num_classes} clases | device={device}")

    def objective(trial: Trial) -> float:

        # ── Espacio de búsqueda ───────────────────────────────────────────────
        d_model    = trial.suggest_categorical("d_model",    [128, 256, 384])
        nhead      = trial.suggest_categorical("nhead",      [2, 4, 8])
        num_layers = trial.suggest_int("num_layers", 2, 5)
        dim_ff     = trial.suggest_categorical("dim_ff",     [256, 512, 1024])
        dropout    = trial.suggest_float("dropout",    0.05, 0.35, step=0.05)
        lr         = trial.suggest_float("lr",         5e-5, 5e-3, log=True)
        lam        = trial.suggest_float("lam",        0.0,  0.3,  step=0.05)
        tau        = trial.suggest_float("tau",        0.2,  1.0,  step=0.1)
        beta       = trial.suggest_float("beta",       0.0,  0.3,  step=0.05)
        warmup_ep  = trial.suggest_int("warmup_epochs", 5, 15)
        aimclr_w   = trial.suggest_int("aimclr_warmup", 10, 30)

        # nhead debe dividir d_model
        if d_model % nhead != 0:
            raise optuna.exceptions.TrialPruned()

        # ── DataLoaders ───────────────────────────────────────────────────────
        view_gen = AimCLRViewGenerator(p_flip=0.5, p_axis=0.3, p_blur=0.5,
                                       blur_sigma=1.5, noise_sigma=0.02)
        train_collate = make_aimclr_collate(view_gen, tmax=TMAX)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,
                                  num_workers=1, collate_fn=train_collate,
                                  pin_memory=(device.type == "cuda"))
        val_loader   = DataLoader(val_ds, batch_size=cfg.batch * 2, shuffle=False,
                                  num_workers=1, collate_fn=collate_fn,
                                  pin_memory=(device.type == "cuda"))

        # ── Modelo ────────────────────────────────────────────────────────────
        model = LSMTransformer(
            num_classes=num_classes, input_dim=INPUT_DIM,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout,
            tmax=TMAX, trigger_window=30, use_eadm=True,
        ).to(device)

        # ── Pérdidas ──────────────────────────────────────────────────────────
        class_weights = train_ds.get_label_weights().to(device)
        ce_loss  = nn.CrossEntropyLoss(weight=class_weights)
        bce_loss = nn.BCEWithLogitsLoss()
        d3m_loss = D3MLoss(tau=tau)

        # ── Optimizador + scheduler con warmup ───────────────────────────────
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        def lr_lambda(epoch):
            if epoch < warmup_ep: v
                return (epoch + 1) / warmup_ep
            progress = (epoch - warmup_ep) / max(1, cfg.epochs - warmup_ep)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        es = EarlyStopping(patience=10)   # paciencia corta en búsqueda
        best_f1 = 0.0

        # ── Loop ──────────────────────────────────────────────────────────────
        for epoch in range(1, cfg.epochs + 1):
            lam_eff = lam * min(1.0, epoch / max(1, aimclr_w))

            model.train()
            for batch in train_loader:
                train_step(model, batch, optimizer, ce_loss, bce_loss, d3m_loss,
                           alpha=1.0, beta=beta, lam=lam_eff,
                           device=device, scaler=scaler, clip_norm=1.0)

            val_metrics = val_epoch(model, val_loader, ce_loss, device)
            scheduler.step()

            f1 = val_metrics["macro_f1"]
            if f1 > best_f1:
                best_f1 = f1

            # Pruning de Optuna: cortar trials que claramente no van a mejorar
            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if es.step(f1) is False and es.should_stop:
                break

        return best_f1

    return objective, num_classes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_search(cfg: argparse.Namespace):
    out_dir = Path(cfg.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    objective_fn, num_classes = make_objective(cfg)

    # Pruner: corta trials que a mitad de entrenamiento van muy por debajo del mejor
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="lsm_transformer_hparam",
        storage=f"sqlite:///{out_dir}/optuna.db",
        load_if_exists=True,   # permite retomar si se interrumpe
    )

    print(f"\n[HPSearch] Iniciando búsqueda: {cfg.trials} trials × {cfg.epochs} epochs\n")
    study.optimize(objective_fn, n_trials=cfg.trials, show_progress_bar=True)

    # ── Resultados ────────────────────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "="*60)
    print(f"MEJOR TRIAL #{best.number}  —  val Macro-F1: {best.value:.4f}")
    print("="*60)
    for k, v in best.params.items():
        print(f"  {k:20s}: {v}")

    # Guardar resultados en JSON
    results = {
        "best_trial":  best.number,
        "best_val_f1": best.value,
        "best_params": best.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ]
    }
    results_path = out_dir / "hparam_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en: {results_path}")

    # ── Comando listo para usar con los mejores hiperparámetros ───────────────
    p = best.params
    cmd = (
        f"\npython -m src.train \\\n"
        f"    --parquet {cfg.parquet} \\\n"
        f"    --output  runs/best_hparam \\\n"
        f"    --epochs  150 \\\n"
        f"    --batch   {cfg.batch} \\\n"
        f"    --device  {cfg.device} --amp \\\n"
        f"    --d_model {p['d_model']} --nhead {p['nhead']} "
        f"--num_layers {p['num_layers']} --dim_ff {p['dim_ff']} \\\n"
        f"    --dropout {p['dropout']} --lr {p['lr']:.2e} "
        f"--lam {p['lam']} --tau {p['tau']} --beta {p['beta']} \\\n"
        f"    --warmup_epochs {p['warmup_epochs']} "
        f"--aimclr_warmup {p['aimclr_warmup']} --patience 30"
    )
    print("\nComando para entrenamiento final:")
    print(cmd)

    # Guardar comando también
    (out_dir / "best_command.sh").write_text(f"#!/bin/bash\n{cmd}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Búsqueda de hiperparámetros LSMTransformer")
    p.add_argument("--parquet",  required=True)
    p.add_argument("--output",   default="runs/hparam")
    p.add_argument("--trials",   type=int, default=40,  help="Número de trials Optuna")
    p.add_argument("--epochs",   type=int, default=40,  help="Epochs por trial (corto)")
    p.add_argument("--batch",    type=int, default=32)
    p.add_argument("--device",   default="cuda")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    cfg = _parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run_search(cfg)