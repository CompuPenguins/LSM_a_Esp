"""
src/test_inspection.py
======================
Evaluación detallada del modelo entrenado sobre el test set.

Muestra:
  - Métricas globales (Top-1, Top-3, Macro-F1)
  - Predicciones muestra por muestra con confianzas
  - Las glosas donde el modelo más se equivoca
  - Distribución de confianzas en aciertos vs errores
  - Top-K accuracy para saber si la respuesta correcta
    está en las primeras K predicciones

Uso:
    python src/test_inspection.py \
        --parquet corpus_LSM_esp/lsm_dataset.parquet \
        --ckpt    runs/exp06/best_model.pt

    # Ver solo las primeras 30 predicciones:
    python src/test_inspection.py --parquet ... --ckpt ... --n_show 30

    # Ver solo los errores:
    python src/test_inspection.py --parquet ... --ckpt ... --only_errors
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from dataset import split_dataset, collate_fn, TMAX, INPUT_DIM
from model   import LSMTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Carga del modelo
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, num_classes, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("cfg", {})

    model = LSMTransformer(
        num_classes     = num_classes,
        input_dim       = INPUT_DIM,
        d_model         = cfg.get("d_model", 256),
        nhead           = cfg.get("nhead", 4),
        num_layers      = cfg.get("num_layers", 3),
        dim_feedforward = cfg.get("dim_ff", 512),
        dropout         = 0.0,   # sin dropout en inferencia
        tmax            = TMAX,
        use_eadm        = False,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_f1 = ckpt.get("val_f1", "?")
    epoch  = ckpt.get("epoch",  "?")
    print(f"Modelo cargado: época {epoch} | val F1 (entrenamiento) = {val_f1:.4f}")
    return model, ckpt.get("glosa2idx", {})


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia sobre el test set completo
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, test_ds, device, batch_size=64):
    """
    Corre el modelo sobre todo el test set.
    Retorna listas paralelas de resultados por muestra.
    """
    loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    all_video_ids = []
    all_labels    = []
    all_preds     = []
    all_probs     = []   # (N, num_classes)

    for batch in loader:
        kpts   = batch["keypoints"].to(device)
        mask   = batch["valid_mask"].to(device)
        labels = batch["label"]

        logits, _ = model(kpts, mask)
        probs     = F.softmax(logits, dim=-1).cpu()

        all_video_ids.extend(batch["video_id"])
        all_labels.extend(labels.tolist())
        all_probs.append(probs)

    all_probs  = torch.cat(all_probs, dim=0).numpy()   # (N, num_classes)
    all_preds  = all_probs.argmax(axis=-1)
    all_labels = np.array(all_labels)

    return all_video_ids, all_labels, all_preds, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# Métricas globales
# ─────────────────────────────────────────────────────────────────────────────

def print_global_metrics(labels, preds, probs, num_classes, idx2glosa):
    from sklearn.metrics import f1_score, top_k_accuracy_score

    top1 = (preds == labels).mean()
    top3 = top_k_accuracy_score(labels, probs, k=3,  labels=list(range(num_classes)))
    top5 = top_k_accuracy_score(labels, probs, k=5,  labels=list(range(num_classes)))
    f1   = f1_score(labels, preds, average="macro",
                    labels=list(range(num_classes)), zero_division=0)

    # Confianza promedio en aciertos vs errores
    correct_mask = (preds == labels)
    conf_correct = probs[correct_mask,  preds[correct_mask]].mean()  if correct_mask.any()  else 0
    conf_wrong   = probs[~correct_mask, preds[~correct_mask]].mean() if (~correct_mask).any() else 0

    print("=" * 55)
    print("  MÉTRICAS GLOBALES — TEST SET")
    print("=" * 55)
    print(f"  Muestras evaluadas : {len(labels)}")
    print(f"  Clases             : {num_classes}")
    print(f"  Top-1 Accuracy     : {top1:.4f}  ({int(top1*len(labels))}/{len(labels)} correctas)")
    print(f"  Top-3 Accuracy     : {top3:.4f}  (respuesta en top 3)")
    print(f"  Top-5 Accuracy     : {top5:.4f}  (respuesta en top 5)")
    print(f"  Macro F1           : {f1:.4f}")
    print(f"  Confianza aciertos : {conf_correct:.3f}  (promedio)")
    print(f"  Confianza errores  : {conf_wrong:.3f}  (promedio)")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Predicciones muestra a muestra
# ─────────────────────────────────────────────────────────────────────────────

def print_predictions(video_ids, labels, preds, probs, idx2glosa,
                      n_show=50, only_errors=False, top_k=5):
    """
    Imprime predicción por muestra con las top_k glosas y sus confianzas.
    """
    print(f"\n{'─'*75}")
    print(f"  PREDICCIONES {'(solo errores)' if only_errors else f'(primeras {n_show})'}")
    print(f"{'─'*75}")
    print(f"  {'Video':<15} {'Real':<22} {'Predicho':<22} {'Conf':>6}  {'OK'}")
    print(f"  {'─'*13} {'─'*20} {'─'*20} {'─'*6}  {'─'*4}")

    shown = 0
    for i, (vid, true_lbl, pred_lbl) in enumerate(zip(video_ids, labels, preds)):
        correct = (true_lbl == pred_lbl)
        if only_errors and correct:
            continue

        true_glosa = idx2glosa.get(int(true_lbl), str(true_lbl))
        pred_glosa = idx2glosa.get(int(pred_lbl), str(pred_lbl))
        conf       = probs[i, pred_lbl]
        ok         = "✓" if correct else "✗"

        print(f"  {str(vid):<15} {true_glosa:<22} {pred_glosa:<22} {conf:>6.3f}  {ok}")

        # Mostrar top_k si es un error
        if not correct:
            top_idx  = np.argsort(probs[i])[::-1][:top_k]
            top_line = "  " + " " * 15 + "  top-k: " + " | ".join(
                f"{idx2glosa.get(int(j), str(j))}({probs[i,j]:.2f})"
                for j in top_idx
            )
            print(top_line)

        shown += 1
        if shown >= n_show:
            break

    if shown == 0:
        print("  (sin errores en las muestras mostradas)" if only_errors else "  (sin muestras)")


# ─────────────────────────────────────────────────────────────────────────────
# Glosas con peor accuracy
# ─────────────────────────────────────────────────────────────────────────────

def print_worst_glosas(labels, preds, probs, idx2glosa, n=20):
    from collections import Counter

    print(f"\n{'─'*55}")
    print(f"  TOP {n} GLOSAS CON PEOR ACCURACY")
    print(f"{'─'*55}")
    print(f"  {'Glosa':<25} {'Acc':>6} {'N':>4} {'Conf_media':>10}")
    print(f"  {'─'*23} {'─'*6} {'─'*4} {'─'*10}")

    per_class = {}
    for cls in np.unique(labels):
        mask      = labels == cls
        acc       = (preds[mask] == cls).mean()
        n_samples = mask.sum()
        # Confianza media en la clase correcta (aunque no la prediga)
        conf_true = probs[mask, cls].mean()
        per_class[cls] = (acc, n_samples, conf_true)

    worst = sorted(per_class.items(), key=lambda x: (x[1][0], -x[1][1]))[:n]
    for cls, (acc, n_s, conf) in worst:
        glosa = idx2glosa.get(int(cls), str(cls))
        print(f"  {glosa:<25} {acc:>6.3f} {n_s:>4} {conf:>10.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Distribución de confianzas
# ─────────────────────────────────────────────────────────────────────────────

def print_confidence_distribution(labels, preds, probs):
    correct_mask = (preds == labels)
    confs_ok  = probs[correct_mask,  preds[correct_mask]]  if correct_mask.any()  else np.array([])
    confs_err = probs[~correct_mask, preds[~correct_mask]] if (~correct_mask).any() else np.array([])

    def bucket(confs, label):
        if len(confs) == 0:
            return
        buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        print(f"\n  Confianza ({label}, n={len(confs)}):")
        for lo, hi in buckets:
            n = ((confs >= lo) & (confs < hi)).sum()
            bar = "█" * int(n / len(confs) * 30)
            print(f"    [{lo:.1f}-{hi:.1f}): {n:>4}  {bar}")

    print(f"\n{'─'*55}")
    print(f"  DISTRIBUCIÓN DE CONFIANZAS")
    print(f"{'─'*55}")
    bucket(confs_ok,  "ACIERTOS")
    bucket(confs_err, "ERRORES")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",      required=True)
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--device",       default="auto")
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--n_show",       type=int, default=50,
                   help="Nº de predicciones a mostrar")
    p.add_argument("--only_errors",  action="store_true",
                   help="Mostrar solo predicciones incorrectas")
    p.add_argument("--top_k",        type=int, default=5,
                   help="Top-K a mostrar en cada error")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Cargar dataset (mismo split que entrenamiento) ────────────────────────
    _, _, test_ds = split_dataset(
        args.parquet,
        seed=args.seed,
        score_thresh=0.3,
    )
    num_classes = test_ds.num_classes
    idx2glosa   = test_ds.idx2glosa
    print(f"Test set: {len(test_ds)} muestras | {num_classes} clases")

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    model, glosa2idx_ckpt = load_model(args.ckpt, num_classes, device)

    # Si el checkpoint tiene su propio glosa2idx, usarlo
    if glosa2idx_ckpt:
        idx2glosa = {v: k for k, v in glosa2idx_ckpt.items()}

    # ── Inferencia ────────────────────────────────────────────────────────────
    print("Corriendo inferencia sobre test set...")
    video_ids, labels, preds, probs = run_inference(
        model, test_ds, device, batch_size=args.batch_size
    )

    # ── Mostrar resultados ────────────────────────────────────────────────────
    print_global_metrics(labels, preds, probs, num_classes, idx2glosa)
    print_predictions(video_ids, labels, preds, probs, idx2glosa,
                      n_show=args.n_show,
                      only_errors=args.only_errors,
                      top_k=args.top_k)
    print_worst_glosas(labels, preds, probs, idx2glosa, n=20)
    print_confidence_distribution(labels, preds, probs)
    print()


if __name__ == "__main__":
    main()