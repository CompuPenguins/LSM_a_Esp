"""
UMAP de espacio de landmarks (LSM)
=================================

Proyecta cada video a un vector compacto y calcula una proyección 2D con UMAP.
Útil para inspeccionar cómo se distribuyen las glosas en el espacio latente.

Entrada esperada:
  parquet con columnas:
    - glosa (str)
    - keypoints (bytes -> np.float32 (T, 133, 3))
    - person_detected (bytes -> np.bool_ (T,))

Salida:
  - umap_embeddings.csv
  - umap_scatter_by_glosa.png
  - umap_density.png

Uso:
  python umap_landmark_space.py \
      --parquet corpus_LSM_esp/lsm_dataset.parquet \
      --outdir reports/umap \
      --max-samples 3000
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import umap


# Igual que backend actual: cuerpo (0-16) + manos (91-132)
KEEP_IDX = list(range(0, 17)) + list(range(91, 133))  # 59 landmarks


def bytes_to_array(blob: bytes, dtype=None) -> np.ndarray:
    arr = np.load(io.BytesIO(blob), allow_pickle=False)
    if dtype is not None:
        return arr.astype(dtype)
    return arr


def preprocess_xy(keypoints: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """
    keypoints: (T,133,3), detected: (T,)
    return: (T,59,2)
    """
    kpts = keypoints[:, KEEP_IDX, :].astype(np.float32)  # (T,59,3)
    xy = kpts[:, :, :2].copy()
    score = kpts[:, :, 2]

    # low confidence
    xy[score < 0.3] = 0.0

    # center by shoulder midpoint
    shoulder_mid = (xy[:, 5, :] + xy[:, 6, :]) / 2.0
    xy -= shoulder_mid[:, None, :]

    # scale by shoulder-hip distance
    hip_mid = (xy[:, 11, :] + xy[:, 12, :]) / 2.0
    scale = np.linalg.norm(hip_mid, axis=-1)
    scale = np.clip(scale, 1e-8, None)
    xy /= scale[:, None, None]

    # remove frames without person
    if detected.shape[0] == xy.shape[0]:
        xy[~detected] = 0.0

    return xy


def temporal_pool_features(xy: np.ndarray) -> np.ndarray:
    """
    Convierte (T,59,2) -> vector por video.
    Usamos mean/std temporal por coordenada para mantener tamaño manejable.
    salida: (59*2*2,) = 236
    """
    flat = xy.reshape(xy.shape[0], -1)  # (T,118)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Proyección UMAP de landmarks LSM")
    ap.add_argument("--parquet", required=True, help="Ruta parquet dataset")
    ap.add_argument("--outdir", default="reports/umap", help="Directorio de salida")
    ap.add_argument("--max-samples", type=int, default=3000, help="Máximo de videos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--metric", default="cosine")
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not parquet_path.exists():
        raise FileNotFoundError(f"No existe parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "glosa" not in df.columns or "keypoints" not in df.columns or "person_detected" not in df.columns:
        raise ValueError("El parquet no tiene columnas requeridas: glosa, keypoints, person_detected")

    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)

    feats = []
    glosas = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Construyendo features"):
        try:
            kpts = bytes_to_array(row["keypoints"], np.float32)       # (T,133,3)
            det = bytes_to_array(row["person_detected"], np.bool_)    # (T,)
            xy = preprocess_xy(kpts, det)                              # (T,59,2)
            feat = temporal_pool_features(xy)                          # (236,)
            feats.append(feat)
            glosas.append(str(row["glosa"]))
        except Exception:
            # Salta filas corruptas o no parseables
            continue

    if not feats:
        raise RuntimeError("No se pudo construir ninguna feature válida.")

    X = np.stack(feats, axis=0)
    y = np.array(glosas)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    emb = reducer.fit_transform(X)

    le = LabelEncoder()
    y_id = le.fit_transform(y)

    out_csv = outdir / "umap_embeddings.csv"
    out_df = pd.DataFrame({
        "umap_x": emb[:, 0],
        "umap_y": emb[:, 1],
        "glosa": y,
        "glosa_id": y_id,
    })
    out_df.to_csv(out_csv, index=False)

    # Scatter por glosa_id (sin leyenda, muchas clases)
    plt.figure(figsize=(10, 8))
    plt.scatter(emb[:, 0], emb[:, 1], c=y_id, s=8, alpha=0.7, cmap="tab20")
    plt.title("UMAP del espacio de landmarks (color = glosa_id)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    out_scatter = outdir / "umap_scatter_by_glosa.png"
    plt.savefig(out_scatter, dpi=180)
    plt.close()

    # Densidad global
    plt.figure(figsize=(10, 8))
    plt.hexbin(emb[:, 0], emb[:, 1], gridsize=45, cmap="magma", mincnt=1)
    plt.colorbar(label="# muestras")
    plt.title("Densidad UMAP (todas las glosas)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    out_density = outdir / "umap_density.png"
    plt.savefig(out_density, dpi=180)
    plt.close()

    print("\n✅ UMAP completado")
    print(f"   Muestras: {len(out_df)}")
    print(f"   Clases:   {len(le.classes_)}")
    print(f"   CSV:      {out_csv}")
    print(f"   Scatter:  {out_scatter}")
    print(f"   Densidad: {out_density}")


if __name__ == "__main__":
    main()
