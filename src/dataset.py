"""
src/dataset.py
──────────────────────────────────────────────────────────────────────────────
Loader de landmarks LSM para PyTorch + splits robustos sin data leakage.

Contenido:
  - LSMDataset              : torch.utils.data.Dataset que lee el parquet y
                              devuelve tensores (keypoints, label, mask).
  - collate_fn              : agrupa muestras de distinta duración (T variable)
                              con padding y máscara de atención.
  - inspect_signer_inference: diagnóstico — verifica si el prefijo de 2 dígitos
                              del campo 'intento' identifica al señador.
  - make_splits             : genera splits train/val/test robustos con dos
                              estrategias: "by_signer" o "stratified".

Convención del dataset (confirmada):
    intento = "{señador_id:02d}{glosa_id:03d}"
    Ej: "01001" → señador 01, glosa 001
    → df["intento"].str[:2]  extrae el id de señador de forma fiable.

Uso rápido:
    from src.dataset import LSMDataset, collate_fn, make_splits

    train_ids, val_ids, test_ids = make_splits(
        parquet_path="corpus_LSM_esp/lsm_dataset.parquet",
        strategy="by_signer",   # o "stratified"
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_ds = LSMDataset("corpus_LSM_esp/lsm_dataset.parquet", video_ids=train_ids)
    val_ds   = LSMDataset("corpus_LSM_esp/lsm_dataset.parquet", video_ids=val_ids)

    train_loader = DataLoader(train_ds, batch_size=16,
                              shuffle=True, collate_fn=collate_fn)

Uso como script (diagnóstico + splits):
    python src/dataset.py --parquet corpus_LSM_esp/lsm_dataset.parquet --diagnose
    python src/dataset.py --parquet corpus_LSM_esp/lsm_dataset.parquet --save splits.json
"""

from __future__ import annotations

import io
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
#  Constantes
# ──────────────────────────────────────────────────────────────────────────────
N_KPT  = 133   # keypoints COCO-WholeBody
N_FEAT = 2     # x, y  (score descartado — no es [0,1] en RTMPose)


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers de deserialización
# ──────────────────────────────────────────────────────────────────────────────
def _bytes_to_array(b: bytes) -> np.ndarray:
    """Deserializa bytes guardados con np.save → ndarray."""
    return np.load(io.BytesIO(b))


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────────────
class LSMDataset(Dataset):
    """
    Dataset de landmarks LSM a partir de un archivo Parquet.

    Cada muestra devuelve un dict con:
        "keypoints"  : FloatTensor  (T, 133, 3)   — x, y, score por frame
        "label"      : LongTensor   ()             — índice de clase (glosa)
        "video_id"   : str                         — identificador del video
        "T"          : int                         — número real de frames

    Parámetros
    ----------
    parquet_path : str | Path
        Ruta al archivo Parquet generado por build_lsm_dataset.py.
    video_ids : list[str] | None
        Si se proporciona, filtra el dataset a esos video_ids (útil para splits).
        Si es None, usa todos los registros.
    normalize : bool
        Si True, normaliza x e y al rango [0, 1] usando width/height del video.
        El score (canal 2) no se modifica.
    min_detection_ratio : float
        Descarta muestras donde la fracción de frames con persona detectada
        sea menor a este umbral. Default 0.0 (sin filtro).
    label_col : str
        Columna del parquet que contiene la etiqueta de clase. Default "glosa".
    """

    def __init__(
        self,
        parquet_path: str | Path,
        video_ids: list[str] | None = None,
        normalize: bool = True,
        min_detection_ratio: float = 0.0,
        label_col: str = "glosa",
        augment=None,
    ):
        """
        augment : instancia de Compose (o cualquier callable (T,133,2)→(T,133,2))
                  Si se pasa, se aplica on-the-fly en __getitem__ solo sobre x,y.
                  Para val/test dejar en None.
        """
        self.parquet_path = Path(parquet_path)
        self.normalize    = normalize
        self.label_col    = label_col
        self.augment      = augment

        # ── Cargar tabla ──────────────────────────────────────
        df = pd.read_parquet(self.parquet_path)

        # Filtrar por video_ids si se proveen
        if video_ids is not None:
            df = df[df["video_id"].isin(video_ids)].reset_index(drop=True)

        # Filtrar por calidad de detección
        if min_detection_ratio > 0.0:
            df = self._filter_by_detection(df, min_detection_ratio)

        self.df = df.reset_index(drop=True)

        # ── Construir mapa label → índice entero ──────────────
        labels_sorted = sorted(self.df[label_col].unique())
        self.label2idx: dict[str, int] = {lbl: i for i, lbl in enumerate(labels_sorted)}
        self.idx2label: list[str]      = labels_sorted

        print(
            f"LSMDataset cargado: {len(self.df)} muestras | "
            f"{len(self.label2idx)} clases | "
            f"normalize={normalize} | augment={'on' if augment else 'off'}"
        )

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def _filter_by_detection(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Elimina filas donde muy pocos frames tienen persona detectada."""
        def detection_ratio(row):
            det = _bytes_to_array(row["person_detected"])  # bool (T,)
            return det.mean()

        ratios = df.apply(detection_ratio, axis=1)
        before = len(df)
        df = df[ratios >= threshold]
        print(f"  Filtro detección ({threshold:.0%}): {before - len(df)} muestras eliminadas")
        return df

    # ── Dataset API ───────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Deserializar keypoints → (T, 133, 3)
        kpts = _bytes_to_array(row["keypoints"]).astype(np.float32)  # (T, 133, 3)

        # Normalizar coordenadas x, y al rango [0, 1]
        if self.normalize:
            w = float(row["width"])
            h = float(row["height"])
            if w > 0 and h > 0:
                kpts[:, :, 0] /= w   # x
                kpts[:, :, 1] /= h   # y

        # Extraer solo x, y → (T, 133, 2)
        # El canal score no es [0,1] en RTMPose — se maneja en el modelo
        xy = kpts[:, :, :2]

        # Aplicar augmentaciones on-the-fly (solo en train)
        if self.augment is not None:
            xy = self.augment(xy)

        label_str = row[self.label_col]
        label_idx = self.label2idx[label_str]

        return {
            "keypoints": torch.from_numpy(xy.astype(np.float32)),  # (T, 133, 2)
            "label":     torch.tensor(label_idx, dtype=torch.long),
            "video_id":  row["video_id"],
            "T":         xy.shape[0],
        }

    @property
    def num_classes(self) -> int:
        return len(self.label2idx)


# ──────────────────────────────────────────────────────────────────────────────
#  collate_fn  —  padding a longitud máxima del batch
# ──────────────────────────────────────────────────────────────────────────────
def collate_fn(batch: list[dict]) -> dict:
    """
    Agrupa muestras de longitud variable con padding por ceros.

    Entrada:
        lista de dicts con keys: keypoints (T_i, 133, 3), label, video_id, T

    Salida (dict):
        keypoints  : FloatTensor  (B, T_max, 133, 3)
        labels     : LongTensor   (B,)
        padding_mask : BoolTensor  (B, T_max)
            True  → posición válida (no padding)
            False → posición de relleno
        video_ids  : list[str]
        lengths    : list[int]     duración real de cada muestra
    """
    lengths   = [item["T"] for item in batch]
    T_max     = max(lengths)
    B         = len(batch)

    kpts_pad  = torch.zeros(B, T_max, N_KPT, N_FEAT, dtype=torch.float32)
    mask      = torch.zeros(B, T_max, dtype=torch.bool)   # False = padding
    labels    = torch.stack([item["label"] for item in batch])
    video_ids = [item["video_id"] for item in batch]

    for i, item in enumerate(batch):
        t = item["T"]
        kpts_pad[i, :t] = item["keypoints"]
        mask[i, :t]     = True   # frames reales → True

    return {
        "keypoints":    kpts_pad,     # (B, T_max, 133, 3)
        "labels":       labels,       # (B,)
        "padding_mask": mask,         # (B, T_max)  True = válido
        "video_ids":    video_ids,
        "lengths":      lengths,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  inspect_signer_inference  —  diagnóstico del campo 'intento'
# ──────────────────────────────────────────────────────────────────────────────
def inspect_signer_inference(parquet_path: str | Path) -> dict:
    """
    Analiza el campo 'intento' y verifica si el prefijo de 2 dígitos identifica
    consistentemente al señador.

    Convención esperada: intento = "{señador:02d}{glosa:03d}"
    Ej: "01001" → señador 01 / glosa 001

    Retorna un dict con estadísticas y una recomendación sobre si usar
    by_signer es viable.
    """
    df = pd.read_parquet(parquet_path, columns=["glosa", "intento", "video_id"])
    df["signer"] = df["intento"].str[:2]

    n_videos   = len(df)
    n_glosas   = df["glosa"].nunique()
    n_intentos = df["intento"].nunique()
    n_signers  = df["signer"].nunique()

    intentos_por_signer = (
        df.groupby("signer")["intento"]
        .nunique()
        .describe()
        .round(2)
        .to_dict()
    )

    # Heurística adaptada a la convención del dataset:
    #   intento = "{señador:02d}{glosa:03d}"  →  1 intento por señador por glosa.
    # En este caso ratio_multi siempre es ~0, pero el prefijo sí identifica
    # al señador. La verdadera señal de viabilidad es:
    #   - Hay ≥3 señadores distintos, Y
    #   - Cada señador cubre una fracción razonable de las glosas
    #     (mean intentos_por_signer ≥ 0.5 * n_glosas)
    mean_intentos  = intentos_por_signer.get("mean", 0)
    coverage_ratio = mean_intentos / n_glosas if n_glosas > 0 else 0

    # ratio_multi se mantiene por compatibilidad pero ya no es el criterio
    per_glosa    = df.groupby("glosa").agg(n_signers=("signer", "nunique"))
    glosas_multi = per_glosa[per_glosa["n_signers"] > 1]
    ratio_multi  = len(glosas_multi) / len(per_glosa)

    if n_signers <= 2:
        recommendation = (
            "⚠  Solo se detectan ≤2 señadores distintos. "
            "Usa 'stratified' en su lugar."
        )
        viable = False
    elif coverage_ratio >= 0.5:
        recommendation = (
            f"✓  {n_signers} señadores detectados, cada uno cubre en promedio "
            f"{mean_intentos:.0f}/{n_glosas} glosas ({coverage_ratio:.0%}). "
            f"'by_signer' es viable."
        )
        viable = True
    else:
        recommendation = (
            f"⚠  Los señadores cubren en promedio solo el {coverage_ratio:.0%} "
            f"de las glosas. Puede haber clases sin representación en algún split."
        )
        viable = False

    return {
        "n_videos":               n_videos,
        "n_glosas":               n_glosas,
        "n_intentos_unicos":      n_intentos,
        "n_signers_unicos":       n_signers,
        "glosas_multi_por_signer": len(glosas_multi),
        "ratio_multi":            round(ratio_multi, 3),
        "intentos_por_signer":    intentos_por_signer,
        "by_signer_viable":       viable,
        "recommendation":         recommendation,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  make_splits  —  genera índices train / val / test
# ──────────────────────────────────────────────────────────────────────────────
def make_splits(
    parquet_path: str | Path,
    strategy: Literal["by_signer", "stratified"] = "by_signer",
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    label_col: str = "glosa",
    verbose: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """
    Genera splits train / val / test sin data leakage.

    Estrategias
    -----------
    "by_signer"
        Separa señadores completos entre splits usando el prefijo de 2 dígitos
        del campo 'intento' (ej. "01001" → señador "01").
        Ningún señador del test aparece en train → evaluación realista.
        Si se detectan ≤2 señadores, cae automáticamente a "stratified".

    "stratified"
        Reparto aleatorio proporcional por clase (glosa).
        Un mismo señador puede aparecer en train y test (más optimista).
        Útil como baseline rápido.

    Retorna
    -------
    (train_ids, val_ids, test_ids)  — listas de video_id strings.
    """
    assert 0 < val_ratio + test_ratio < 1, "val_ratio + test_ratio debe ser < 1"

    rng = random.Random(seed)
    df  = pd.read_parquet(parquet_path, columns=["video_id", label_col, "intento"])
    df["signer"] = df["intento"].str[:2]

    # ── Estrategia by_signer ──────────────────────────────────
    if strategy == "by_signer":
        unique_signers = sorted(df["signer"].unique())

        if len(unique_signers) <= 2:
            print(
                "  ⚠  Solo se detectaron ≤2 señadores distintos. "
                "Cayendo a estrategia 'stratified'."
            )
            strategy = "stratified"
        else:
            rng.shuffle(unique_signers)
            n      = len(unique_signers)
            n_test = max(1, round(n * test_ratio))
            n_val  = max(1, round(n * val_ratio))

            test_signers  = set(unique_signers[:n_test])
            val_signers   = set(unique_signers[n_test : n_test + n_val])
            train_signers = set(unique_signers[n_test + n_val :])

            train_ids = df[df["signer"].isin(train_signers)]["video_id"].tolist()
            val_ids   = df[df["signer"].isin(val_signers)]  ["video_id"].tolist()
            test_ids  = df[df["signer"].isin(test_signers)] ["video_id"].tolist()

            if verbose:
                _print_split_report(df, train_ids, val_ids, test_ids,
                                    label_col, "by_signer")
            return train_ids, val_ids, test_ids

    # ── Estrategia stratified ─────────────────────────────────
    class_to_ids: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        class_to_ids[row[label_col]].append(row["video_id"])

    train_ids, val_ids, test_ids = [], [], []

    for ids in class_to_ids.values():
        rng.shuffle(ids)
        n = len(ids)

        if n < 3:
            train_ids.extend(ids)
            continue

        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))

        test_ids .extend(ids[:n_test])
        val_ids  .extend(ids[n_test : n_test + n_val])
        train_ids.extend(ids[n_test + n_val :])

    if verbose:
        _print_split_report(df, train_ids, val_ids, test_ids,
                            label_col, "stratified")
    return train_ids, val_ids, test_ids


# ──────────────────────────────────────────────────────────────────────────────
#  _print_split_report  —  reporte con overlap, clases y señadores
# ──────────────────────────────────────────────────────────────────────────────
def _print_split_report(
    df: pd.DataFrame,
    train_ids: list[str],
    val_ids:   list[str],
    test_ids:  list[str],
    label_col: str,
    strategy:  str,
) -> None:
    total = len(train_ids) + len(val_ids) + len(test_ids)
    sep   = "─" * 55
    print(f"\n{sep}")
    print(f"Split: {strategy}")
    print(sep)
    print(f"  Total  : {total}")
    print(f"  Train  : {len(train_ids):>5}  ({len(train_ids)/total:.1%})")
    print(f"  Val    : {len(val_ids):>5}  ({len(val_ids)/total:.1%})")
    print(f"  Test   : {len(test_ids):>5}  ({len(test_ids)/total:.1%})")

    # ── Overlaps de video_id ──────────────────────────────────
    s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
    overlaps = {
        "train∩val":  len(s_tr & s_va),
        "train∩test": len(s_tr & s_te),
        "val∩test":   len(s_va & s_te),
    }
    if any(overlaps.values()):
        print(f"\n  ⚠  OVERLAPS de video_id: {overlaps}")
    else:
        print(f"\n  ✓  Sin overlaps de video_id entre splits")

    # ── Cobertura de clases ───────────────────────────────────
    def classes_in(ids):
        return set(df[df["video_id"].isin(ids)][label_col])

    c_train, c_val, c_test = classes_in(train_ids), classes_in(val_ids), classes_in(test_ids)
    print(f"\n  Clases en train : {len(c_train)}")
    print(f"  Clases en val   : {len(c_val)}")
    print(f"  Clases en test  : {len(c_test)}")
    missing_test = c_train - c_test
    if missing_test:
        print(f"  ⚠  {len(missing_test)} clases de train ausentes en test "
              f"(esperable en by_signer con pocas muestras por señador)")

    # ── Señadores por split ───────────────────────────────────
    if "signer" in df.columns:
        def signers_in(ids):
            return set(df[df["video_id"].isin(ids)]["signer"])

        sg_tr = signers_in(train_ids)
        sg_va = signers_in(val_ids)
        sg_te = signers_in(test_ids)
        print(f"\n  Señadores en train : {sorted(sg_tr)}")
        print(f"  Señadores en val   : {sorted(sg_va)}")
        print(f"  Señadores en test  : {sorted(sg_te)}")

        shared = (sg_tr & sg_te) | (sg_tr & sg_va) | (sg_va & sg_te)
        if shared:
            print(f"  ⚠  Señadores compartidos entre splits: {shared}")
        else:
            print(f"  ✓  Ningún señador compartido entre splits")

    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
#  Script standalone: diagnóstico + splits + DataLoader de prueba
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    p = argparse.ArgumentParser(
        description="Diagnostica señadores, genera splits e inspecciona el dataset LSM."
    )
    p.add_argument("--parquet",    "-p", required=True,
                   help="Ruta al Parquet (ej. corpus_LSM_esp/lsm_dataset.parquet)")
    p.add_argument("--strategy",   "-s", default="by_signer",
                   choices=["by_signer", "stratified"])
    p.add_argument("--val-ratio",  type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--diagnose",   action="store_true",
                   help="Solo muestra el diagnóstico de señadores, sin generar splits")
    p.add_argument("--save",       metavar="FILE",
                   help="Guarda los splits en un JSON (ej. splits.json)")
    args = p.parse_args()

    # ── Diagnóstico ───────────────────────────────────────────
    print("\n=== Diagnóstico de señadores ===")
    info = inspect_signer_inference(args.parquet)
    for k, v in info.items():
        if k != "intentos_por_signer":
            print(f"  {k:<40}: {v}")
    print(f"  {'intentos_por_signer':<40}: {info['intentos_por_signer']}")

    if args.diagnose:
        raise SystemExit(0)

    # ── Generar splits ────────────────────────────────────────
    strategy = args.strategy
    if strategy == "by_signer" and not info["by_signer_viable"]:
        print("\n⚠  by_signer no parece viable. Considera --strategy stratified\n")

    print(f"\n=== Generando splits (strategy={strategy}) ===")
    train_ids, val_ids, test_ids = make_splits(
        parquet_path=args.parquet,
        strategy=strategy,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # ── Guardar JSON ──────────────────────────────────────────
    if args.save:
        out = {
            "strategy":   strategy,
            "val_ratio":  args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed":       args.seed,
            "train":      train_ids,
            "val":        val_ids,
            "test":       test_ids,
        }
        Path(args.save).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nSplits guardados en: {args.save}")

    # ── DataLoader de prueba ──────────────────────────────────
    print("\n=== Cargando LSMDataset (train) ===")
    ds     = LSMDataset(args.parquet, video_ids=train_ids)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_fn)

    print(f"Num classes: {ds.num_classes}")
    print("Ejemplo batch:")
    for batch in loader:
        print(f"  keypoints shape : {batch['keypoints'].shape}")
        print(f"  labels          : {batch['labels']}")
        print(f"  padding_mask    : {batch['padding_mask'].shape}")
        print(f"  lengths         : {batch['lengths']}")
        break