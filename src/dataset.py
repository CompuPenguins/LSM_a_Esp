# """
# src/dataset.py
# ==============
# Dataset, constantes globales, collate_fn y split_dataset para LSM.

# Constantes exportadas (usadas en train.py y augmentations.py):
#     TMAX        = 200   — longitud máxima de secuencia en frames
#     N_LANDMARKS = 133   — keypoints COCO-WholeBody
#     INPUT_DIM   = 266   — 133 × 2 (x, y — sin score)

# Funciones exportadas:
#     split_dataset(parquet_path, ...)  → (train_ds, val_ds, test_ds)
#     collate_fn(batch)                 → dict de tensores con padding dinámico

# Clase exportada:
#     LSMDataset
#         .num_classes          : int
#         .glosa2idx            : dict[str, int]
#         .idx2glosa            : dict[int, str]
#         .get_label_weights()  : Tensor (num_classes,)
# """

# from __future__ import annotations

# import io
# from pathlib import Path
# from typing import Callable, Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from torch import Tensor
# from torch.utils.data import Dataset


# # ─────────────────────────────────────────────────────────────────────────────
# # Constantes globales
# # ─────────────────────────────────────────────────────────────────────────────

# TMAX        : int = 200              # frames máximos por secuencia
# N_LANDMARKS : int = 133              # keypoints COCO-WholeBody
# INPUT_DIM   : int = N_LANDMARKS * 2  # 266  (x, y — score ignorado)


# # ─────────────────────────────────────────────────────────────────────────────
# # Preprocesamiento de keypoints
# # ─────────────────────────────────────────────────────────────────────────────

# def preprocess(
#     kpts: np.ndarray,           # (T, 133, 3)  float32
#     score_thresh: float = 0.3,
# ) -> np.ndarray:
#     """
#     1. Anula landmarks con score < umbral.
#     2. Centra respecto al punto medio de los hombros (kpts 5, 6).
#     3. Escala por distancia hombro → cadera (kpts 11, 12).

#     Returns: (T, 133, 2)  float32  — solo x, y normalizados
#     """
#     xy     = kpts[:, :, :2].copy()   # (T, 133, 2)
#     scores = kpts[:, :, 2]           # (T, 133)

#     # Anular landmarks poco confiables
#     low = scores < score_thresh
#     xy[low] = 0.0

#     # Centrado en hombros
#     shoulder_mid = (xy[:, 5, :] + xy[:, 6, :]) / 2.0   # (T, 2)
#     xy -= shoulder_mid[:, None, :]

#     # Escalado por altura torso
#     hip_mid = (xy[:, 11, :] + xy[:, 12, :]) / 2.0       # (T, 2)
#     scale   = np.linalg.norm(hip_mid, axis=-1)            # (T,)
#     scale   = np.clip(scale, 1e-8, None)
#     xy     /= scale[:, None, None]

#     return xy.astype(np.float32)                          # (T, 133, 2)


# # ─────────────────────────────────────────────────────────────────────────────
# # LSMDataset
# # ─────────────────────────────────────────────────────────────────────────────

# class LSMDataset(Dataset):
#     """
#     Carga secuencias de landmarks LSM desde un DataFrame ya filtrado al split.

#     Parquet schema esperado:
#         glosa          : str
#         intento        : int
#         video_id       : str
#         fps            : float
#         total_frames   : int
#         width, height  : int
#         keypoints      : bytes  →  np.ndarray float32 (T, 133, 3)
#         person_detected: bytes  →  np.ndarray bool    (T,)

#     Parámetros
#     ----------
#     df           : DataFrame del split correspondiente
#     glosa2idx    : dict str → int  (compartido entre train/val/test)
#     augmenter    : callable (T,133,3) → (T,133,3) ó None
#     score_thresh : umbral de confianza para filtrar landmarks
#     tmax         : longitud máxima de secuencia
#     return_aug   : si True, añade 'keypoints_aug' al item (para AimCLR)
#     view_gen     : callable (T,133,3) → (T,133,3) para la vista contrastiva
#     """

#     def __init__(
#         self,
#         df:           pd.DataFrame,
#         glosa2idx:    Dict[str, int],
#         augmenter:    Optional[Callable]  = None,
#         score_thresh: float               = 0.3,
#         tmax:         int                 = TMAX,
#         return_aug:   bool                = False,
#         view_gen:     Optional[Callable]  = None,
#     ):
#         self.df           = df.reset_index(drop=True)
#         self.glosa2idx    = glosa2idx
#         self.idx2glosa    = {v: k for k, v in glosa2idx.items()}
#         self.num_classes  = len(glosa2idx)
#         self.augmenter    = augmenter
#         self.score_thresh = score_thresh
#         self.tmax         = tmax
#         self.return_aug   = return_aug
#         self.view_gen     = view_gen

#     # ── utilidades ────────────────────────────────────────────────────────────

#     def _load_kpts(self, row) -> np.ndarray:
#         """Deserializa bytes → (T, 133, 3) float32."""
#         buf = io.BytesIO(row["keypoints"])
#         return np.load(buf)                              # (T, 133, 3)

#     def get_label_weights(self) -> Tensor:
#         """
#         Pesos inversos de frecuencia por clase para CrossEntropyLoss.
#         Clases más raras reciben mayor peso.
#         Returns: Tensor (num_classes,) float32
#         """
#         labels = self.df["glosa"].map(self.glosa2idx).values
#         counts = np.bincount(labels, minlength=self.num_classes).astype(np.float32)
#         counts = np.clip(counts, 1, None)
#         weights = 1.0 / counts
#         weights = weights / weights.sum() * self.num_classes   # escala a media=1
#         return torch.from_numpy(weights)

#     # ── Dataset API ───────────────────────────────────────────────────────────

#     def __len__(self) -> int:
#         return len(self.df)

#     def __getitem__(self, idx: int) -> Dict:
#         row      = self.df.iloc[idx]
#         kpts_raw = self._load_kpts(row)                  # (T, 133, 3)

#         # Aumentación anatómica sobre los datos crudos (antes de normalizar)
#         if self.augmenter is not None:
#             kpts_proc = self.augmenter(kpts_raw)
#         else:
#             kpts_proc = kpts_raw

#         # Preprocesar: normalizar y quitar score → (T, 133, 2)
#         xy    = preprocess(kpts_proc, self.score_thresh)  # (T, 133, 2)
#         T     = min(xy.shape[0], self.tmax)
#         xy_flat = xy[:T].reshape(T, INPUT_DIM)            # (T, 266)

#         label    = int(self.glosa2idx[row["glosa"]])
#         video_id = str(row.get("video_id", idx))

#         item = {
#             "keypoints": torch.from_numpy(xy_flat),      # (T, 266) longitud variable
#             "label":     torch.tensor(label, dtype=torch.long),
#             "video_id":  video_id,
#             "T":         T,
#         }

#         # Vista AimCLR: segunda vista extrema de los mismos datos crudos
#         if self.return_aug and self.view_gen is not None:
#             kpts_aug   = self.view_gen(kpts_raw)          # (T, 133, 3)
#             xy_aug     = preprocess(kpts_aug, self.score_thresh)
#             xy_aug_flat = xy_aug[:T].reshape(T, INPUT_DIM)
#             item["keypoints_aug"] = torch.from_numpy(xy_aug_flat)  # (T, 266)

#         return item


# # ─────────────────────────────────────────────────────────────────────────────
# # collate_fn — padding dinámico al máximo del batch
# # ─────────────────────────────────────────────────────────────────────────────

# def collate_fn(batch: List[Dict]) -> Dict:
#     """
#     Agrupa una lista de items en un batch con padding dinámico.
#     Rellena hasta la secuencia más larga del batch (no hasta TMAX global),
#     lo que ahorra memoria y cómputo en batches de secuencias cortas.

#     Returns dict con:
#         keypoints   : (B, T_max_batch, 266)  float32
#         valid_mask  : (B, T_max_batch)        bool  — True = frame real
#         label       : (B,)                    int64
#         video_id    : List[str]
#         keypoints_aug (si existe): (B, T_max_batch, 266)
#     """
#     T_max = max(item["T"] for item in batch)
#     B     = len(batch)

#     kpts_padded = torch.zeros(B, T_max, INPUT_DIM, dtype=torch.float32)
#     valid_mask  = torch.zeros(B, T_max, dtype=torch.bool)
#     labels      = torch.stack([item["label"] for item in batch])
#     video_ids   = [item["video_id"] for item in batch]

#     has_aug    = "keypoints_aug" in batch[0]
#     aug_padded = torch.zeros(B, T_max, INPUT_DIM, dtype=torch.float32) if has_aug else None

#     for i, item in enumerate(batch):
#         T = item["T"]
#         kpts_padded[i, :T] = item["keypoints"]
#         valid_mask[i,  :T] = True
#         if has_aug:
#             aug_padded[i, :T] = item["keypoints_aug"]

#     out = {
#         "keypoints":  kpts_padded,   # (B, T_max, 266)
#         "valid_mask": valid_mask,     # (B, T_max)
#         "label":      labels,         # (B,)
#         "video_id":   video_ids,
#     }
#     if has_aug:
#         out["keypoints_aug"] = aug_padded

#     return out


# # ─────────────────────────────────────────────────────────────────────────────
# # split_dataset — punto de entrada principal desde train.py
# # ─────────────────────────────────────────────────────────────────────────────

# def split_dataset(
#     parquet_path:  str,
#     train_ratio:   float              = 0.70,
#     val_ratio:     float              = 0.15,
#     seed:          int                = 42,
#     augment_train: Optional[Callable] = None,
#     score_thresh:  float              = 0.3,
#     tmax:          int                = TMAX,
#     view_gen:      Optional[Callable] = None,
# ) -> Tuple[LSMDataset, LSMDataset, LSMDataset]:
#     """
#     Carga el parquet, construye vocabulario de glosas y divide en
#     train / val / test con split estratificado por glosa.

#     Parámetros
#     ----------
#     parquet_path  : ruta al .parquet
#     train_ratio   : fracción train  (0.70)
#     val_ratio     : fracción val    (0.15)  → test = 1 - train - val
#     seed          : semilla
#     augment_train : LandmarkAugmenter ó None — solo aplicado en train
#     score_thresh  : umbral confianza landmarks
#     tmax          : longitud máxima de secuencia
#     view_gen      : AimCLRViewGenerator ó None
#                     Si se pasa, train incluye 'keypoints_aug' en cada item

#     Returns
#     -------
#     (train_ds, val_ds, test_ds)
#     """
#     print(f"[split_dataset] Cargando {parquet_path} …")
#     df = pd.read_parquet(parquet_path).reset_index(drop=True)

#     # ── Vocabulario ───────────────────────────────────────────────────────────
#     glosas_sorted = sorted(df["glosa"].unique().tolist())
#     glosa2idx     = {g: i for i, g in enumerate(glosas_sorted)}
#     num_classes   = len(glosa2idx)
#     print(f"[split_dataset] {len(df)} videos | {num_classes} glosas únicas")

#     # ── Split estratificado por glosa ─────────────────────────────────────────
#     rng = np.random.default_rng(seed)
#     train_idx, val_idx, test_idx = [], [], []

#     for glosa, group in df.groupby("glosa"):
#         idx = group.index.tolist()
#         rng.shuffle(idx)
#         n = len(idx)

#         if n == 1:
#             # Solo 1 muestra: va a train
#             train_idx.extend(idx)
#         elif n == 2:
#             train_idx.append(idx[0])
#             val_idx.append(idx[1])
#         else:
#             n_tr = max(1, int(round(n * train_ratio)))
#             n_va = max(1, int(round(n * val_ratio)))
#             n_te = max(0, n - n_tr - n_va)
#             # Ajustar si la suma supera n
#             while n_tr + n_va + n_te > n:
#                 if n_te > 0:
#                     n_te -= 1
#                 elif n_va > 1:
#                     n_va -= 1
#                 else:
#                     n_tr -= 1

#             train_idx.extend(idx[:n_tr])
#             val_idx.extend(idx[n_tr: n_tr + n_va])
#             test_idx.extend(idx[n_tr + n_va: n_tr + n_va + n_te])

#     df_train = df.loc[train_idx]
#     df_val   = df.loc[val_idx]
#     df_test  = df.loc[test_idx]

#     print(
#         f"[split_dataset] "
#         f"train={len(df_train)} | val={len(df_val)} | test={len(df_test)}"
#     )

#     # ── Instanciar datasets ───────────────────────────────────────────────────
#     train_ds = LSMDataset(
#         df=df_train,
#         glosa2idx=glosa2idx,
#         augmenter=augment_train,
#         score_thresh=score_thresh,
#         tmax=tmax,
#         return_aug=(view_gen is not None),
#         view_gen=view_gen,
#     )
#     val_ds = LSMDataset(
#         df=df_val,
#         glosa2idx=glosa2idx,
#         augmenter=None,
#         score_thresh=score_thresh,
#         tmax=tmax,
#         return_aug=False,
#     )
#     test_ds = LSMDataset(
#         df=df_test,
#         glosa2idx=glosa2idx,
#         augmenter=None,
#         score_thresh=score_thresh,
#         tmax=tmax,
#         return_aug=False,
#     )

#     return train_ds, val_ds, test_ds
"""
src/dataset.py
==============
Dataset, constantes globales, collate_fn y split_dataset para LSM.

Constantes exportadas (usadas en train.py y augmentations.py):
    TMAX        = 200   — longitud máxima de secuencia en frames
    N_LANDMARKS = 133   — keypoints COCO-WholeBody
    INPUT_DIM   = 266   — 133 × 2 (x, y — sin score)

Funciones exportadas:
    split_dataset(parquet_path, ...)  → (train_ds, val_ds, test_ds)
    collate_fn(batch)                 → dict de tensores con padding dinámico

Clase exportada:
    LSMDataset
        .num_classes          : int
        .glosa2idx            : dict[str, int]
        .idx2glosa            : dict[int, str]
        .get_label_weights()  : Tensor (num_classes,)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Constantes globales
# ─────────────────────────────────────────────────────────────────────────────

TMAX        : int = 200              # frames máximos por secuencia
# N_LANDMARKS : int = 133              # keypoints COCO-WholeBody
# INPUT_DIM   : int = N_LANDMARKS * 2  # 266  (x, y — score ignorado)

_KEEP_IDX   = list(range(0, 17)) + list(range(91, 133))   # 59 landmarks
N_LANDMARKS : int = len(_KEEP_IDX)   # 59
INPUT_DIM   : int = N_LANDMARKS * 2

# ─────────────────────────────────────────────────────────────────────────────
# Preprocesamiento de keypoints
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    kpts: np.ndarray,           # (T, 133, 3)  float32
    score_thresh: float = 0.3,
) -> np.ndarray:
    # ── NUEVO: filtrar solo los landmarks útiles ──────────────────────────────
    kpts = kpts[:, _KEEP_IDX, :]        # (T, 59, 3)
    # Los índices de hombros y caderas cambian tras el filtrado:
    # original 5,6 → new 5,6  (siguen siendo los mismos dentro de 0-16)
    # original 11,12 → new 11,12  (ídem)
    # ─────────────────────────────────────────────────────────────────────────

    xy     = kpts[:, :, :2].copy()      # (T, 59, 2)
    scores = kpts[:, :, 2]

    low = scores < score_thresh
    xy[low] = 0.0

    shoulder_mid = (xy[:, 5, :] + xy[:, 6, :]) / 2.0
    xy -= shoulder_mid[:, None, :]

    hip_mid = (xy[:, 11, :] + xy[:, 12, :]) / 2.0
    scale   = np.linalg.norm(hip_mid, axis=-1)
    scale   = np.clip(scale, 1e-8, None)
    xy     /= scale[:, None, None]

    return xy.astype(np.float32)        # (T, 59, 2)
    

# ─────────────────────────────────────────────────────────────────────────────
# LSMDataset
# ─────────────────────────────────────────────────────────────────────────────

class LSMDataset(Dataset):
    """
    Carga secuencias de landmarks LSM desde un DataFrame ya filtrado al split.

    Parquet schema esperado:
        glosa          : str
        intento        : int
        video_id       : str
        fps            : float
        total_frames   : int
        width, height  : int
        keypoints      : bytes  →  np.ndarray float32 (T, 133, 3)
        person_detected: bytes  →  np.ndarray bool    (T,)

    Parámetros
    ----------
    df           : DataFrame del split correspondiente
    glosa2idx    : dict str → int  (compartido entre train/val/test)
    augmenter    : callable (T,133,3) → (T,133,3) ó None
    score_thresh : umbral de confianza para filtrar landmarks
    tmax         : longitud máxima de secuencia
    return_aug   : si True, añade 'keypoints_aug' al item (para AimCLR)
    view_gen     : callable (T,133,3) → (T,133,3) para la vista contrastiva
    """

    def __init__(
        self,
        df:           pd.DataFrame,
        glosa2idx:    Dict[str, int],
        augmenter:    Optional[Callable]  = None,
        score_thresh: float               = 0.3,
        tmax:         int                 = TMAX,
        return_aug:   bool                = False,
        view_gen:     Optional[Callable]  = None,
    ):
        self.df           = df.reset_index(drop=True)
        self.glosa2idx    = glosa2idx
        self.idx2glosa    = {v: k for k, v in glosa2idx.items()}
        self.num_classes  = len(glosa2idx)
        self.augmenter    = augmenter
        self.score_thresh = score_thresh
        self.tmax         = tmax
        self.return_aug   = return_aug
        self.view_gen     = view_gen

    # ── utilidades ────────────────────────────────────────────────────────────

    def _load_kpts(self, row) -> np.ndarray:
        """Deserializa bytes → (T, 133, 3) float32."""
        buf = io.BytesIO(row["keypoints"])
        return np.load(buf)                              # (T, 133, 3)

    def get_label_weights(self) -> Tensor:
        """
        Pesos inversos de frecuencia por clase para CrossEntropyLoss.
        Clases más raras reciben mayor peso.
        Returns: Tensor (num_classes,) float32
        """
        labels = self.df["glosa"].map(self.glosa2idx).values
        counts = np.bincount(labels, minlength=self.num_classes).astype(np.float32)
        counts = np.clip(counts, 1, None)
        weights = 1.0 / counts
        weights = weights / weights.sum() * self.num_classes   # escala a media=1
        return torch.from_numpy(weights)

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row      = self.df.iloc[idx]
        kpts_raw = self._load_kpts(row)                  # (T, 133, 3)

        # Aumentación anatómica sobre los datos crudos (antes de normalizar)
        if self.augmenter is not None:
            kpts_proc = self.augmenter(kpts_raw)
        else:
            kpts_proc = kpts_raw

        # Preprocesar: normalizar y quitar score → (T, 133, 2)
        xy    = preprocess(kpts_proc, self.score_thresh)  # (T, 133, 2)
        T     = min(xy.shape[0], self.tmax)
        xy_flat = xy[:T].reshape(T, INPUT_DIM)            # (T, 266)

        label    = int(self.glosa2idx[row["glosa"]])
        video_id = str(row.get("video_id", idx))

        item = {
            "keypoints": torch.from_numpy(xy_flat),      # (T, 266) longitud variable
            "label":     torch.tensor(label, dtype=torch.long),
            "video_id":  video_id,
            "T":         T,
        }

        # Vista AimCLR: segunda vista extrema de los mismos datos crudos
        if self.return_aug and self.view_gen is not None:
            kpts_aug   = self.view_gen(kpts_raw)          # (T, 133, 3)
            xy_aug     = preprocess(kpts_aug, self.score_thresh)
            xy_aug_flat = xy_aug[:T].reshape(T, INPUT_DIM)
            item["keypoints_aug"] = torch.from_numpy(xy_aug_flat)  # (T, 266)

        return item


# ─────────────────────────────────────────────────────────────────────────────
# collate_fn — padding dinámico al máximo del batch
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Agrupa una lista de items en un batch con padding dinámico.
    Rellena hasta la secuencia más larga del batch (no hasta TMAX global),
    lo que ahorra memoria y cómputo en batches de secuencias cortas.

    Returns dict con:
        keypoints   : (B, T_max_batch, 266)  float32
        valid_mask  : (B, T_max_batch)        bool  — True = frame real
        label       : (B,)                    int64
        video_id    : List[str]
        keypoints_aug (si existe): (B, T_max_batch, 266)
    """
    T_max = max(item["T"] for item in batch)
    B     = len(batch)

    kpts_padded = torch.zeros(B, T_max, INPUT_DIM, dtype=torch.float32)
    valid_mask  = torch.zeros(B, T_max, dtype=torch.bool)
    labels      = torch.stack([item["label"] for item in batch])
    video_ids   = [item["video_id"] for item in batch]

    has_aug    = "keypoints_aug" in batch[0]
    aug_padded = torch.zeros(B, T_max, INPUT_DIM, dtype=torch.float32) if has_aug else None

    for i, item in enumerate(batch):
        T = item["T"]
        kpts_padded[i, :T] = item["keypoints"]
        valid_mask[i,  :T] = True
        if has_aug:
            aug_padded[i, :T] = item["keypoints_aug"]

    out = {
        "keypoints":  kpts_padded,   # (B, T_max, 266)
        "valid_mask": valid_mask,     # (B, T_max)
        "label":      labels,         # (B,)
        "video_id":   video_ids,
    }
    if has_aug:
        out["keypoints_aug"] = aug_padded

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: extraer ID de señante desde el campo intento
# ─────────────────────────────────────────────────────────────────────────────

def _extract_speaker(intento: str) -> str:
    """
    Extrae el ID de señante desde el campo intento.
    Formato: 'SSXXX' donde SS = señante (2 dígitos), XXX = glosa (3 dígitos).
    Ejemplo: '01001' → señante '01'
             '10249' → señante '10'
    """
    return str(intento).zfill(5)[:2]


# ─────────────────────────────────────────────────────────────────────────────
# split_dataset — punto de entrada principal desde train.py
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    parquet_path:   str,
    train_ratio:    float              = 0.70,
    val_ratio:      float              = 0.15,
    seed:           int                = 42,
    augment_train:  Optional[Callable] = None,
    score_thresh:   float              = 0.3,
    tmax:           int                = TMAX,
    view_gen:       Optional[Callable] = None,
    split_by_speaker: bool             = False,
    val_speakers:   Optional[List[str]] = None,
    test_speakers:  Optional[List[str]] = None,
) -> Tuple[LSMDataset, LSMDataset, LSMDataset]:
    """
    Carga el parquet y divide en train / val / test.

    Hay dos modos:

    1. split_by_speaker=True (recomendado para generalización):
       Divide por señante completo — ningún señante aparece en más de un split.
       Esto evalúa correctamente la capacidad del modelo de reconocer glosas
       de personas nuevas nunca vistas en entrenamiento.

       Por defecto asigna:
         - val_speakers  : el penúltimo señante  (ej. '09')
         - test_speakers : el último señante      (ej. '10')
         - train         : todos los demás        (ej. '01'–'08')

       Se puede especificar manualmente:
         val_speakers=['09'], test_speakers=['10']

    2. split_by_speaker=False (split aleatorio por video):
       Split estratificado por glosa — mezcla señantes entre splits.
       Útil para comparar con baselines de la literatura, pero infla
       las métricas si el objetivo es generalizar a personas nuevas.

    Parámetros
    ----------
    parquet_path     : ruta al .parquet
    train_ratio      : fracción train si split_by_speaker=False
    val_ratio        : fracción val   si split_by_speaker=False
    seed             : semilla para split aleatorio
    augment_train    : LandmarkAugmenter ó None
    score_thresh     : umbral confianza landmarks
    tmax             : longitud máxima de secuencia
    view_gen         : AimCLRViewGenerator ó None
    split_by_speaker : True = split por señante (recomendado)
    val_speakers     : lista de IDs de señante para val (ej. ['09'])
    test_speakers    : lista de IDs de señante para test (ej. ['10'])

    Returns
    -------
    (train_ds, val_ds, test_ds)
    """
    print(f"[split_dataset] Cargando {parquet_path} …")
    df = pd.read_parquet(parquet_path).reset_index(drop=True)

    # ── Vocabulario ───────────────────────────────────────────────────────────
    glosas_sorted = sorted(df["glosa"].unique().tolist())
    glosa2idx     = {g: i for i, g in enumerate(glosas_sorted)}
    num_classes   = len(glosa2idx)
    print(f"[split_dataset] {len(df)} videos | {num_classes} glosas únicas")

    if split_by_speaker:
        # ── Split por señante ─────────────────────────────────────────────────
        df["_speaker"]  = df["intento"].astype(str).str.zfill(5).str[:2]
        all_speakers    = sorted(df["_speaker"].unique().tolist())
        n_glosas_total  = df["glosa"].nunique()
        print(f"[split_dataset] Señantes detectados: {all_speakers}")

        # Clasificar señantes: completos (≥90% de glosas) e incompletos
        glosas_por_speaker = df.groupby("_speaker")["glosa"].nunique()
        umbral      = int(n_glosas_total * 0.90)
        completos   = sorted(glosas_por_speaker[glosas_por_speaker >= umbral].index.tolist())
        incompletos = sorted(glosas_por_speaker[glosas_por_speaker <  umbral].index.tolist())
        if incompletos:
            print(f"[split_dataset] Completos  (≥{umbral} glosas): {completos}")
            print(f"[split_dataset] Incompletos (<{umbral} glosas): {incompletos}")

        # Val y test solo desde señantes completos para evaluación justa
        if val_speakers is None:
            val_speakers  = [completos[-2]] if len(completos) >= 2 else [completos[-1]]
        if test_speakers is None:
            test_speakers = [completos[-1]]

        # Train = todos los demás, incluyendo incompletos
        # (aportan variabilidad sin contaminar evaluación)
        train_speakers = [s for s in all_speakers
                          if s not in val_speakers and s not in test_speakers]

        print(f"[split_dataset] Train señantes: {train_speakers}")
        print(f"[split_dataset] Val  señantes : {val_speakers}")
        print(f"[split_dataset] Test señantes : {test_speakers}")

        df_train = df[df["_speaker"].isin(train_speakers)].copy()
        df_val   = df[df["_speaker"].isin(val_speakers)].copy()
        df_test  = df[df["_speaker"].isin(test_speakers)].copy()

        # Verificar cobertura
        train_glosas = set(df_train["glosa"].unique())
        missing_val  = train_glosas - set(df_val["glosa"].unique())
        missing_test = train_glosas - set(df_test["glosa"].unique())
        if missing_val:
            print(f"  ⚠ {len(missing_val)} glosas sin representación en val")
        else:
            print(f"  ✓ Val cubre todas las glosas de train")
        if missing_test:
            print(f"  ⚠ {len(missing_test)} glosas sin representación en test")
        else:
            print(f"  ✓ Test cubre todas las glosas de train")

    else:
        # ── Split aleatorio estratificado por glosa (modo legacy) ─────────────
        print("[split_dataset] Modo: split aleatorio por glosa (split_by_speaker=False)")
        rng = np.random.default_rng(seed)
        train_idx, val_idx, test_idx = [], [], []

        for glosa, group in df.groupby("glosa"):
            idx = group.index.tolist()
            rng.shuffle(idx)
            n = len(idx)

            if n == 1:
                train_idx.extend(idx)
            elif n == 2:
                train_idx.append(idx[0])
                val_idx.append(idx[1])
            else:
                n_tr = max(1, int(round(n * train_ratio)))
                n_va = max(1, int(round(n * val_ratio)))
                n_te = max(0, n - n_tr - n_va)
                while n_tr + n_va + n_te > n:
                    if n_te > 0:   n_te -= 1
                    elif n_va > 1: n_va -= 1
                    else:          n_tr -= 1
                train_idx.extend(idx[:n_tr])
                val_idx.extend(idx[n_tr: n_tr + n_va])
                test_idx.extend(idx[n_tr + n_va: n_tr + n_va + n_te])

        df_train = df.loc[train_idx]
        df_val   = df.loc[val_idx]
        df_test  = df.loc[test_idx]

    print(
        f"[split_dataset] "
        f"train={len(df_train)} | val={len(df_val)} | test={len(df_test)}"
    )

    # ── Instanciar datasets ───────────────────────────────────────────────────
    train_ds = LSMDataset(
        df=df_train,
        glosa2idx=glosa2idx,
        augmenter=augment_train,
        score_thresh=score_thresh,
        tmax=tmax,
        return_aug=(view_gen is not None),
        view_gen=view_gen,
    )
    val_ds = LSMDataset(
        df=df_val,
        glosa2idx=glosa2idx,
        augmenter=None,
        score_thresh=score_thresh,
        tmax=tmax,
        return_aug=False,
    )
    test_ds = LSMDataset(
        df=df_test,
        glosa2idx=glosa2idx,
        augmenter=None,
        score_thresh=score_thresh,
        tmax=tmax,
        return_aug=False,
    )

    return train_ds, val_ds, test_ds