"""
src/dataset.py
==============
LSMDataset: carga landmarks desde Parquet, normaliza, aplica padding/masking
y expone un collate_fn listo para DataLoader.

Estructura Parquet esperada:
  glosa (str), intento (str), video_id (str), fps (float32),
  total_frames (int32), width (int32), height (int32),
  keypoints (bytes: float32 (T, 133, 3)),
  person_detected (bytes: bool (T,))
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

# ── Constantes COCO-WholeBody ────────────────────────────────────────────────
TMAX = 200                  # frames máximos (cubre >99 % del dataset)
N_LANDMARKS = 133
N_CHANNELS = 3              # x, y, score
INPUT_DIM = N_LANDMARKS * 2    # 266 — solo x,y (score no aporta señal discriminativa)

# Índices clave (COCO-WholeBody)
SHOULDER_L, SHOULDER_R = 5, 6
HIP_L, HIP_R = 11, 12

# Manos: muñecas y dedos
WRIST_L, WRIST_R = 91, 112
HAND_L_SLICE = slice(92, 113)   # 21 puntos mano izq
HAND_R_SLICE = slice(113, 134)  # 21 puntos mano der


# ── Preprocesamiento ──────────────────────────────────────────────────────────

def preprocess(
    keypoints: np.ndarray,          # (T, 133, 3)
    person_detected: np.ndarray,    # (T,) bool
    score_thresh: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normaliza la secuencia de landmarks:
      1. Convierte scores RTMPose (logits) → [0,1] con sigmoid.
      2. Centra respecto al punto medio de los hombros.
      3. Escala por la distancia hombros → caderas.
      4. Enmascara landmarks bajo umbral de confianza.

    Returns
    -------
    output   : (T, 133, 3)  float32 normalizado
    low_conf : (T, 133)     bool — True donde confianza < score_thresh
    """
    kpts = keypoints.copy().astype(np.float32)   # (T, 133, 3)

    # 1. Scores ya están en [0, 1] (RTMPose los normaliza en el pipeline de extracción)
    #    No aplicar sigmoid — solo verificar rango

    # 2. Máscara de confianza (ahora sí en [0,1])
    low_conf = kpts[:, :, 2] < score_thresh       # (T, 133)

    # 3. Punto medio de hombros → centrado
    shoulder_mid = (kpts[:, SHOULDER_L, :2] + kpts[:, SHOULDER_R, :2]) / 2  # (T, 2)
    kpts[:, :, :2] -= shoulder_mid[:, None, :]

    # 4. Escala: distancia hombro_mid → cadera_mid
    hip_mid = (kpts[:, HIP_L, :2] + kpts[:, HIP_R, :2]) / 2
    scale = np.linalg.norm(hip_mid - shoulder_mid, axis=-1)                  # (T,)
    scale = np.clip(scale, 1e-6, None)
    kpts[:, :, :2] /= scale[:, None, None]

    # 5. Anular landmarks de baja confianza
    kpts[low_conf] = 0.0

    # 6. Enmascarar frames sin persona detectada
    no_person = ~person_detected
    kpts[no_person] = 0.0

    return kpts, low_conf


def pad_or_truncate(
    kpts: np.ndarray,                  # (T, 133, 3)
    tmax: int = TMAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trunca o rellena con ceros hasta `tmax`.

    Returns
    -------
    padded      : (tmax, 133, 3)  float32
    valid_mask  : (tmax,)         bool — True en frames reales
    """
    T = kpts.shape[0]
    if T >= tmax:
        padded = kpts[:tmax].astype(np.float32)
        valid_mask = np.ones(tmax, dtype=bool)
    else:
        padded = np.zeros((tmax, N_LANDMARKS, N_CHANNELS), dtype=np.float32)
        padded[:T] = kpts.astype(np.float32)
        valid_mask = np.zeros(tmax, dtype=bool)
        valid_mask[:T] = True
    return padded, valid_mask


# ── Dataset ───────────────────────────────────────────────────────────────────

class LSMDataset(Dataset):
    """
    Dataset de Lengua de Señas Mexicana sobre landmarks.

    Parameters
    ----------
    parquet_path  : Ruta al archivo .parquet consolidado.
    augment       : Instancia de LandmarkAugmenter (o None para val/test).
    score_thresh  : Umbral de confianza para enmascarar landmarks.
    tmax          : Longitud máxima de secuencia (padding target).
    glosa_filter  : Lista de glosas a incluir; None = todas.
    """

    def __init__(
        self,
        parquet_path: str,
        augment=None,
        score_thresh: float = 0.3,
        tmax: int = TMAX,
        glosa_filter: Optional[List[str]] = None,
    ):
        super().__init__()
        self.tmax = tmax
        self.score_thresh = score_thresh
        self.augment = augment

        df = pd.read_parquet(parquet_path)

        if glosa_filter is not None:
            df = df[df["glosa"].isin(glosa_filter)].reset_index(drop=True)

        # Construir mapeo glosa ↔ índice
        glosas_sorted = sorted(df["glosa"].unique())
        self.glosa2idx: Dict[str, int] = {g: i for i, g in enumerate(glosas_sorted)}
        self.idx2glosa: Dict[int, str] = {i: g for g, i in self.glosa2idx.items()}
        self.num_classes = len(self.glosa2idx)

        self.df = df.reset_index(drop=True)

    # ── helpers internos ──────────────────────────────────────────────────────

    @staticmethod
    def _deserialize(blob: bytes, dtype=np.float32) -> np.ndarray:
        """Reconstruye ndarray desde bytes serializado con np.save."""
        return np.load(io.BytesIO(blob), allow_pickle=False).astype(dtype)

    def _load_row(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        row = self.df.iloc[idx]
        kpts = self._deserialize(row["keypoints"], np.float32)          # (T, 133, 3)
        person = self._deserialize(row["person_detected"], np.bool_)    # (T,)
        return kpts, person

    # ── interfaz pública ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        kpts, person = self._load_row(idx)
        glosa = self.df.iloc[idx]["glosa"]
        label = self.glosa2idx[glosa]

        # Normalizar
        kpts, _ = preprocess(kpts, person, score_thresh=self.score_thresh)

        # Augmentación (solo si se proporcionó)
        if self.augment is not None:
            kpts = self.augment(kpts)

        # Padding / truncamiento
        padded, valid_mask = pad_or_truncate(kpts, self.tmax)

        # Aplanar solo xy → (T, 266)
        flat = padded[:, :, :2].reshape(self.tmax, INPUT_DIM)

        return {
            "keypoints":   torch.from_numpy(flat),          # (T, 399)
            "valid_mask":  torch.from_numpy(valid_mask),    # (T,) bool
            "label":       torch.tensor(label, dtype=torch.long),
            "video_id":    self.df.iloc[idx].get("video_id", ""),
        }

    def get_label_weights(self) -> Tensor:
        """
        Devuelve pesos inversamente proporcionales a la frecuencia de clase,
        útil para loss ponderada con clases desbalanceadas.
        """
        counts = self.df["glosa"].value_counts()
        weights = np.array([1.0 / counts.get(self.idx2glosa[i], 1) for i in range(self.num_classes)], dtype=np.float32)
        weights /= weights.sum()
        return torch.from_numpy(weights)


# ── collate_fn ────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict[str, Tensor]:
    """
    Agrupa una lista de muestras en tensores de batch.
    Asume que todos los tensores ya tienen la misma longitud (TMAX).
    """
    return {
        "keypoints":  torch.stack([b["keypoints"] for b in batch]),    # (B, T, 399)
        "valid_mask": torch.stack([b["valid_mask"] for b in batch]),   # (B, T)
        "label":      torch.stack([b["label"] for b in batch]),        # (B,)
        "video_id":   [b["video_id"] for b in batch],
    }


# ── Split helper ──────────────────────────────────────────────────────────────

def split_dataset(
    parquet_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    augment_train=None,
    score_thresh: float = 0.3,
    tmax: int = TMAX,
) -> Tuple["LSMDataset", "LSMDataset", "LSMDataset"]:
    """
    Divide el dataset en train/val/test de forma estratificada por glosa.
    Retorna tres instancias de LSMDataset.
    """
    import sklearn.model_selection as ms

    df = pd.read_parquet(parquet_path)
    glosas = df["glosa"].values

    # Primera división: train vs (val + test)
    idx_all = np.arange(len(df))
    idx_train, idx_valtest = ms.train_test_split(
        idx_all, test_size=1.0 - train_ratio, random_state=seed, stratify=glosas
    )

    # Segunda división: val vs test (dentro del 30 % restante)
    val_frac = val_ratio / (1.0 - train_ratio)
    idx_val, idx_test = ms.train_test_split(
        idx_valtest, test_size=1.0 - val_frac,
        random_state=seed, stratify=glosas[idx_valtest]
    )

    def _make_subset(indices, augment):
        sub = df.iloc[indices].reset_index(drop=True)
        # Guardamos subset en memoria (evita re-lectura de parquet)
        ds = _SubsetDataset(sub, augment=augment, score_thresh=score_thresh, tmax=tmax)
        return ds

    train_ds = _make_subset(idx_train, augment_train)
    val_ds   = _make_subset(idx_val,   None)
    test_ds  = _make_subset(idx_test,  None)

    # Compartir mapeo de clases
    all_glosas = sorted(df["glosa"].unique())
    g2i = {g: i for i, g in enumerate(all_glosas)}
    i2g = {i: g for g, i in g2i.items()}
    for ds in (train_ds, val_ds, test_ds):
        ds.glosa2idx = g2i
        ds.idx2glosa = i2g
        ds.num_classes = len(g2i)

    return train_ds, val_ds, test_ds


class _SubsetDataset(LSMDataset):
    """Variante interna que opera sobre un DataFrame ya filtrado en memoria."""

    def __init__(self, df: pd.DataFrame, **kwargs):
        # Saltamos el __init__ padre para no re-leer parquet
        Dataset.__init__(self)
        self.tmax = kwargs.get("tmax", TMAX)
        self.score_thresh = kwargs.get("score_thresh", 0.3)
        self.augment = kwargs.get("augment", None)
        self.df = df.reset_index(drop=True)

        glosas_sorted = sorted(df["glosa"].unique())
        self.glosa2idx = {g: i for i, g in enumerate(glosas_sorted)}
        self.idx2glosa = {i: g for g, i in self.glosa2idx.items()}
        self.num_classes = len(self.glosa2idx)