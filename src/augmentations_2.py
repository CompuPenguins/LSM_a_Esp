"""
src/augmentations.py
====================
Pipeline de aumentación para secuencias de landmarks LSM.

Dos niveles:
  1. LandmarkAugmenter — aumentaciones anatómicas suaves para entrenamiento supervisado.
  2. AimCLRViewGenerator — genera dos vistas contrastivas (extremas) para la pérdida D3M.

Formato de entrada/salida: np.ndarray de shape (T, 133, 3) — [x, y, score] normalizados.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from typing import Optional, Tuple

# ── Índices COCO-WholeBody ────────────────────────────────────────────────────
SHOULDER_L, SHOULDER_R = 5, 6
HIP_L, HIP_R = 11, 12

LEFT_ARM_SLICE  = slice(7, 11)    # left elbow → left wrist (body)
RIGHT_ARM_SLICE = slice(12, 16)   # right elbow → right wrist (body)

WRIST_L, WRIST_R = 91, 112
HAND_L_SLICE = slice(92, 113)     # 21 puntos mano izquierda
HAND_R_SLICE = slice(113, 134)    # 21 puntos mano derecha


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _rng(seed=None) -> np.random.Generator:
    return np.random.default_rng(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Aumentaciones anatómicas individuales
# ─────────────────────────────────────────────────────────────────────────────

def aug_build_variation(
    kpts: np.ndarray,
    scale: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simula variación de complexión ensanchando/acortando la distancia entre hombros.
    Desplaza lateralmente brazos y manos para mantener coherencia esqueletal.

    scale ∈ [0.90, 1.10]  →  fornido ↔ delgado
    """
    if rng is None:
        rng = _rng()
    if scale is None:
        scale = rng.uniform(0.90, 1.10)

    kpts = kpts.copy()
    shoulder_vec = (kpts[:, SHOULDER_R, :2] - kpts[:, SHOULDER_L, :2])  # (T, 2)
    dx = shoulder_vec * (scale - 1.0) / 2.0                              # (T, 2)

    kpts[:, LEFT_ARM_SLICE,  :2] -= dx[:, None, :]
    kpts[:, RIGHT_ARM_SLICE, :2] += dx[:, None, :]
    kpts[:, WRIST_L, :2]         -= dx
    kpts[:, WRIST_R, :2]         += dx
    kpts[:, HAND_L_SLICE, :2]    -= dx[:, None, :]
    kpts[:, HAND_R_SLICE, :2]    += dx[:, None, :]

    return kpts


def aug_hand_scale(
    kpts: np.ndarray,
    scale: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Escala las manos respecto a sus muñecas.
    scale ∈ [0.95, 1.05]  →  manos pequeñas ↔ grandes
    """
    if rng is None:
        rng = _rng()
    if scale is None:
        scale = rng.uniform(0.95, 1.05)

    kpts = kpts.copy()
    for wrist_idx, hand_slice in [(WRIST_L, HAND_L_SLICE), (WRIST_R, HAND_R_SLICE)]:
        pivot = kpts[:, wrist_idx, :2]                   # (T, 2)
        offset = kpts[:, hand_slice, :2] - pivot[:, None, :]
        kpts[:, hand_slice, :2] = pivot[:, None, :] + offset * scale

    return kpts


def aug_gaussian_noise(
    kpts: np.ndarray,
    sigma: float | None = None,
    score_thresh: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Ruido gaussiano score-aware: solo afecta landmarks de alta confianza.
    sigma ∈ [0, 2.0] píxeles (en espacio normalizado; ajustar si se desea)
    """
    if rng is None:
        rng = _rng()
    if sigma is None:
        sigma = rng.uniform(0.0, 0.02)   # ~2 píx en espacio norm ≈ 0.02 unid.

    kpts = kpts.copy()
    noise = rng.normal(0, sigma, size=(*kpts.shape[:2], 2)).astype(np.float32)
    mask = (kpts[:, :, 2] > score_thresh)[:, :, None]
    kpts[:, :, :2] += noise * mask
    return kpts


def aug_temporal_jitter(
    kpts: np.ndarray,
    max_shift: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Desplazamiento temporal aleatorio (roll) dentro de la ventana de frames válidos.
    """
    if rng is None:
        rng = _rng()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(kpts, shift, axis=0)


def aug_speed_perturbation(
    kpts: np.ndarray,
    rate: float | None = None,
    target_len: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Perturbación de velocidad: resamplea la secuencia temporal.
    rate ∈ [0.8, 1.2]  →  más lento ↔ más rápido
    Devuelve la misma longitud que la entrada (interpolado y truncado/padded).
    """
    if rng is None:
        rng = _rng()
    if rate is None:
        rate = rng.uniform(0.8, 1.2)

    T = kpts.shape[0]
    new_len = max(1, int(T * rate))

    # Eje original y eje re-muestreado, ambos en [0, T-1]
    src_idx = np.arange(T, dtype=np.float32)               # posiciones reales (T,)
    dst_idx = np.linspace(0, T - 1, new_len, dtype=np.float32)  # posiciones "estiradas" (new_len,)
    out_idx = np.linspace(0, T - 1, T, dtype=np.float32)  # re-muestrear a T frames

    # Interpolar a new_len → luego re-muestrear a T para mantener longitud fija
    flat = kpts.reshape(T, -1)                             # (T, 399)
    out  = np.zeros_like(flat)
    for c in range(flat.shape[1]):
        stretched = np.interp(dst_idx, src_idx, flat[:, c])   # (new_len,)
        out[:, c] = np.interp(out_idx, dst_idx, stretched)    # (T,)
    return out.reshape(kpts.shape)


# ─────────────────────────────────────────────────────────────────────────────
# LandmarkAugmenter  (entrenamiento supervisado)
# ─────────────────────────────────────────────────────────────────────────────

class LandmarkAugmenter:
    """
    Compone aumentaciones anatómicas con probabilidades configurables.

    Uso:
        augmenter = LandmarkAugmenter(p_build=0.5, p_hand=0.5, p_noise=0.7, p_speed=0.4)
        kpts_aug = augmenter(kpts)   # (T, 133, 3)
    """

    def __init__(
        self,
        p_build:  float = 0.5,
        p_hand:   float = 0.5,
        p_noise:  float = 0.7,
        p_jitter: float = 0.3,
        p_speed:  float = 0.4,
        seed:     int | None = None,
    ):
        self.p_build  = p_build
        self.p_hand   = p_hand
        self.p_noise  = p_noise
        self.p_jitter = p_jitter
        self.p_speed  = p_speed
        self._rng = _rng(seed)

    def __call__(self, kpts: np.ndarray) -> np.ndarray:
        rng = self._rng
        if rng.random() < self.p_build:
            kpts = aug_build_variation(kpts, rng=rng)
        if rng.random() < self.p_hand:
            kpts = aug_hand_scale(kpts, rng=rng)
        if rng.random() < self.p_noise:
            kpts = aug_gaussian_noise(kpts, rng=rng)
        if rng.random() < self.p_jitter:
            kpts = aug_temporal_jitter(kpts, rng=rng)
        if rng.random() < self.p_speed:
            kpts = aug_speed_perturbation(kpts, rng=rng)
        return kpts


# ─────────────────────────────────────────────────────────────────────────────
# Aumentaciones extremas para AimCLR
# ─────────────────────────────────────────────────────────────────────────────

def aug_temporal_flip(kpts: np.ndarray) -> np.ndarray:
    """Invierte la secuencia en tiempo."""
    return kpts[::-1].copy()


def aug_axis_masking(
    kpts: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Enmascara aleatoriamente el eje X o Y completo."""
    if rng is None:
        rng = _rng()
    axis = int(rng.integers(0, 2))   # 0=X, 1=Y
    kpts = kpts.copy()
    kpts[:, :, axis] = 0.0
    return kpts


def aug_gaussian_blur_temporal(
    kpts: np.ndarray,
    sigma: float = 1.5,
) -> np.ndarray:
    """Suaviza la secuencia temporal con un filtro gaussiano 1-D."""
    kpts = kpts.copy()
    # Aplicar a cada landmark y cada canal por separado
    T, L, C = kpts.shape
    flat = kpts.reshape(T, -1)
    blurred = gaussian_filter1d(flat, sigma=sigma, axis=0)
    return blurred.reshape(T, L, C).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# AimCLRViewGenerator
# ─────────────────────────────────────────────────────────────────────────────

class AimCLRViewGenerator:
    """
    Genera dos vistas contrastivas (extremas) de una misma secuencia.
    Cada vista aplica una combinación aleatoria de:
      - Temporal flip       (p_flip)
      - Axis masking        (p_axis)
      - Gaussian blur 1-D   (p_blur)
      - Ruido gaussiano      (siempre, σ mayor que en supervisado)

    Returns
    -------
    view1, view2 : np.ndarray de shape (T, 133, 3)
    """

    def __init__(
        self,
        p_flip:  float = 0.5,
        p_axis:  float = 0.5,
        p_blur:  float = 0.5,
        blur_sigma: float = 1.5,
        noise_sigma: float = 0.04,
        seed: int | None = None,
    ):
        self.p_flip  = p_flip
        self.p_axis  = p_axis
        self.p_blur  = p_blur
        self.blur_sigma  = blur_sigma
        self.noise_sigma = noise_sigma
        self._rng = _rng(seed)

    def _transform(self, kpts: np.ndarray) -> np.ndarray:
        rng = self._rng
        k = kpts.copy()
        if rng.random() < self.p_flip:
            k = aug_temporal_flip(k)
        if rng.random() < self.p_axis:
            k = aug_axis_masking(k, rng=rng)
        if rng.random() < self.p_blur:
            k = aug_gaussian_blur_temporal(k, sigma=self.blur_sigma)
        # Ruido moderado siempre
        k = aug_gaussian_noise(k, sigma=self.noise_sigma, rng=rng)
        return k

    def __call__(self, kpts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        view1 = self._transform(kpts)
        view2 = self._transform(kpts)
        return view1, view2