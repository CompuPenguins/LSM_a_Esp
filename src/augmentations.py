"""
src/augmentations.py
====================
Aumentaciones para secuencias de landmarks LSM.

Solo se mantienen las que producen un cambio perceptible y verificado:

  LandmarkAugmenter (entrenamiento estándar) — 6 aumentaciones:
    1. _speed_perturbation   ★ más valiosa: misma forma, distinta velocidad
    2. _spatial_scale          escala amplitud del movimiento (distancia a cámara)
    3. _wrist_trajectory_noise ruido suavizado sobre trayectoria de muñecas
    4. _temporal_blur          suaviza movimientos rápidos (σ=1.5 validado)
    5. _temporal_crop_pad      simula inicio/fin tardío de grabación
    6. _rotation_2d            inclinación de cámara (±10°)

  AimCLRViewGenerator (pérdida contrastiva D3M) — vistas extremas:
    Base: LandmarkAugmenter con rangos agresivos
    Extra: _temporal_flip + _keypoint_group_dropout + _temporal_blur

Eliminadas por ser inefectivas o redundantes:
    - _vary_build       : propagar a 50+ kpts es frágil; efecto marginal
    - _vary_hand_size   : solapado con _spatial_scale post-normalización
    - _add_gaussian_noise uniforme : solapado con _wrist_trajectory_noise
    - _finger_pose_noise: sigma demasiado bajo, solapado con wrist noise
    - _keypoint_dropout frame-a-frame: lento y menos realista que group dropout
    - _temporal_segment_shuffle: en AimCLR basta con _temporal_flip
    - _mirror_horizontal: riesgo de mezclar glosas simétricas distintas
"""

from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# Índices COCO-WholeBody (133 kpts)
# ─────────────────────────────────────────────────────────────────────────────

_WRIST_L  = 17                          # era 91
_WRIST_R  = 38                          # era 112
_HAND_L   = list(range(17, 38))         # era range(92, 113)  — 21 kpts
_HAND_R   = list(range(38, 59))    
# _WRIST_L  = 91
# _WRIST_R  = 112
# _HAND_L   = list(range(92, 113))   # 21 kpts mano izquierda
# _HAND_R   = list(range(113, 133))  # 21 kpts mano derecha


# ─────────────────────────────────────────────────────────────────────────────
# PRIMITIVAS
# ─────────────────────────────────────────────────────────────────────────────

def _speed_perturbation(kpts: np.ndarray, rate: float) -> np.ndarray:
    """
    ★ Aumentación más valiosa según diagnóstico visual.
    ...
    """
    T, N, C = kpts.shape          # N es dinámico (133 antes, 59 ahora)
    new_len  = max(2, int(round(T * rate)))
    dst_t    = np.arange(T)

    if new_len >= T:
        src_t    = np.linspace(0, T - 1, new_len)
        expanded = np.empty((new_len, N, C), dtype=np.float32)
        for k in range(N):
            for c in range(C):
                expanded[:, k, c] = np.interp(src_t, np.arange(T), kpts[:, k, c])
        out = expanded[:T].copy()
    else:
        src_t = np.linspace(0, T - 1, new_len)
        out   = np.empty((T, N, C), dtype=np.float32)
        for k in range(N):
            for c in range(C):
                compressed      = np.interp(src_t, np.arange(T), kpts[:, k, c])
                out[:, k, c]    = np.interp(dst_t, src_t, compressed)

    return out.astype(np.float32)


def _spatial_scale(kpts: np.ndarray, s: float) -> np.ndarray:
    """
    Escala la amplitud de todo el movimiento respecto al centro de los hombros.
    Simula señantes con diferente amplitud de movimiento o distancia a la cámara.

    Funciona pre y post normalización: post-normalización cambia la amplitud
    relativa de las manos respecto al torso, que sí es señal discriminativa.

    kpts : (T, 133, 3)
    s    : [0.80, 1.20]
    """
    out   = kpts.copy()
    pivot = (kpts[:, 5, :2] + kpts[:, 6, :2]) / 2.0   # (T, 2) centro hombros
    out[:, :, :2] = pivot[:, None, :] + (kpts[:, :, :2] - pivot[:, None, :]) * s
    return out


def _wrist_trajectory_noise(kpts: np.ndarray, sigma: float) -> np.ndarray:
    """
    Ruido suavizado temporalmente sobre la trayectoria de ambas muñecas.
    Simula variación natural inter-señante en la trayectoria del movimiento.

    El ruido se filtra temporalmente (σ_t=3 frames) para que parezca una
    variación continua de trayectoria, no ruido frame a frame.

    kpts  : (T, 133, 3)
    sigma : [0.02, 0.10]
    """
    out = kpts.copy()
    T   = kpts.shape[0]
    for wrist in [_WRIST_L, _WRIST_R]:
        raw    = np.random.normal(0.0, sigma, (T, 2)).astype(np.float32)
        smooth = gaussian_filter1d(raw, sigma=3.0, axis=0).astype(np.float32)
        out[:, wrist, :2] += smooth
    return out


def _temporal_blur(kpts: np.ndarray, sigma: float) -> np.ndarray:
    """
    Suavizado gaussiano 1D sobre la dimensión temporal.
    Simula movimientos más fluidos / menos temblorosos.
    σ=1.5 validado como sweet spot (Δ_rel≈1.4%, razonable).

    kpts  : (T, 133, 3)
    sigma : [0.5, 2.5]
    """
    out = kpts.copy()
    out[:, :, :2] = gaussian_filter1d(
        kpts[:, :, :2].astype(np.float64), sigma=sigma, axis=0
    ).astype(np.float32)
    return out


def _temporal_crop_pad(kpts: np.ndarray, crop_ratio: float) -> np.ndarray:
    """
    Recorta un porcentaje del inicio o fin y rellena repitiendo el frame extremo.
    Simula que la grabación empezó tarde o terminó antes de que la seña terminara,
    que es el caso real en tiempo real con buffer deslizante.

    kpts       : (T, 133, 3)
    crop_ratio : [0.05, 0.20]
    """
    T   = kpts.shape[0]
    n   = max(1, int(T * crop_ratio))
    out = kpts.copy()
    if np.random.random() < 0.5:
        out[:n] = kpts[n]           # rellenar inicio con primer frame válido
    else:
        out[T - n:] = kpts[T-n-1]  # rellenar fin con último frame válido
    return out


def _rotation_2d(kpts: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotación 2D de toda la figura respecto al centro de los hombros.
    Simula inclinación de la cámara o del señante.
    Semánticamente válida hasta ±15°.

    kpts      : (T, 133, 3)
    angle_deg : [-10, 10]
    """
    theta = np.deg2rad(angle_deg)
    c, s  = np.cos(theta), np.sin(theta)
    R     = np.array([[c, -s], [s, c]], dtype=np.float32)

    out     = kpts.copy()
    pivot   = (kpts[:, 5, :2] + kpts[:, 6, :2]) / 2.0          # (T, 2)
    centered = kpts[:, :, :2] - pivot[:, None, :]               # (T, 133, 2)
    out[:, :, :2] = np.einsum('tki,ij->tkj', centered, R.T) + pivot[:, None, :]
    return out


def _temporal_flip(kpts: np.ndarray) -> np.ndarray:
    """
    Invierte la secuencia en tiempo.
    Solo para AimCLR — rompe la semántica temporal de la seña.
    """
    return kpts[::-1].copy()


def _keypoint_group_dropout(kpts: np.ndarray, p_group: float = 0.5) -> np.ndarray:
    """
    Elimina una mano completa durante un segmento de tiempo aleatorio.
    Simula oclusión realista: una mano sale del cuadro durante N frames
    consecutivos (no dropout frame a frame, que es menos realista).

    Solo para AimCLR.

    kpts    : (T, 133, 3)
    p_group : prob. de aplicar a cada mano
    """
    out = kpts.copy()
    T   = kpts.shape[0]
    for hand_idx in [_HAND_L + [_WRIST_L], _HAND_R + [_WRIST_R]]:
        if np.random.random() < p_group:
            duration = np.random.randint(max(1, T // 6), max(2, T // 2))
            start    = np.random.randint(0, max(1, T - duration))
            out[start:start + duration, hand_idx, :] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LandmarkAugmenter — aumentación estándar para entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

class LandmarkAugmenter:
    """
    6 aumentaciones efectivas para entrenamiento estándar.
    Todas preservan la semántica de la glosa.

    Se aplica ANTES de preprocess() sobre (T, 133, 3) con score.

    En promedio se activan ~4 de las 6 por muestra (suficiente diversidad).
    Con batch_size=32 y ~1700 muestras de train, cada época ve
    efectivamente ~6800 variantes distintas (~4x el dataset original).
    """

    def __init__(
        self,
        p_speed:       float = 0.8,
        p_scale:       float = 0.6,
        p_wrist_noise: float = 0.7,
        p_blur:        float = 0.5,
        p_crop:        float = 0.5,
        p_rotation:    float = 0.4,
        # Rangos
        speed_range:        Tuple[float, float] = (0.6, 1.4),
        scale_range:        Tuple[float, float] = (0.80, 1.20),
        wrist_noise_range:  Tuple[float, float] = (0.02, 0.10),
        blur_sigma_range:   Tuple[float, float] = (0.5, 2.5),
        crop_ratio_range:   Tuple[float, float] = (0.05, 0.20),
        rotation_range_deg: Tuple[float, float] = (-10.0, 10.0),
    ):
        self.p_speed       = p_speed
        self.p_scale       = p_scale
        self.p_wrist_noise = p_wrist_noise
        self.p_blur        = p_blur
        self.p_crop        = p_crop
        self.p_rotation    = p_rotation

        self.speed_range        = speed_range
        self.scale_range        = scale_range
        self.wrist_noise_range  = wrist_noise_range
        self.blur_sigma_range   = blur_sigma_range
        self.crop_ratio_range   = crop_ratio_range
        self.rotation_range_deg = rotation_range_deg

    def __call__(self, kpts: np.ndarray) -> np.ndarray:
        """
        kpts : (T, 133, 3) float32 — crudos con score, PRE preprocess()
        Returns (T, 133, 3) float32 aumentado
        """
        kpts = kpts.copy()

        if np.random.random() < self.p_scale:
            kpts = _spatial_scale(kpts, np.random.uniform(*self.scale_range))

        if np.random.random() < self.p_rotation:
            kpts = _rotation_2d(kpts, np.random.uniform(*self.rotation_range_deg))

        if np.random.random() < self.p_speed:
            kpts = _speed_perturbation(kpts, np.random.uniform(*self.speed_range))

        if np.random.random() < self.p_blur:
            kpts = _temporal_blur(kpts, np.random.uniform(*self.blur_sigma_range))

        if np.random.random() < self.p_crop:
            kpts = _temporal_crop_pad(kpts, np.random.uniform(*self.crop_ratio_range))

        if np.random.random() < self.p_wrist_noise:
            kpts = _wrist_trajectory_noise(kpts, np.random.uniform(*self.wrist_noise_range))

        return kpts.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# AimCLRViewGenerator — dos vistas extremas para pérdida D3M
# ─────────────────────────────────────────────────────────────────────────────

class AimCLRViewGenerator:
    """
    Genera dos vistas independientes y agresivamente aumentadas para
    la pérdida contrastiva D3M (AimCLR).

    Cada vista pasa primero por LandmarkAugmenter con rangos extremos,
    luego añade: flip temporal + group dropout + blur fuerte.

    Las dos vistas se generan independientemente para que la red aprenda
    representaciones invariantes a estas perturbaciones.
    """

    def __init__(
        self,
        p_flip:          float = 0.5,
        p_group_dropout: float = 0.5,
        p_blur:          float = 0.6,
        blur_sigma:      float = 2.5,
    ):
        self.p_flip          = p_flip
        self.p_group_dropout = p_group_dropout
        self.p_blur          = p_blur
        self.blur_sigma      = blur_sigma

        # LandmarkAugmenter con rangos más agresivos como base
        self._base = LandmarkAugmenter(
            p_speed=0.9,       p_scale=0.7,
            p_wrist_noise=0.8, p_blur=0.6,
            p_crop=0.6,        p_rotation=0.5,
            speed_range=(0.5, 1.5),
            scale_range=(0.75, 1.25),
            wrist_noise_range=(0.04, 0.14),
            blur_sigma_range=(1.0, 3.5),
            crop_ratio_range=(0.10, 0.25),
            rotation_range_deg=(-15.0, 15.0),
        )

    def _make_view(self, kpts: np.ndarray) -> np.ndarray:
        kpts = self._base(kpts)

        if np.random.random() < self.p_flip:
            kpts = _temporal_flip(kpts)

        if np.random.random() < self.p_group_dropout:
            kpts = _keypoint_group_dropout(kpts, p_group=0.6)

        if np.random.random() < self.p_blur:
            kpts = _temporal_blur(kpts, self.blur_sigma)

        return kpts.astype(np.float32)

    def __call__(self, kpts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        kpts : (T, 133, 3)
        Returns: (view1, view2) generadas independientemente
        """
        return self._make_view(kpts), self._make_view(kpts)