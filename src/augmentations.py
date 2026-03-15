"""
src/augmentations.py
──────────────────────────────────────────────────────────────────────────────
Augmentaciones de landmarks para LSM. Diseñadas para aplicarse on-the-fly
dentro de LSMDataset sobre tensores numpy (T, 133, 2) — solo canales x, y.

Todas las augmentaciones:
  - Reciben y devuelven arrays numpy float32 de shape (T, 133, 2)
  - Son probabilísticas: cada una tiene un parámetro `p` de aplicación
  - Preservan la semántica de la seña (no distorsionan más allá de lo realista)

Grupos implementados (roadmap claude.md sección 7A):
  Espacial geométrica:
    - RandomTranslation   : traslación aleatoria del cuerpo en x, y
    - RandomScale         : escalado uniforme alrededor del centroide
    - RandomRotation      : rotación pequeña alrededor del centroide
    - GaussianNoise       : ruido gaussiano leve en coordenadas

  Temporal:
    - TemporalJitter      : desplazamiento aleatorio de frames individuales
    - FrameDrop           : elimina frames aleatorios e interpola
    - SpeedPerturbation   : resamplea la secuencia (más rápido / más lento)
    - TimeWarp            : deformación temporal suave con spline

  Oclusión sintética:
    - RegionDropout       : pone a cero una región corporal completa por N frames

  Score-aware:
    - ScoreBasedNoise     : más ruido en keypoints con score bajo (canal 2)
      (requiere el tensor completo (T, 133, 3) — ver docstring)

  Composición:
    - Compose             : aplica una lista de augmentaciones en secuencia
    - build_train_augments: preset recomendado para entrenamiento

Uso:
    from src.augmentations import build_train_augments

    aug = build_train_augments()
    xy_aug = aug(xy)   # xy: numpy (T, 133, 2)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


# ──────────────────────────────────────────────────────────────────────────────
#  Índices de regiones COCO-WholeBody 133
# ──────────────────────────────────────────────────────────────────────────────
REGIONS = {
    "body":    list(range(0,   17)),
    "foot_l":  list(range(17,  21)),
    "foot_r":  list(range(21,  25)),
    "face":    list(range(25,  92)),
    "hand_l":  list(range(92,  113)),
    "hand_r":  list(range(113, 133)),
}


# ──────────────────────────────────────────────────────────────────────────────
#  Base
# ──────────────────────────────────────────────────────────────────────────────
class LandmarkAugmentation:
    """Clase base. Todas las subclases implementan __call__(xy) → xy."""
    def __init__(self, p: float = 0.5):
        assert 0.0 <= p <= 1.0
        self.p = p

    def apply(self, xy: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return self.apply(xy.copy())
        return xy


# ──────────────────────────────────────────────────────────────────────────────
#  Espaciales
# ──────────────────────────────────────────────────────────────────────────────
class RandomTranslation(LandmarkAugmentation):
    """
    Traslada todos los keypoints por un offset aleatorio en x e y.
    El offset se muestrea en fracción del rango visible [0,1].

    max_shift: máximo desplazamiento como fracción del espacio normalizado.
    """
    def __init__(self, max_shift: float = 0.05, p: float = 0.5):
        super().__init__(p)
        self.max_shift = max_shift

    def apply(self, xy: np.ndarray) -> np.ndarray:
        dx = np.random.uniform(-self.max_shift, self.max_shift)
        dy = np.random.uniform(-self.max_shift, self.max_shift)
        xy[:, :, 0] += dx
        xy[:, :, 1] += dy
        return xy


class RandomScale(LandmarkAugmentation):
    """
    Escala los keypoints alrededor de su centroide temporal.
    scale_range: (min_scale, max_scale), ej. (0.85, 1.15)
    """
    def __init__(self, scale_range: tuple[float, float] = (0.85, 1.15), p: float = 0.5):
        super().__init__(p)
        self.scale_range = scale_range

    def apply(self, xy: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(*self.scale_range)
        # Centroide sobre todos los frames y keypoints
        cx = xy[:, :, 0].mean()
        cy = xy[:, :, 1].mean()
        xy[:, :, 0] = cx + (xy[:, :, 0] - cx) * scale
        xy[:, :, 1] = cy + (xy[:, :, 1] - cy) * scale
        return xy


class RandomRotation(LandmarkAugmentation):
    """
    Rotación 2D alrededor del centroide.
    max_angle_deg: máximo ángulo de rotación en grados (en ambas direcciones).
    Recomendado: ≤ 15° para no distorsionar la semántica de la seña.
    """
    def __init__(self, max_angle_deg: float = 10.0, p: float = 0.5):
        super().__init__(p)
        self.max_angle_rad = np.deg2rad(max_angle_deg)

    def apply(self, xy: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-self.max_angle_rad, self.max_angle_rad)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        cx = xy[:, :, 0].mean()
        cy = xy[:, :, 1].mean()

        x_c = xy[:, :, 0] - cx
        y_c = xy[:, :, 1] - cy

        xy[:, :, 0] = cx + cos_a * x_c - sin_a * y_c
        xy[:, :, 1] = cy + sin_a * x_c + cos_a * y_c
        return xy


class GaussianNoise(LandmarkAugmentation):
    """
    Añade ruido gaussiano independiente a cada coordenada.
    std: desviación estándar del ruido como fracción del espacio [0,1].
    """
    def __init__(self, std: float = 0.01, p: float = 0.5):
        super().__init__(p)
        self.std = std

    def apply(self, xy: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.std, size=xy.shape).astype(np.float32)
        return xy + noise


# ──────────────────────────────────────────────────────────────────────────────
#  Temporales
# ──────────────────────────────────────────────────────────────────────────────
class TemporalJitter(LandmarkAugmentation):
    """
    Desplaza ligeramente el orden de frames individuales.
    Simula pequeñas irregularidades en la captura de video.
    max_shift: máximo número de frames a desplazar.
    """
    def __init__(self, max_shift: int = 2, p: float = 0.5):
        super().__init__(p)
        self.max_shift = max_shift

    def apply(self, xy: np.ndarray) -> np.ndarray:
        T = xy.shape[0]
        if T <= 2 * self.max_shift + 1:
            return xy
        # Generar índices con jitter
        indices = np.arange(T)
        jitter  = np.random.randint(-self.max_shift, self.max_shift + 1, size=T)
        indices = np.clip(indices + jitter, 0, T - 1)
        return xy[indices]


class FrameDrop(LandmarkAugmentation):
    """
    Elimina aleatoriamente una fracción de frames e interpola los huecos.
    drop_ratio: fracción de frames a eliminar (ej. 0.1 = 10%).
    Preserva al menos min_frames frames.
    """
    def __init__(self, drop_ratio: float = 0.1, min_frames: int = 8, p: float = 0.5):
        super().__init__(p)
        self.drop_ratio = drop_ratio
        self.min_frames = min_frames

    def apply(self, xy: np.ndarray) -> np.ndarray:
        T = xy.shape[0]
        n_drop = int(T * self.drop_ratio)
        if T - n_drop < self.min_frames:
            return xy

        # Seleccionar frames a conservar
        keep_mask = np.ones(T, dtype=bool)
        drop_idx  = np.random.choice(T, size=n_drop, replace=False)
        keep_mask[drop_idx] = False

        kept_frames = xy[keep_mask]          # (T - n_drop, 133, 2)
        kept_times  = np.where(keep_mask)[0] # índices originales conservados

        # Interpolar de vuelta a T frames
        original_times = np.arange(T)
        K = xy.shape[1]

        result = np.zeros_like(xy)
        for k in range(K):
            for c in range(2):
                f = interp1d(kept_times, kept_frames[:, k, c],
                             kind="linear", fill_value="extrapolate")
                result[:, k, c] = f(original_times)

        return result.astype(np.float32)


class SpeedPerturbation(LandmarkAugmentation):
    """
    Resamplea la secuencia temporal para simular ejecución más rápida o lenta.
    La salida tiene exactamente T frames (se interpola o subsamplea).
    speed_range: (min_factor, max_factor), ej. (0.8, 1.2)
      < 1.0 → más lento (se expande y recorta)
      > 1.0 → más rápido (se comprime y rellena)
    """
    def __init__(self, speed_range: tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        super().__init__(p)
        self.speed_range = speed_range

    def apply(self, xy: np.ndarray) -> np.ndarray:
        T  = xy.shape[0]
        factor = np.random.uniform(*self.speed_range)

        # Número de frames de la señal "acelerada/ralentizada"
        T_new = max(2, int(round(T * factor)))

        original_times = np.linspace(0, T - 1, T)
        new_times      = np.linspace(0, T - 1, T_new)

        K = xy.shape[1]
        resampled = np.zeros((T_new, K, 2), dtype=np.float32)

        for k in range(K):
            for c in range(2):
                f = interp1d(original_times, xy[:, k, c],
                             kind="linear", fill_value="extrapolate")
                resampled[:, k, c] = f(new_times)

        # Volver a T frames recortando o replicando el último frame
        if T_new >= T:
            return resampled[:T]
        else:
            pad = np.tile(resampled[-1:], (T - T_new, 1, 1))
            return np.concatenate([resampled, pad], axis=0)


class TimeWarp(LandmarkAugmentation):
    """
    Deformación temporal suave usando un mapa de tiempo no lineal.
    Simula que partes de la seña se ejecutan más rápido o lento.
    n_anchors: número de puntos de control del warp.
    max_warp  : máxima desviación de cada punto de control (en fracción de T).
    """
    def __init__(self, n_anchors: int = 4, max_warp: float = 0.1, p: float = 0.5):
        super().__init__(p)
        self.n_anchors = n_anchors
        self.max_warp  = max_warp

    def apply(self, xy: np.ndarray) -> np.ndarray:
        T = xy.shape[0]
        if T < 4:
            return xy

        # Puntos de control distribuidos uniformemente
        anchor_x = np.linspace(0, T - 1, self.n_anchors + 2)
        # Perturbación de los puntos interiores (los extremos se quedan fijos)
        perturbation = np.random.uniform(
            -self.max_warp * T,
             self.max_warp * T,
            size=self.n_anchors,
        )
        anchor_y = anchor_x.copy()
        anchor_y[1:-1] += perturbation
        anchor_y = np.clip(anchor_y, 0, T - 1)
        # Garantizar que sea monótonamente creciente
        anchor_y = np.maximum.accumulate(anchor_y)

        # Mapa de tiempo: para cada frame original → frame warpeado
        warp_fn     = interp1d(anchor_x, anchor_y, kind="cubic",
                               fill_value="extrapolate")
        new_times   = np.clip(warp_fn(np.arange(T)), 0, T - 1)

        K = xy.shape[1]
        result = np.zeros_like(xy)
        original_times = np.arange(T, dtype=float)

        for k in range(K):
            for c in range(2):
                f = interp1d(original_times, xy[:, k, c],
                             kind="linear", fill_value="extrapolate")
                result[:, k, c] = f(new_times)

        return result.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  Oclusión sintética
# ──────────────────────────────────────────────────────────────────────────────
class RegionDropout(LandmarkAugmentation):
    """
    Pone a cero una región corporal completa durante una ventana aleatoria
    de frames consecutivos. Simula oclusión parcial.

    regions     : lista de nombres de región a considerar (ver REGIONS)
    max_drop_ratio: máxima fracción de frames a ocluir (ej. 0.3 = 30%)
    """
    def __init__(
        self,
        regions: list[str] | None = None,
        max_drop_ratio: float = 0.3,
        p: float = 0.3,
    ):
        super().__init__(p)
        self.regions        = regions or ["hand_l", "hand_r", "face"]
        self.max_drop_ratio = max_drop_ratio

    def apply(self, xy: np.ndarray) -> np.ndarray:
        T = xy.shape[0]

        # Elegir región y ventana aleatoria
        region_name = np.random.choice(self.regions)
        kpt_indices = REGIONS[region_name]

        n_drop  = max(1, int(T * np.random.uniform(0.05, self.max_drop_ratio)))
        t_start = np.random.randint(0, max(1, T - n_drop))
        t_end   = min(T, t_start + n_drop)

        xy[t_start:t_end, kpt_indices, :] = 0.0
        return xy


# ──────────────────────────────────────────────────────────────────────────────
#  Score-aware
# ──────────────────────────────────────────────────────────────────────────────
class ScoreBasedNoise(LandmarkAugmentation):
    """
    Añade más ruido a los keypoints con score de confianza bajo.
    Degrada realísticamente los puntos menos confiables.

    IMPORTANTE: esta augmentación requiere el tensor completo (T, 133, 3)
    con el canal de score en la posición 2. Devuelve solo (T, 133, 2).

    base_std   : ruido base para keypoints con score=1.0
    max_std    : ruido máximo para keypoints con score=0.0
    score_col  : índice del canal score en el tensor de entrada
    """
    def __init__(
        self,
        base_std:  float = 0.005,
        max_std:   float = 0.05,
        score_col: int   = 2,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.base_std  = base_std
        self.max_std   = max_std
        self.score_col = score_col

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        """
        Entrada: xyz de shape (T, 133, 3)  — x, y, score
        Salida : xy  de shape (T, 133, 2)  — x, y aumentados
        """
        if np.random.random() >= self.p:
            return xyz[:, :, :2]

        xy    = xyz[:, :, :2].copy()
        score = xyz[:, :, self.score_col]         # (T, 133)

        # Normalizar score a [0, 1] si está fuera de rango
        s_min, s_max = score.min(), score.max()
        if s_max > s_min:
            score_norm = (score - s_min) / (s_max - s_min)
        else:
            score_norm = np.ones_like(score)

        # std inversamente proporcional al score: score bajo → más ruido
        std_map = self.base_std + (1.0 - score_norm) * (self.max_std - self.base_std)
        noise   = np.random.normal(0, 1, size=xy.shape).astype(np.float32)
        noise  *= std_map[:, :, np.newaxis]

        return xy + noise

    def apply(self, xy):
        # No se usa directamente — __call__ maneja la lógica completa
        return xy


# ──────────────────────────────────────────────────────────────────────────────
#  Composición
# ──────────────────────────────────────────────────────────────────────────────
class Compose:
    """
    Aplica una lista de augmentaciones en secuencia.
    Cada augmentación decide independientemente si se aplica según su p.
    """
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            xy = t(xy)
        return xy


# ──────────────────────────────────────────────────────────────────────────────
#  Preset recomendado
# ──────────────────────────────────────────────────────────────────────────────
def build_train_augments(
    use_spatial:  bool = True,
    use_temporal: bool = True,
    use_occlusion: bool = True,
) -> Compose:
    """
    Preset de augmentaciones para entrenamiento.
    Las probabilidades están calibradas para no distorsionar demasiado
    con dataset pequeño (~6 muestras/clase).

    Nota: ScoreBasedNoise no se incluye en Compose porque requiere el tensor
    completo (T,133,3). Aplícala manualmente antes de extraer x,y si la necesitas.

    Uso:
        aug = build_train_augments()
        xy_aug = aug(xy)   # xy: numpy (T, 133, 2)
    """
    transforms = []

    if use_spatial:
        transforms += [
            RandomTranslation(max_shift=0.04,  p=0.5),
            RandomScale(scale_range=(0.88, 1.12), p=0.5),
            RandomRotation(max_angle_deg=8.0,  p=0.4),
            GaussianNoise(std=0.008,            p=0.5),
        ]

    if use_temporal:
        transforms += [
            TemporalJitter(max_shift=2,          p=0.4),
            FrameDrop(drop_ratio=0.10,           p=0.4),
            SpeedPerturbation(speed_range=(0.85, 1.15), p=0.5),
            TimeWarp(n_anchors=4, max_warp=0.08, p=0.4),
        ]

    if use_occlusion:
        transforms += [
            RegionDropout(
                regions=["hand_l", "hand_r", "face"],
                max_drop_ratio=0.25,
                p=0.3,
            ),
        ]

    return Compose(transforms)