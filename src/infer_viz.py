"""
infer_viz.py
============
Utilidad de inferencia + visualización para LSMTransformer en Jupyter Notebook.

Muestra una animación inline con:
  - Frames del video original como fondo
  - Overlay del skeleton COCO-WholeBody (59 landmarks filtrados)
  - Predicción del modelo (glosa, confianza) en el título

Uso rápido
----------
from infer_viz import InferenceVisualizer

viz = InferenceVisualizer(
    checkpoint_path = "runs/exp01/best_model.pt",
    parquet_path    = "corpus_LSM_esp/lsm_dataset.parquet",
    video_root      = "../corpus_LSM_esp/Mexican sign language dataset/MSLwords1",
    device          = "cuda",   # o "cpu"
)

# Visualizar N muestras aleatorias del split de test
viz.show_random(n=4, split="test")

# Visualizar un video específico por video_id
viz.show(video_id="01001")
"""

from __future__ import annotations

import io
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

# ─────────────────────────────────────────────────────────────────────────────
# Intentar importar cv2 (OpenCV) para leer frames del video
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    import warnings
    warnings.warn(
        "OpenCV no encontrado (pip install opencv-python). "
        "Se mostrará solo el skeleton sin el video de fondo."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Importar módulos del proyecto (ajusta el path si es necesario)
# ─────────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))          # permite importar src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import (
    LSMDataset, split_dataset, preprocess, collate_fn,
    TMAX, N_LANDMARKS, INPUT_DIM, _KEEP_IDX,
)
from model import LSMTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Definición de conexiones y regiones del skeleton
# ─────────────────────────────────────────────────────────────────────────────

# Índices sobre los 133 landmarks COCO-WholeBody originales
# BODY  : 0-16
# HANDS : 91-132 (izq 91-111, der 112-132)

BODY_CONNECTIONS_133 = [
    (0, 1), (0, 2), (1, 3), (2, 4),                    # cabeza
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),           # brazos
    (5, 11), (6, 12), (11, 12),                         # torso
    (11, 13), (13, 15), (12, 14), (14, 16),             # piernas
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                    # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),                    # índice
    (0, 9), (9, 10), (10, 11), (11, 12),                # medio
    (0, 13), (13, 14), (14, 15), (15, 16),              # anular
    (0, 17), (17, 18), (18, 19), (19, 20),              # meñique
]

HAND_L_OFFSET = 91    # mano izquierda en 133-kpts
HAND_R_OFFSET = 112   # mano derecha   en 133-kpts

HAND_CONNECTIONS_L_133 = [(s + HAND_L_OFFSET, e + HAND_L_OFFSET) for s, e in HAND_CONNECTIONS
                           if s + HAND_L_OFFSET < 133 and e + HAND_L_OFFSET < 133]
HAND_CONNECTIONS_R_133 = [(s + HAND_R_OFFSET, e + HAND_R_OFFSET) for s, e in HAND_CONNECTIONS
                           if s + HAND_R_OFFSET < 133 and e + HAND_R_OFFSET < 133]

# Remapear conexiones a los 59 landmarks filtrados (_KEEP_IDX)
_IDX_MAP = {orig: new for new, orig in enumerate(_KEEP_IDX)}

def _remap(connections_133):
    """Convierte conexiones en espacio-133 a espacio-59."""
    out = []
    for s, e in connections_133:
        if s in _IDX_MAP and e in _IDX_MAP:
            out.append((_IDX_MAP[s], _IDX_MAP[e]))
    return out

BODY_CONNECTIONS  = _remap(BODY_CONNECTIONS_133)
HAND_CONNECTIONS_L = _remap(HAND_CONNECTIONS_L_133)
HAND_CONNECTIONS_R = _remap(HAND_CONNECTIONS_R_133)

# Regiones de colores para los puntos
_BODY_IDX  = list(range(0, 17))        # 0-16 → después del remap siguen igual
_HANDL_IDX = [_IDX_MAP[i] for i in range(HAND_L_OFFSET, HAND_L_OFFSET + 21) if i in _IDX_MAP]
_HANDR_IDX = [_IDX_MAP[i] for i in range(HAND_R_OFFSET, HAND_R_OFFSET + 21) if i in _IDX_MAP]

REGIONS = {
    "body":  (_BODY_IDX,  "dodgerblue", 20),
    "handL": (_HANDL_IDX, "limegreen",  12),
    "handR": (_HANDR_IDX, "tomato",     12),
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: encontrar el archivo de video en el árbol de directorios
# ─────────────────────────────────────────────────────────────────────────────

def _parse_video_id(video_id: str):
    """
    Parsea video_id a (glosa_dir, speaker_dir).

    Formatos soportados:
      'GGG_SSGGG'  → glosa_dir='GGG',  speaker_dir='SSGGG'   (ej. '080_05080')
      'SSGGG'      → glosa_dir='GGG',  speaker_dir='SSGGG'   (ej. '05080')
    """
    if "_" in video_id:
        # Formato nuevo: 'GGG_SSGGG'
        glosa_dir, speaker_dir = video_id.split("_", 1)
    else:
        # Formato legacy: 'SSGGG' (5 dígitos)
        vid = video_id.zfill(5)
        glosa_dir   = vid[2:]   # últimos 3 → número de glosa
        speaker_dir = vid       # todo → carpeta señante
    return glosa_dir, speaker_dir


def _find_video_file(video_root: str, video_id: str) -> Optional[str]:
    """
    Busca el archivo de video en el árbol del dataset.

    Árbol esperado:
        <root>/<GGG>/<SSGGG>/<archivo.mp4|.mov>

    Soporta video_id en formato 'GGG_SSGGG' (ej. '080_05080')
    y en formato legacy 'SSGGG' (ej. '05080').
    """
    root = Path(video_root).resolve()
    glosa_dir, speaker_dir = _parse_video_id(video_id)

    # ── Ruta canónica ─────────────────────────────────────────────────────────
    candidate_dir = root / glosa_dir / speaker_dir
    if candidate_dir.exists():
        for ext in ("*.mp4", "*.mov", "*.avi", "*.MP4", "*.MOV"):
            files = list(candidate_dir.glob(ext))
            if files:
                return str(sorted(files)[0])

    # ── Fallback: buscar por speaker_dir en todo el árbol ────────────────────
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".mp4", ".mov", ".avi"):
            # Coincidencia exacta de la carpeta señante
            if speaker_dir in path.parts:
                return str(path)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Cargar frames del video con OpenCV
# ─────────────────────────────────────────────────────────────────────────────

def _load_video_frames(video_path: str, max_frames: int = TMAX) -> Optional[np.ndarray]:
    """
    Lee hasta max_frames frames del video.
    Intenta múltiples backends de OpenCV para soportar .mov y .mp4.
    Returns: (T, H, W, 3) uint8 RGB  ó  None si falla
    """
    if not _HAS_CV2:
        return None

    # Intentar con el backend por defecto, luego con FFMPEG explícito
    backends = [cv2.CAP_ANY, cv2.CAP_FFMPEG]
    for backend in backends:
        try:
            cap = cv2.VideoCapture(video_path, backend)
        except Exception:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            continue

        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if frames:
            return np.stack(frames)

    return None


def _make_gradient_bg(height: int, width: int, n_frames: int) -> np.ndarray:
    """
    Genera un fondo degradado oscuro para usar cuando no hay video disponible.
    Returns: (n_frames, H, W, 3) uint8
    """
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    # Degradado vertical suave azul oscuro → casi negro
    for y in range(height):
        val = int(20 + 25 * (1 - y / height))
        bg[y, :] = [val // 3, val // 3, val]
    return np.stack([bg] * n_frames)


# ─────────────────────────────────────────────────────────────────────────────
# InferenceVisualizer
# ─────────────────────────────────────────────────────────────────────────────

class InferenceVisualizer:
    """
    Carga el checkpoint, el parquet y genera animaciones de inferencia.

    Parameters
    ----------
    checkpoint_path : ruta al .pt guardado por train.py
    parquet_path    : ruta al .parquet del dataset
    video_root      : raíz del árbol de videos (carpeta MSLwords1)
    device          : 'cuda' o 'cpu'
    score_thresh    : umbral de confianza de landmarks (igual que en train)
    """

    def __init__(
        self,
        checkpoint_path: str,
        parquet_path:    str,
        video_root:      str,
        device:          str  = "cuda",
        score_thresh:    float = 0.3,
    ):
        self.parquet_path  = parquet_path
        self.video_root    = str(Path(video_root).resolve())
        self.score_thresh  = score_thresh
        self.device        = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )

        # ── Cargar checkpoint ─────────────────────────────────────────────────
        print(f"[InferenceVisualizer] Cargando checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg  = ckpt["cfg"]

        self.glosa2idx: Dict[str, int] = ckpt["glosa2idx"]
        self.idx2glosa: Dict[int, str] = {v: k for k, v in self.glosa2idx.items()}
        num_classes = len(self.glosa2idx)

        self.model = LSMTransformer(
            num_classes     = num_classes,
            input_dim       = INPUT_DIM,
            d_model         = cfg.get("d_model", 256),
            nhead           = cfg.get("nhead", 4),
            num_layers      = cfg.get("num_layers", 3),
            dim_feedforward = cfg.get("dim_ff", 512),
            dropout         = cfg.get("dropout", 0.2),
            tmax            = TMAX,
            trigger_window  = cfg.get("trigger_window", 30),
            use_eadm        = cfg.get("use_eadm", True),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"[InferenceVisualizer] {self.model.param_count()}")
        print(f"[InferenceVisualizer] {num_classes} glosas | epoch {ckpt.get('epoch','?')} "
              f"| val F1 {ckpt.get('val_f1', 0):.4f}")

        # ── Cargar parquet (solo columnas necesarias) ─────────────────────────
        print(f"[InferenceVisualizer] Cargando parquet…")
        self.df = pd.read_parquet(
            parquet_path,
            columns=["video_id", "glosa", "intento", "keypoints", "width", "height"],
        ).reset_index(drop=True)
        print(f"[InferenceVisualizer] {len(self.df)} videos disponibles")

        # ── Construir splits para poder filtrar test ──────────────────────────
        self._build_splits(cfg)

    def _build_splits(self, cfg: dict):
        """Re-construye los mismos splits que se usaron en train."""
        split_by_speaker = cfg.get("split_by_speaker", True)
        seed             = cfg.get("seed", 42)

        # Identificar señantes
        # Soporta video_id en formato 'GGG_SSGGG' (ej. '080_05080') o 'SSGGG' (ej. '05080')
        def _extract_speaker(val: str) -> str:
            val = str(val)
            if "_" in val:
                # '080_05080' → parte derecha '05080' → señante '05'
                return val.split("_", 1)[1].zfill(5)[:2]
            return val.zfill(5)[:2]

        self.df["_speaker"] = self.df["video_id"].apply(_extract_speaker)
        all_speakers = sorted(self.df["_speaker"].unique())

        if split_by_speaker:
            n_glosas = self.df["glosa"].nunique()
            umbral   = int(n_glosas * 0.90)
            gps      = self.df.groupby("_speaker")["glosa"].nunique()
            completos = sorted(gps[gps >= umbral].index.tolist())
            val_sp   = [completos[-2]] if len(completos) >= 2 else [completos[-1]]
            test_sp  = [completos[-1]]
            train_sp = [s for s in all_speakers if s not in val_sp and s not in test_sp]
        else:
            # Sin split por señante: usamos mismo seed para reproducir
            rng  = np.random.default_rng(seed)
            all_idx = self.df.index.tolist()
            rng.shuffle(all_idx)
            n = len(all_idx)
            n_tr = int(n * 0.70)
            n_va = int(n * 0.15)
            self._train_ids = set(self.df.iloc[all_idx[:n_tr]]["video_id"])
            self._val_ids   = set(self.df.iloc[all_idx[n_tr:n_tr+n_va]]["video_id"])
            self._test_ids  = set(self.df.iloc[all_idx[n_tr+n_va:]]["video_id"])
            return

        self._train_ids = set(self.df[self.df["_speaker"].isin(train_sp)]["video_id"])
        self._val_ids   = set(self.df[self.df["_speaker"].isin(val_sp)]["video_id"])
        self._test_ids  = set(self.df[self.df["_speaker"].isin(test_sp)]["video_id"])
        print(f"[InferenceVisualizer] Split — train:{len(self._train_ids)} "
              f"val:{len(self._val_ids)} test:{len(self._test_ids)}")

    # ── Inferencia sobre un array de keypoints ────────────────────────────────

    @torch.no_grad()
    def _infer(self, kpts_raw: np.ndarray) -> dict:
        """
        kpts_raw: (T, 133, 3) float32
        Returns dict con top3_labels, top3_probs, end_prob
        """
        xy    = preprocess(kpts_raw, self.score_thresh)   # (T, 59, 2)
        T     = min(xy.shape[0], TMAX)
        x_flat = torch.from_numpy(xy[:T].reshape(T, INPUT_DIM)).unsqueeze(0).to(self.device)
        mask   = torch.ones(1, T, dtype=torch.bool, device=self.device)

        glosa_logits, trigger_logits = self.model(x_flat, mask)
        probs    = F.softmax(glosa_logits, dim=-1)[0].cpu().numpy()
        end_prob = torch.sigmoid(trigger_logits)[0, 0].item()

        top3_idx   = probs.argsort()[::-1][:3]
        top3_labels = [self.idx2glosa[i] for i in top3_idx]
        top3_probs  = probs[top3_idx].tolist()

        return {
            "top1_label": top3_labels[0],
            "top1_conf":  top3_probs[0],
            "top3_labels": top3_labels,
            "top3_probs":  top3_probs,
            "end_prob":    end_prob,
        }

    # ── Animación principal ───────────────────────────────────────────────────

    def _find_pred_video(self, pred_glosa: str, exclude_video_id: str) -> Optional[str]:
        """Busca un video del dataset cuya glosa sea pred_glosa."""
        _, speaker_input = _parse_video_id(exclude_video_id)
        speaker_id = speaker_input.zfill(5)[:2]
        candidates = self.df[
            (self.df["glosa"] == pred_glosa) &
            (self.df["video_id"] != exclude_video_id)
        ]
        if candidates.empty:
            return None
        same = candidates[candidates["video_id"].apply(
            lambda v: _parse_video_id(v)[1].zfill(5)[:2] == speaker_id
        )]
        pool = same if not same.empty else candidates
        return str(pool.iloc[0]["video_id"])

    def _animate(
        self,
        video_id:   str,
        interval:   int  = 80,
        figsize:    Tuple[int, int] = (7, 6),
    ) -> HTML:
        """
        Un solo axes por panel:
          Panel izquierdo → video del input  + skeleton encima
          Panel derecho   → video de la glosa predicha + skeleton encima

        Título muestra GT y Pred con color verde/rojo.
        """
        # ── Cargar datos del video de entrada ─────────────────────────────────
        rows = self.df[self.df["video_id"] == video_id]
        if rows.empty:
            raise ValueError(f"video_id '{video_id}' no encontrado")
        row = rows.iloc[0]

        kpts_raw = np.load(io.BytesIO(row["keypoints"]))
        T_clip   = min(kpts_raw.shape[0], TMAX)
        W, H     = int(row["width"]), int(row["height"])
        glosa_gt = str(row["glosa"])

        # ── Inferencia ────────────────────────────────────────────────────────
        pred       = self._infer(kpts_raw)
        pred_glosa = pred["top1_label"]
        correct    = pred_glosa == glosa_gt
        color_flag = "green" if correct else "red"
        mark       = "✓" if correct else "✗"

        # ── Cargar video de entrada ────────────────────────────────────────────
        video_path  = _find_video_file(self.video_root, video_id)
        frames_in   = None
        if video_path:
            frames_in = _load_video_frames(video_path, max_frames=T_clip)
            if frames_in is not None:
                print(f"  Input video : {video_path} ({len(frames_in)} frames)")

        # ── Cargar video de la glosa predicha ─────────────────────────────────
        pred_vid_id = self._find_pred_video(pred_glosa, video_id)
        frames_pred = None
        if pred_vid_id:
            pred_path = _find_video_file(self.video_root, pred_vid_id)
            if pred_path:
                frames_pred = _load_video_frames(pred_path, max_frames=TMAX)
                if frames_pred is not None:
                    print(f"  Pred  video : {pred_path} ({len(frames_pred)} frames)")

        # ── Skeleton: keypoints tal como los vio el modelo → píxeles ──────────
        xy_model  = preprocess(kpts_raw, self.score_thresh)[:T_clip]
        kpts_clip = kpts_raw[:T_clip, _KEEP_IDX, :]
        scores_c  = kpts_clip[:, :, 2]
        xy_orig   = kpts_clip[:, :, :2].copy()
        xy_orig[scores_c < self.score_thresh] = 0.0
        shoulder_mid = (xy_orig[:, 5, :] + xy_orig[:, 6, :]) / 2.0
        hip_mid_c    = (xy_orig[:, 11, :] + xy_orig[:, 12, :]) / 2.0
        scale        = np.clip(np.linalg.norm(hip_mid_c - shoulder_mid, axis=-1), 1e-8, None)
        xy_px        = xy_model * scale[:, None, None] + shoulder_mid[:, None, :]  # píxeles

        # ── Figura: dos paneles lado a lado ───────────────────────────────────
        vid_aspect = (W / H) if H > 0 else (4/3)
        fig_h = figsize[1]
        fig_w = fig_h * vid_aspect * 2

        fig, (ax_in, ax_pr) = plt.subplots(
            1, 2, figsize=(fig_w, fig_h),
            gridspec_kw={"wspace": 0.05},
        )

        # Subtítulos de cada panel
        ax_in.set_title(f"Original: {glosa_gt}", fontsize=10,
                        color="white", pad=4,
                        bbox=dict(facecolor="#222", alpha=0.7, pad=3))
        ax_pr.set_title(f"Predicho: {pred_glosa}  {mark}",
                        fontsize=10, color=color_flag, pad=4,
                        bbox=dict(facecolor="#222", alpha=0.7, pad=3))

        fig.patch.set_facecolor("black")

        def _setup_ax(ax, first_frame):
            """Configura un axes con imshow de fondo en coordenadas de píxel."""
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_aspect("auto")
            ax.axis("off")
            if first_frame is not None:
                im = ax.imshow(first_frame, extent=[0, W, H, 0],
                               aspect="auto", zorder=0)
            else:
                ax.set_facecolor("#0d1117")
                im = None
            return im

        # Panel izquierdo: video del input
        bg_in = frames_in[0]  if frames_in  is not None else None
        bg_pr = frames_pred[0] if frames_pred is not None else None
        im_in = _setup_ax(ax_in, bg_in)
        im_pr = _setup_ax(ax_pr, bg_pr)

        if bg_pr is None:
            ax_pr.text(0.5, 0.5, f"Sin video\npara '{pred_glosa}'",
                       ha="center", va="center", color="gray",
                       fontsize=10, transform=ax_pr.transAxes)

        # ── Artistas del skeleton (solo panel izquierdo) ───────────────────────
        def _make_skeleton(ax):
            scs = {}
            for region, (indices, color, size) in REGIONS.items():
                scs[region] = ax.scatter([], [], c=color, s=size, zorder=3,
                                         alpha=0.95, linewidths=0)
            bl  = [ax.plot([], [], color="deepskyblue", lw=2.0, alpha=0.85, zorder=2)[0]
                   for _ in BODY_CONNECTIONS]
            hl  = [ax.plot([], [], color="limegreen",   lw=1.5, alpha=0.9,  zorder=2)[0]
                   for _ in HAND_CONNECTIONS_L]
            hr  = [ax.plot([], [], color="tomato",      lw=1.5, alpha=0.9,  zorder=2)[0]
                   for _ in HAND_CONNECTIONS_R]
            return scs, bl, hl, hr

        scs_in, bl_in, hl_in, hr_in = _make_skeleton(ax_in)

        # Contador de frames
        ftxt_in = ax_in.text(W - 5, 5, "", fontsize=7, ha="right", va="top",
                             color="white", zorder=5)
        ftxt_pr = ax_pr.text(W - 5, 5, "", fontsize=7, ha="right", va="top",
                             color="white", zorder=5)

        # Top-3 en esquina inferior del panel predicho
        top3_str = "\n".join(
            f"#{i+1} {lbl}  {p:.1%}"
            for i, (lbl, p) in enumerate(zip(pred["top3_labels"], pred["top3_probs"]))
        )
        ax_pr.text(5, H - 5, top3_str, fontsize=7.5, va="bottom", color="white",
                   zorder=5, bbox=dict(boxstyle="round,pad=0.3",
                                       facecolor="black", alpha=0.6))

        plt.tight_layout(pad=0.3)

        # ── Update ────────────────────────────────────────────────────────────
        T_pred = len(frames_pred) if frames_pred is not None else 0

        def _update(fi):
            # Panel izquierdo: video input + skeleton
            fi_in = min(fi, T_clip - 1)
            if im_in is not None:
                im_in.set_data(frames_in[fi_in])
            ftxt_in.set_text(f"{fi_in+1}/{T_clip}")

            x_px = xy_px[fi_in, :, 0]
            y_px = xy_px[fi_in, :, 1]
            for region, (indices, _, _) in REGIONS.items():
                scs_in[region].set_offsets(np.c_[x_px[indices], y_px[indices]])
            for line, (s, e) in zip(bl_in, BODY_CONNECTIONS):
                line.set_data([x_px[s], x_px[e]], [y_px[s], y_px[e]])
            for line, (s, e) in zip(hl_in, HAND_CONNECTIONS_L):
                line.set_data([x_px[s], x_px[e]], [y_px[s], y_px[e]])
            for line, (s, e) in zip(hr_in, HAND_CONNECTIONS_R):
                line.set_data([x_px[s], x_px[e]], [y_px[s], y_px[e]])

            # Panel derecho: video predicho (sin skeleton, se cicla)
            if im_pr is not None and T_pred > 0:
                fi_pr = fi % T_pred
                im_pr.set_data(frames_pred[fi_pr])
                ftxt_pr.set_text(f"{fi_pr+1}/{T_pred}")

        T_total = max(T_clip, T_pred if T_pred > 0 else T_clip)
        anim = FuncAnimation(fig, _update, frames=T_total,
                             interval=interval, blit=False)
        plt.close()
        return HTML(anim.to_jshtml())

    # ── API pública ───────────────────────────────────────────────────────────

    def show(
        self,
        video_id: str,
        interval: int = 80,
    ) -> None:
        """Muestra la animación para un video_id específico."""
        print(f"\n── {video_id} ──")
        anim = self._animate(video_id, interval=interval)
        display(anim)

    def show_random(
        self,
        n:        int  = 4,
        split:    str  = "test",   # "train", "val" o "test"
        interval: int  = 80,
        seed:     Optional[int] = None,
    ) -> None:
        """
        Muestra n animaciones aleatorias del split indicado.

        Parameters
        ----------
        n        : número de videos a mostrar
        split    : 'train', 'val' o 'test'
        interval : ms entre frames
        seed     : semilla para reproducibilidad
        """
        id_pool = {
            "train": self._train_ids,
            "val":   self._val_ids,
            "test":  self._test_ids,
        }.get(split, self._test_ids)

        candidates = [vid for vid in self.df["video_id"] if vid in id_pool]

        if not candidates:
            print(f"No hay videos en el split '{split}'.")
            return

        rng  = random.Random(seed)
        vids = rng.sample(candidates, min(n, len(candidates)))

        for vid in vids:
            self.show(vid, interval=interval)

    def show_by_glosa(
        self,
        glosa:    str,
        split:    str  = "test",
        interval: int  = 80,
    ) -> None:
        """Muestra todos los videos de una glosa específica en un split."""
        id_pool = {
            "train": self._train_ids,
            "val":   self._val_ids,
            "test":  self._test_ids,
        }.get(split, self._test_ids)

        rows = self.df[
            (self.df["glosa"] == glosa) &
            (self.df["video_id"].isin(id_pool))
        ]
        if rows.empty:
            print(f"No se encontró la glosa '{glosa}' en el split '{split}'.")
            return

        for _, row in rows.iterrows():
            self.show(row["video_id"], interval=interval)

    def evaluate_split(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Corre inferencia en todo el split (o max_samples muestras)
        y devuelve un DataFrame con resultados.

        Columns: video_id, glosa_gt, pred_top1, conf, correct, end_prob
        """
        id_pool = {
            "train": self._train_ids,
            "val":   self._val_ids,
            "test":  self._test_ids,
        }.get(split, self._test_ids)

        rows_split = self.df[self.df["video_id"].isin(id_pool)]
        if max_samples:
            rows_split = rows_split.sample(min(max_samples, len(rows_split)), random_state=0)

        results = []
        for _, row in rows_split.iterrows():
            kpts_raw = np.load(io.BytesIO(row["keypoints"]))
            pred     = self._infer(kpts_raw)
            results.append({
                "video_id": row["video_id"],
                "glosa_gt": row["glosa"],
                "pred_top1": pred["top1_label"],
                "conf":      round(pred["top1_conf"], 4),
                "correct":   pred["top1_label"] == row["glosa"],
                "end_prob":  round(pred["end_prob"], 4),
            })

        df_res = pd.DataFrame(results)
        acc    = df_res["correct"].mean()
        print(f"\n[evaluate_split] {split} — Accuracy: {acc:.3%} ({df_res['correct'].sum()}/{len(df_res)})")
        return df_res