"""
record_and_infer.py
===================
Flujo:
    1. Cámara siempre abierta, mostrando landmarks en vivo.
    2. Presiona  ESPACIO  →  empieza a grabar frames.
    3. Presiona  ESPACIO  →  para de grabar.
    4. El modelo corre automáticamente sobre lo grabado.
    5. La glosa resultante se muestra en pantalla (la cámara no se cierra).
    6. Repite desde el paso 2 cuantas veces quieras.
    7. Presiona  Q / ESC  para salir.

Uso:
    # Demo sin nada (pesos aleatorios, labels ficticias)
    python record_and_infer.py

    # Con modelo real
    python record_and_infer.py --checkpoint modelo.pt --labels glosas.txt
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

try:
    from model import LSMTransformer
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from model import LSMTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

TMAX       = 200
N_LM       = 133
INPUT_DIM  = 266   # 133 * 2  (solo x, y)

# Colores BGR
C_WHITE  = (240, 240, 240)
C_GREEN  = (60,  210, 80)
C_RED    = (50,  50,  220)
C_YELLOW = (30,  210, 230)
C_CYAN   = (220, 210, 40)
C_DIM    = (110, 110, 120)
C_BG     = (18,  18,  22)


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe → COCO-WholeBody 133
# ─────────────────────────────────────────────────────────────────────────────

MP_POSE_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def extract_landmarks(results, img_w: int, img_h: int) -> np.ndarray:
    """MediaPipe Holistic results → (133, 3) float32."""
    kpts = np.zeros((N_LM, 3), dtype=np.float32)

    if results.pose_landmarks:
        for ci, mi in enumerate(MP_POSE_TO_COCO17):
            lm = results.pose_landmarks.landmark[mi]
            kpts[ci] = [lm.x * img_w, lm.y * img_h, getattr(lm, "visibility", 1.0)]

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            kpts[91 + i] = [lm.x * img_w, lm.y * img_h, 1.0]

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            kpts[112 + i] = [lm.x * img_w, lm.y * img_h, 1.0]

    return kpts


# ─────────────────────────────────────────────────────────────────────────────
# Preprocesamiento (igual que dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

S_L, S_R = 5, 6
H_L, H_R = 11, 12


def preprocess_and_pad(frames: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lista de (133,3) → tensores (1, TMAX, 266) y (1, TMAX) para el modelo.
    """
    T   = min(len(frames), TMAX)
    seq = np.stack(frames[:T]).astype(np.float32)   # (T, 133, 3)

    # Baja confianza → 0
    seq[seq[:, :, 2] < 0.3] = 0.0

    # Centrar en hombros
    smid = (seq[:, S_L, :2] + seq[:, S_R, :2]) / 2
    seq[:, :, :2] -= smid[:, None, :]

    # Escala hombros→caderas
    hmid  = (seq[:, H_L, :2] + seq[:, H_R, :2]) / 2
    scale = np.linalg.norm(hmid - smid, axis=-1).clip(1e-6)
    seq[:, :, :2] /= scale[:, None, None]

    # Solo x,y → aplanar
    flat = seq[:, :, :2].reshape(T, INPUT_DIM)   # (T, 266)

    padded = np.zeros((TMAX, INPUT_DIM), dtype=np.float32)
    padded[:T] = flat
    valid = np.zeros(TMAX, dtype=bool)
    valid[:T] = True

    x     = torch.from_numpy(padded).unsqueeze(0)   # (1, TMAX, 266)
    vmask = torch.from_numpy(valid).unsqueeze(0)    # (1, TMAX)
    return x, vmask


# ─────────────────────────────────────────────────────────────────────────────
# Modelo
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: Optional[str], num_classes: int,
               device: torch.device) -> Tuple[LSMTransformer, bool, Optional[List[str]]]:
    """
    Returns (model, demo_mode, labels_from_ckpt).
    Lee la cfg guardada en el checkpoint para reconstruir el modelo
    con los hiperparámetros exactos con los que fue entrenado.
    """
    labels_from_ckpt = None
    demo = True

    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

        # ── 1. glosa2idx ───────────────────────────────────────────────────
        ckpt_num_classes = num_classes
        if isinstance(ckpt, dict) and "glosa2idx" in ckpt:
            g2i = ckpt["glosa2idx"]
            labels_from_ckpt = [g for g, _ in sorted(g2i.items(), key=lambda x: x[1])]
            ckpt_num_classes  = len(labels_from_ckpt)
            print(f"[INFO] glosa2idx del checkpoint — {ckpt_num_classes} clases.")

        # ── 2. Hiperparámetros desde cfg (si existe) ───────────────────────
        # Defaults seguros — se sobreescriben con lo que haya en cfg
        # ── 2. Extraer state_dict primero ─────────────────────────────────
        if isinstance(ckpt, dict):
            state = (ckpt.get("model_state")
                     or ckpt.get("model_state_dict")
                     or ckpt.get("state_dict")
                     or ckpt)
        else:
            state = ckpt

        # ── 3. Inferir TODOS los hiperparámetros directo desde los pesos ──
        # Esto es 100% confiable — la cfg puede tener valores incorrectos
        hp = dict(
            num_classes     = ckpt_num_classes,
            input_dim       = INPUT_DIM,
            d_model         = 512,
            nhead           = 8,
            num_layers      = 6,
            dim_feedforward = 2048,
            dropout         = 0.1,
            tmax            = TMAX,
            trigger_window  = 30,
            use_eadm        = False,
        )

        if "embedding.0.weight" in state:
            hp["d_model"] = state["embedding.0.weight"].shape[0]

        n_layers = sum(1 for k in state
                       if k.startswith("transformer.layers.")
                       and k.endswith(".norm1.weight"))
        if n_layers > 0:
            hp["num_layers"] = n_layers

        if "transformer.layers.0.linear1.weight" in state:
            hp["dim_feedforward"] = state["transformer.layers.0.linear1.weight"].shape[0]

        if "trigger_head.net.0.weight" in state:
            hp["trigger_window"] = state["trigger_head.net.0.weight"].shape[1] // hp["d_model"]

        # nhead: inferir desde in_proj_weight (shape = [3*d_model, d_model])
        # No es directamente inferible, usar cfg si existe, sino default 8
        if isinstance(ckpt, dict) and "cfg" in ckpt:
            cfg = ckpt["cfg"]
            get = (lambda k, d: cfg.get(k, d)) if isinstance(cfg, dict) \
                  else (lambda k, d: getattr(cfg, k, d))
            hp["nhead"] = get("nhead", hp["nhead"])

        print(f"[INFO] Hiperparámetros inferidos desde pesos — "
              f"d_model={hp['d_model']} | layers={hp['num_layers']} | "
              f"ffn={hp['dim_feedforward']} | trigger_window={hp['trigger_window']} | "
              f"nhead={hp['nhead']}")

        # ── 4. Construir modelo y cargar pesos ─────────────────────────────
        model = LSMTransformer(**hp)
        model.load_state_dict(state, strict=True)
        print(f"[INFO] Checkpoint cargado OK: {checkpoint}")
        demo = False

    else:
        print("[WARN] Sin checkpoint — modo DEMO (pesos aleatorios).")
        model = LSMTransformer(
            num_classes=num_classes, input_dim=INPUT_DIM,
            d_model=512, nhead=8, num_layers=6,
            dim_feedforward=2048, dropout=0.1,
            tmax=TMAX, trigger_window=30, use_eadm=False,
        )

    return model.to(device).eval(), demo, labels_from_ckpt


def load_labels(path: Optional[str], n: int) -> List[str]:
    if path and Path(path).exists():
        labels = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
        print(f"[INFO] {len(labels)} glosas cargadas.")
        return labels
    print(f"[WARN] Sin labels — usando etiquetas ficticias.")
    return [f"GLOSA_{i:03d}" for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia sobre los frames grabados
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    frames: List[np.ndarray],
    model: LSMTransformer,
    labels: List[str],
    device: torch.device,
) -> Tuple[str, float, float, List[Tuple[str, float]]]:
    """
    Corre el modelo sobre la grabación.

    Returns
    -------
    best_name  : nombre de la glosa #1
    best_conf  : confianza #1
    end_prob   : probabilidad de fin de seña
    top3       : [(nombre, conf), ...] top 3
    """
    x, vmask = preprocess_and_pad(frames)
    x, vmask = x.to(device), vmask.to(device)

    with torch.no_grad():
        glosa_logits, trigger_logits = model(x, vmask)

    probs    = F.softmax(glosa_logits, dim=-1).squeeze(0).cpu().numpy()
    end_prob = float(torch.sigmoid(trigger_logits).item())

    top3_idx = np.argsort(probs)[::-1][:3]
    top3     = [(labels[i] if i < len(labels) else f"clase_{i}", float(probs[i]))
                for i in top3_idx]

    return top3[0][0], top3[0][1], end_prob, top3


# ─────────────────────────────────────────────────────────────────────────────
# Overlay de resultado (se dibuja sobre el frame durante N segundos)
# ─────────────────────────────────────────────────────────────────────────────

def draw_result_overlay(
    frame: np.ndarray,
    top3: List[Tuple[str, float]],
    end_prob: float,
    elapsed: float,         # segundos desde que se mostró el resultado
    display_secs: float,    # cuántos segundos mostrarlo
    demo_mode: bool,
) -> np.ndarray:
    """Dibuja el resultado de la última inferencia centrado en el frame."""
    h, w = frame.shape[:2]
    alpha = max(0.0, 1.0 - (elapsed / display_secs) ** 2)   # fade-out suave

    if alpha <= 0.01:
        return frame

    overlay = frame.copy()

    # Caja central semi-transparente
    bx, by, bw, bh = w // 2 - 220, h // 2 - 130, 440, 270
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), C_BG, -1)
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), C_YELLOW, 2)

    cv2.addWeighted(overlay, alpha * 0.85, frame, 1 - alpha * 0.85, 0, frame)

    def txt(text, x, y, scale=0.6, color=C_WHITE, thick=1):
        # Re-dibujar con alpha manual no es trivial en OpenCV;
        # dibujamos sobre frame directamente (ya fusionado arriba)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    cx = bx + bw // 2

    y = by + 35
    demo_tag = "  [DEMO]" if demo_mode else ""
    label = "Resultado" + demo_tag
    lsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0]
    txt(label, cx - lsize[0] // 2, y, scale=0.62, color=C_YELLOW, thick=2); y += 32

    cv2.line(frame, (bx + 16, y), (bx + bw - 16, y), (60, 60, 70), 1); y += 22

    # Glosa principal
    name0, conf0 = top3[0]
    display = name0 if len(name0) <= 20 else name0[:18] + "…"
    nsize = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
    txt(display, cx - nsize[0] // 2, y + nsize[1], scale=1.1, color=C_GREEN, thick=3); y += 55

    conf_txt = f"Confianza: {conf0:.0%}"
    csize = cv2.getTextSize(conf_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    txt(conf_txt, cx - csize[0] // 2, y, scale=0.55, color=C_DIM); y += 26

    cv2.line(frame, (bx + 16, y), (bx + bw - 16, y), (50, 50, 60), 1); y += 18

    # Top-2 y Top-3
    for rank, (name, conf) in enumerate(top3[1:], start=2):
        col = C_YELLOW if rank == 2 else C_DIM
        line = f"#{rank}  {name[:18]}   {conf:.0%}"
        lsz  = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
        txt(line, cx - lsz[0] // 2, y, scale=0.50, color=col); y += 22

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# HUD de estado (siempre visible en la esquina)
# ─────────────────────────────────────────────────────────────────────────────

def draw_status_bar(frame: np.ndarray, state: str, n_frames: int, demo: bool):
    """Banda inferior con el estado actual."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 38), (w, h), (10, 10, 14), -1)
    cv2.line(frame, (0, h - 38), (w, h - 38), (50, 50, 60), 1)

    def txt(text, x, color=C_WHITE, scale=0.52, thick=1):
        cv2.putText(frame, text, (x, h - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    if state == "IDLE":
        txt("●  Listo  —  ESPACIO para grabar", 14, color=C_DIM)
    elif state == "RECORDING":
        txt(f"⏺  Grabando...  {n_frames} frames  —  ESPACIO para detener",
            14, color=C_RED, thick=2)
    elif state == "PROCESSING":
        txt("⚙  Procesando inferencia...", 14, color=C_YELLOW)

    if demo:
        txt("[DEMO]", w - 90, color=(80, 80, 180), scale=0.44)

    txt("Q / ESC  salir", w - 175, color=(70, 70, 80), scale=0.42)


# ─────────────────────────────────────────────────────────────────────────────
# Bucle principal
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        args.device if args.device != "auto" else "cpu"
    )
    print(f"[INFO] Dispositivo: {device}")

    # Labels desde archivo (puede ser None si no existe)
    labels_file = load_labels(args.labels, args.num_classes)

    # Modelo — puede traer su propio glosa2idx embebido
    model, demo_mode, labels_ckpt = load_model(args.checkpoint, len(labels_file), device)

    # Prioridad: glosa2idx del checkpoint > glosas.txt > ficticias
    if labels_ckpt is not None:
        labels = labels_ckpt
        print(f"[INFO] Usando labels del checkpoint ({len(labels)} glosas).")
    else:
        labels = labels_file

    print(model.param_count())

    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic       = mp.solutions.holistic

    # Estado de la máquina
    # IDLE → RECORDING → PROCESSING → IDLE
    state: str = "IDLE"

    recorded_frames: List[np.ndarray] = []   # frames (133,3) capturados

    # Resultado a mostrar
    last_top3:    List[Tuple[str, float]] = []
    last_end_prob: float = 0.0
    result_shown_at: Optional[float] = None
    DISPLAY_SECS = 4.0   # segundos que permanece el resultado en pantalla

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir cámara {args.camera}")
        return

    # Crear la ventana UNA sola vez antes del loop
    WIN = "LSM Grabacion e inferencia"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 540)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
    ) as holistic:

        while cap.isOpened():
            ok, image = cap.read()
            if not ok:
                continue

            image = cv2.flip(image, 1)
            img_h, img_w = image.shape[:2]

            # ── MediaPipe ──────────────────────────────────────────────────────
            image.flags.writeable = False
            rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            image.flags.writeable = True

            # ── Grabar si estamos en RECORDING ─────────────────────────────────
            if state == "RECORDING":
                kpts = extract_landmarks(results, img_w, img_h)
                recorded_frames.append(kpts)

                # Límite de seguridad: parar automáticamente al llegar a TMAX
                if len(recorded_frames) >= TMAX:
                    state = "PROCESSING"

            # ── Inferencia (se ejecuta un ciclo cuando state=PROCESSING) ───────
            if state == "PROCESSING":
                print(f"[INFO] Inferenciando {len(recorded_frames)} frames…")
                best_name, best_conf, end_prob, top3 = run_inference(
                    recorded_frames, model, labels, device
                )
                print(f"[RESULTADO]  {best_name}  conf={best_conf:.2f}  trigger={end_prob:.2f}")
                print(f"  Top-3: {top3}")

                last_top3       = top3
                last_end_prob   = end_prob
                result_shown_at = time.perf_counter()
                recorded_frames = []
                state = "IDLE"

            # ── Dibujar landmarks sobre el frame ───────────────────────────────
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                vis, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    vis, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    vis, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Borde rojo parpadeante durante grabación
            if state == "RECORDING":
                thickness = 6 if (int(time.perf_counter() * 3) % 2 == 0) else 3
                cv2.rectangle(vis, (0, 0), (img_w - 1, img_h - 1), C_RED, thickness)

            # ── Overlay de resultado ───────────────────────────────────────────
            if last_top3 and result_shown_at is not None:
                elapsed = time.perf_counter() - result_shown_at
                vis = draw_result_overlay(
                    vis, last_top3, last_end_prob,
                    elapsed, DISPLAY_SECS, demo_mode,
                )
                if elapsed > DISPLAY_SECS:
                    last_top3 = []

            # ── Barra de estado inferior ───────────────────────────────────────
            draw_status_bar(vis, state, len(recorded_frames), demo_mode)

            cv2.imshow(WIN, vis)

            # ── Teclas ─────────────────────────────────────────────────────────
            key = cv2.waitKey(5) & 0xFF

            if key in (27, ord("q")):
                break

            elif key == ord(" "):
                if state == "IDLE":
                    recorded_frames = []
                    state = "RECORDING"
                    print("[INFO] ▶ Grabación iniciada.")

                elif state == "RECORDING":
                    if len(recorded_frames) < 5:
                        print("[WARN] Grabación muy corta (< 5 frames), ignorada.")
                        recorded_frames = []
                        state = "IDLE"
                    else:
                        state = "PROCESSING"
                        print(f"[INFO] ■ Grabación detenida — {len(recorded_frames)} frames.")

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Graba una seña con ESPACIO y obtén la glosa automáticamente."
    )
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Ruta al .pt entrenado (omitir = modo demo)")
    p.add_argument("--labels",      type=str, default=None,
                   help="Archivo .txt con una glosa por línea (omitir = ficticias)")
    p.add_argument("--camera",      type=int, default=0)
    p.add_argument("--device",      type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--num_classes", type=int, default=249)
    args = p.parse_args()
    run(args)
