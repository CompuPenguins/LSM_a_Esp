"""
video_to_landmarks_rtmw.py
──────────────────────────
Procesa un video frame a frame con RTMPose Wholebody (rtmlib)
y exporta series de tiempo de landmarks a CSV.

Keypoints exportados (133 en total, formato COCO-WholeBody):
  - Cuerpo  : índices  0–16   (17 puntos)
  - Pie izq : índices 17–20   ( 4 puntos)
  - Pie der : índices 21–24   ( 4 puntos)
  - Cara    : índices 25–91   (68 puntos)
  - Mano izq: índices 92–112  (21 puntos)
  - Mano der: índices 113–133 (21 puntos)

Uso:
    python3 video_to_landmarks_rtmw.py -i video.mp4 -o landmarks.csv
    python3 video_to_landmarks_rtmw.py -i video.mp4 -o landmarks.csv --no-preview
    python3 video_to_landmarks_rtmw.py -i video.mp4 -o landmarks.csv --device cuda
    python3 video_to_landmarks_rtmw.py -i video.mp4 -o landmarks.csv --mode performance
"""

import cv2
import csv
import argparse
import sys
import numpy as np
from rtmlib import Wholebody

# ─────────────────────────────────────────────
#  Índices COCO-WholeBody 133 keypoints
# ─────────────────────────────────────────────
BODY_IDX      = list(range(0,  17))
FOOT_L_IDX    = list(range(17, 21))
FOOT_R_IDX    = list(range(21, 25))
FACE_IDX      = list(range(25, 92))
HAND_L_IDX    = list(range(92,  113))
HAND_R_IDX    = list(range(113, 134))

# Puntos de cara equivalentes a los que usábamos en MediaPipe
# (en COCO-WholeBody los ojos/labios están en la región 25-91)
# Mantenemos toda la cara exportada para máxima información.

# ─────────────────────────────────────────────
#  CSV
# ─────────────────────────────────────────────
def build_csv_header():
    cols = ["frame", "time_s", "person_detected"]

    regions = [
        ("body",   BODY_IDX),
        ("foot_l", FOOT_L_IDX),
        ("foot_r", FOOT_R_IDX),
        ("face",   FACE_IDX),
        ("hand_l", HAND_L_IDX),
        ("hand_r", HAND_R_IDX),
    ]
    for region_name, indices in regions:
        for i in indices:
            cols += [
                f"{region_name}_lm{i:03d}_x",
                f"{region_name}_lm{i:03d}_y",
                f"{region_name}_lm{i:03d}_score",
            ]
    return cols


def keypoints_to_row(keypoints, scores, indices):
    """Extrae x, y, score para los índices dados. Devuelve None si no hay detección."""
    row = []
    if keypoints is None:
        return [None] * (len(indices) * 3)
    for i in indices:
        if i < len(keypoints):
            x, y   = keypoints[i]
            score  = float(scores[i]) if scores is not None else 0.0
            row += [round(float(x), 4), round(float(y), 4), round(score, 4)]
        else:
            row += [None, None, None]
    return row


# ─────────────────────────────────────────────
#  Preview
# ─────────────────────────────────────────────
SKELETON_BODY = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

def draw_points(img, keypoints, scores, indices,
                color=(0, 255, 0), radius=4, score_thresh=0.3):
    if keypoints is None:
        return
    for i in indices:
        if i >= len(keypoints):
            continue
        score = float(scores[i]) if scores is not None else 1.0
        if score < score_thresh:
            continue
        x, y = int(keypoints[i][0]), int(keypoints[i][1])
        cv2.circle(img, (x, y), radius, color, -1)

def draw_skeleton(img, keypoints, scores, connections, global_offset=0,
                  color=(255, 255, 255), score_thresh=0.3):
    if keypoints is None:
        return
    for s, e in connections:
        si, ei = s + global_offset, e + global_offset
        if si >= len(keypoints) or ei >= len(keypoints):
            continue
        if scores is not None:
            if float(scores[si]) < score_thresh or float(scores[ei]) < score_thresh:
                continue
        x1, y1 = int(keypoints[si][0]), int(keypoints[si][1])
        x2, y2 = int(keypoints[ei][0]), int(keypoints[ei][1])
        cv2.line(img, (x1, y1), (x2, y2), color, 1)


def draw_overlay(img, frame_idx, total_frames, fps_proc):
    h, w = img.shape[:2]
    pct   = frame_idx / max(total_frames, 1)
    bar_w = int(w * pct)
    cv2.rectangle(img, (0, h-8), (w, h),      (50, 50, 50),  -1)
    cv2.rectangle(img, (0, h-8), (bar_w, h),  (0, 200, 100), -1)
    cv2.putText(img,
                f"frame {frame_idx}/{total_frames}  |  {fps_proc:.1f} fps",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ─────────────────────────────────────────────
#  Pipeline principal
# ─────────────────────────────────────────────
def process_video(input_path, output_csv, show_preview, mirror, device, mode):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: no se puede abrir '{input_path}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video   : {input_path}")
    print(f"          {vid_w}×{vid_h}  {fps_video:.2f} fps  {total_frames} frames")
    print(f"CSV     : {output_csv}")
    print(f"Device  : {device}  |  Mode: {mode}")
    print("Cargando modelo RTMW... (primera vez descarga ~100-200 MB)")

    # mode: 'performance' usa RTMW-x (más preciso), 'balanced' usa RTMW-m
    detector = Wholebody(
        mode=mode,
        backend='onnxruntime',
        device=device,
    )
    print("Modelo cargado.")

    if show_preview:
        print("Preview activo — ESC / Q para cancelar.")

    csv_file = open(output_csv, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(build_csv_header())

    prev_tick = cv2.getTickCount()
    fps_proc  = 0.0
    frame_idx = 0
    cancelled = False

    regions = [
        ("body",   BODY_IDX),
        ("foot_l", FOOT_L_IDX),
        ("foot_r", FOOT_R_IDX),
        ("face",   FACE_IDX),
        ("hand_l", HAND_L_IDX),
        ("hand_r", HAND_R_IDX),
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        # ── Inferencia ────────────────────────────────────────
        # rtmlib devuelve listas: keypoints[person], scores[person]
        keypoints_list, scores_list = detector(frame)

        # Tomamos la primera persona detectada (más prominente)
        if keypoints_list is not None and len(keypoints_list) > 0:
            kpts   = keypoints_list[0]   # shape (133, 2)
            scores = scores_list[0]      # shape (133,)
            person_detected = 1
        else:
            kpts   = None
            scores = None
            person_detected = 0

        # ── Fila CSV ──────────────────────────────────────────
        time_s = frame_idx / fps_video
        row    = [frame_idx, round(time_s, 5), person_detected]

        for _, indices in regions:
            row += keypoints_to_row(kpts, scores, indices)

        writer.writerow(row)

        # ── Preview ───────────────────────────────────────────
        if show_preview:
            vis = frame.copy()
            if kpts is not None:
                # Cuerpo
                draw_skeleton(vis, kpts, scores, SKELETON_BODY,
                              color=(255, 255, 255))
                draw_points(vis, kpts, scores, BODY_IDX,
                            color=(0, 255, 0))
                # Manos
                draw_skeleton(vis, kpts, scores, HAND_CONNECTIONS,
                              global_offset=92, color=(180, 100, 255))
                draw_skeleton(vis, kpts, scores, HAND_CONNECTIONS,
                              global_offset=113, color=(180, 100, 255))
                draw_points(vis, kpts, scores, HAND_L_IDX,
                            color=(180, 0, 255), radius=3)
                draw_points(vis, kpts, scores, HAND_R_IDX,
                            color=(255, 100, 0), radius=3)
                # Cara
                draw_points(vis, kpts, scores, FACE_IDX,
                            color=(0, 200, 255), radius=2)

            now      = cv2.getTickCount()
            fps_proc = 0.9 * fps_proc + 0.1 * cv2.getTickFrequency() / (now - prev_tick)
            prev_tick = now
            draw_overlay(vis, frame_idx, total_frames, fps_proc)

            cv2.imshow("RTMPose Wholebody", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                cancelled = True
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  {frame_idx}/{total_frames}  ({pct:.1f}%)", end="\r")

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    status = f"Cancelado en frame {frame_idx}" if cancelled else f"Listo. {frame_idx} frames"
    print(f"\n{status}  →  {output_csv}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extrae 133 landmarks wholebody a CSV con RTMPose (rtmlib)."
    )
    p.add_argument("--input",      "-i", required=True,
                   help="Video de entrada")
    p.add_argument("--output",     "-o", required=True,
                   help="CSV de salida")
    p.add_argument("--no-preview", action="store_true",
                   help="Sin ventana de preview")
    p.add_argument("--mirror",     "-m", action="store_true",
                   help="Espejo horizontal")
    p.add_argument("--device",     "-d", default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="Device de inferencia (default: cpu)")
    p.add_argument("--mode",       default="balanced",
                   choices=["performance", "balanced", "lightweight"],
                   help="Tamaño del modelo: performance=RTMW-x, balanced=RTMW-m, lightweight=RTMW-s (default: balanced)")
    args = p.parse_args()

    process_video(
        input_path   = args.input,
        output_csv   = args.output,
        show_preview = not args.no_preview,
        mirror       = args.mirror,
        device       = args.device,
        mode         = args.mode,
    )
