# video_to_landmarks_parquet.py
# ─────────────────────────────────────────────────────────────────────────────
# Igual que video_to_landmarks_rtmw.py pero guarda en Parquet en lugar de CSV.
#
# Formato de salida:
#   Cada fila = 1 video
#   Columnas:
#     video_id        str   → nombre del archivo sin extensión
#     fps             float
#     total_frames    int
#     width           int
#     height          int
#     keypoints       bytes → tensor np.float32 shape (T, 133, 3)  [x, y, score]
#                             serializado con np.save (BytesIO, sin pickle)
#     person_detected bytes → array  np.bool_   shape (T,)
#                             serializado igual
#
# Para cargar / reconstruir el tensor:
#   import pandas as pd, numpy as np, io
#   df   = pd.read_parquet("landmarks.parquet")
#   row  = df.iloc[0]
#   kpts = np.load(io.BytesIO(row["keypoints"]))          # (T, 133, 3)
#   det  = np.load(io.BytesIO(row["person_detected"]))    # (T,)
#
# Instalación:
#   pip install rtmlib opencv-python numpy pandas pyarrow
#
# Uso:
#   python3 video_to_landmarks_parquet.py -i video.mp4 -o landmarks.parquet
#   python3 video_to_landmarks_parquet.py -i video.mp4 -o landmarks.parquet --no-preview
#   python3 video_to_landmarks_parquet.py -i video.mp4 -o landmarks.parquet --device cuda
#   python3 video_to_landmarks_parquet.py -i video.mp4 -o landmarks.parquet --mode performance
#
# Para procesar varios videos en un mismo Parquet:
#   python3 video_to_landmarks_parquet.py -i vid1.mp4 vid2.mp4 -o dataset.parquet
# ─────────────────────────────────────────────────────────────────────────────

import io
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rtmlib import Wholebody

# ─────────────────────────────────────────────
#  Índices COCO-WholeBody 133 keypoints
# ─────────────────────────────────────────────
BODY_IDX   = list(range(0,   17))
FOOT_L_IDX = list(range(17,  21))
FOOT_R_IDX = list(range(21,  25))
FACE_IDX   = list(range(25,  92))
HAND_L_IDX = list(range(92,  113))
HAND_R_IDX = list(range(113, 134))

ALL_IDX    = BODY_IDX + FOOT_L_IDX + FOOT_R_IDX + FACE_IDX + HAND_L_IDX + HAND_R_IDX
N_KPT      = 133   # landmarks totales
N_FEAT     = 3     # x, y, score


# ─────────────────────────────────────────────
#  Serialización  tensor ↔ bytes
# ─────────────────────────────────────────────
def tensor_to_bytes(arr: np.ndarray) -> bytes:
    """Serializa un ndarray a bytes con np.save (sin pickle, portable)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def bytes_to_tensor(b: bytes) -> np.ndarray:
    """Deserializa bytes a ndarray."""
    return np.load(io.BytesIO(b))


# ─────────────────────────────────────────────
#  Preview helpers (igual que el script original)
# ─────────────────────────────────────────────
SKELETON_BODY = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]


def draw_points(img, kpts, scores, indices, color=(0,255,0), radius=4, thresh=0.3):
    if kpts is None:
        return
    for i in indices:
        if i >= len(kpts) or float(scores[i]) < thresh:
            continue
        cv2.circle(img, (int(kpts[i,0]), int(kpts[i,1])), radius, color, -1)


def draw_skeleton(img, kpts, scores, conns, offset=0, color=(255,255,255), thresh=0.3):
    if kpts is None:
        return
    for s, e in conns:
        si, ei = s + offset, e + offset
        if si >= len(kpts) or ei >= len(kpts):
            continue
        if float(scores[si]) < thresh or float(scores[ei]) < thresh:
            continue
        cv2.line(img,
                 (int(kpts[si,0]), int(kpts[si,1])),
                 (int(kpts[ei,0]), int(kpts[ei,1])), color, 1)


def draw_overlay(img, frame_idx, total_frames, fps_proc):
    h, w = img.shape[:2]
    bar_w = int(w * frame_idx / max(total_frames, 1))
    cv2.rectangle(img, (0, h-8), (w, h),     (50,50,50),   -1)
    cv2.rectangle(img, (0, h-8), (bar_w, h), (0,200,100),  -1)
    cv2.putText(img,
                f"frame {frame_idx}/{total_frames}  |  {fps_proc:.1f} fps",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


# ─────────────────────────────────────────────
#  Pipeline por video
# ─────────────────────────────────────────────
def process_one_video(input_path: str, detector, show_preview: bool, mirror: bool) -> dict:
    """
    Procesa un video y devuelve un dict listo para insertar como fila de Parquet.

    Retorna:
    {
      "video_id":        str,
      "fps":             float,
      "total_frames":    int,
      "width":           int,
      "height":          int,
      "keypoints":       bytes,   # tensor float32 (T, 133, 3)
      "person_detected": bytes,   # array  bool    (T,)
    }
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: no se puede abrir '{input_path}'")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_id     = Path(input_path).stem

    print(f"\n▶  {video_id}  |  {vid_w}×{vid_h}  {fps_video:.2f} fps  {total_frames} frames")

    # Pre-aloca el tensor completo → evita listas de Python en el loop
    # Si total_frames es desconocido (stream), crecemos con una lista y concatenamos al final
    if total_frames > 0:
        kpt_tensor = np.zeros((total_frames, N_KPT, N_FEAT), dtype=np.float32)
        det_array  = np.zeros(total_frames, dtype=bool)
    else:
        kpt_tensor = []
        det_array  = []

    prev_tick = cv2.getTickCount()
    fps_proc  = 0.0
    frame_idx = 0
    cancelled = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        # ── Inferencia ────────────────────────────────────────
        keypoints_list, scores_list = detector(frame)

        if keypoints_list is not None and len(keypoints_list) > 0:
            kpts   = keypoints_list[0]   # (133, 2)
            scores = scores_list[0]      # (133,)
            detected = True
        else:
            kpts     = None
            scores   = None
            detected = False

        # ── Guardar en tensor ─────────────────────────────────
        if total_frames > 0:
            # Escritura directa en tensor pre-alocado (sin copias extra)
            if detected:
                kpt_tensor[frame_idx, :, :2] = kpts.astype(np.float32)
                kpt_tensor[frame_idx, :,  2] = scores.astype(np.float32)
            # si no detected → queda en ceros (que es la inicialización)
            det_array[frame_idx] = detected
        else:
            row = np.zeros((N_KPT, N_FEAT), dtype=np.float32)
            if detected:
                row[:, :2] = kpts.astype(np.float32)
                row[:,  2] = scores.astype(np.float32)
            kpt_tensor.append(row)
            det_array.append(detected)

        # ── Preview ───────────────────────────────────────────
        if show_preview:
            vis = frame.copy()
            if detected:
                draw_skeleton(vis, kpts, scores, SKELETON_BODY, color=(255,255,255))
                draw_points(vis, kpts, scores, BODY_IDX,   color=(0,255,0))
                draw_skeleton(vis, kpts, scores, HAND_CONNECTIONS, offset=92,  color=(180,100,255))
                draw_skeleton(vis, kpts, scores, HAND_CONNECTIONS, offset=113, color=(180,100,255))
                draw_points(vis, kpts, scores, HAND_L_IDX, color=(180,0,255), radius=3)
                draw_points(vis, kpts, scores, HAND_R_IDX, color=(255,100,0), radius=3)
                draw_points(vis, kpts, scores, FACE_IDX,   color=(0,200,255), radius=2)

            now      = cv2.getTickCount()
            fps_proc = 0.9 * fps_proc + 0.1 * cv2.getTickFrequency() / (now - prev_tick)
            prev_tick = now
            draw_overlay(vis, frame_idx, total_frames, fps_proc)

            cv2.imshow(f"RTMPose — {video_id}", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                cancelled = True
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  {frame_idx}/{total_frames}  ({pct:.1f}%)", end="\r")

    cap.release()

    if cancelled:
        print(f"\n  ⚠  Cancelado en frame {frame_idx}")

    # Si total_frames era desconocido, apila la lista
    if not isinstance(kpt_tensor, np.ndarray):
        kpt_tensor = np.stack(kpt_tensor, axis=0)  # (T, 133, 3)
        det_array  = np.array(det_array, dtype=bool)
    else:
        # Recorta al número real de frames leídos (por si el header era incorrecto)
        kpt_tensor = kpt_tensor[:frame_idx]
        det_array  = det_array[:frame_idx]

    print(f"  ✓  {frame_idx} frames  →  tensor {kpt_tensor.shape}  "
          f"({kpt_tensor.nbytes / 1024**2:.2f} MB sin comprimir)")

    return {
        "video_id":        video_id,
        "fps":             fps_video,
        "total_frames":    frame_idx,
        "width":           vid_w,
        "height":          vid_h,
        "keypoints":       tensor_to_bytes(kpt_tensor),
        "person_detected": tensor_to_bytes(det_array),
    }


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Extrae 133 landmarks wholebody a Parquet con RTMPose (rtmlib)."
    )
    p.add_argument("--input",      "-i", required=True, nargs="+",
                   help="Video(s) de entrada (acepta múltiples)")
    p.add_argument("--output",     "-o", required=True,
                   help="Parquet de salida (.parquet)")
    p.add_argument("--no-preview", action="store_true",
                   help="Sin ventana de preview")
    p.add_argument("--mirror",     "-m", action="store_true",
                   help="Espejo horizontal")
    p.add_argument("--device",     "-d", default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--mode",       default="balanced",
                   choices=["performance", "balanced", "lightweight"])
    p.add_argument("--append",     action="store_true",
                   help="Agrega filas a un Parquet existente en vez de sobreescribir")
    args = p.parse_args()

    print("Cargando modelo RTMW...")
    detector = Wholebody(mode=args.mode, backend="onnxruntime", device=args.device)
    print("Modelo cargado.\n")

    rows = []
    for vid_path in args.input:
        result = process_one_video(
            input_path   = vid_path,
            detector     = detector,
            show_preview = not args.no_preview,
            mirror       = args.mirror,
        )
        if result is not None:
            rows.append(result)

    cv2.destroyAllWindows()

    if not rows:
        print("No se procesó ningún video.")
        sys.exit(1)

    # ── Schema Parquet explícito ──────────────────────────────
    # Los tensores van como BINARY (bytes). PyArrow los guarda
    # con compresión ZSTD por defecto → muy buen ratio en floats.
    schema = pa.schema([
        pa.field("video_id",        pa.string()),
        pa.field("fps",             pa.float32()),
        pa.field("total_frames",    pa.int32()),
        pa.field("width",           pa.int32()),
        pa.field("height",          pa.int32()),
        pa.field("keypoints",       pa.binary()),   # (T, 133, 3) float32
        pa.field("person_detected", pa.binary()),   # (T,)        bool
    ])

    new_table = pa.Table.from_pylist(rows, schema=schema)

    if args.append and Path(args.output).exists():
        old_table = pq.read_table(args.output)
        final_table = pa.concat_tables([old_table, new_table])
    else:
        final_table = new_table

    pq.write_table(
        final_table,
        args.output,
        compression="zstd",          # mejor ratio que snappy para floats
        compression_level=3,         # 1-22; 3 es buen balance velocidad/ratio
    )

    print(f"\n✅  Parquet guardado → {args.output}")
    print(f"   Filas  : {len(final_table)}")
    print(f"   Tamaño : {Path(args.output).stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()