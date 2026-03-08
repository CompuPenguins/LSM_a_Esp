# build_lsm_dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# Recorre toda la estructura MSLwords1/{glosa}/{intento}/video.{mov,mp4}
# y genera un único Parquet donde cada fila = 1 video.
#
# Estructura esperada:
#   MSLwords1/
#     001/
#       01001/   ← intento
#         video.mov   (o .mp4)
#       02001/
#         ...
#     002/
#       ...
#
# Columnas del Parquet:
#   glosa           str   → nombre del directorio de la palabra  (ej. "001")
#   intento         str   → nombre del directorio del intento    (ej. "01001")
#   video_id        str   → "{glosa}_{intento}"
#   video_path      str   → ruta relativa al video original
#   fps             float32
#   total_frames    int32
#   width           int32
#   height          int32
#   keypoints       bytes → tensor float32 (T, 133, 3)  [x, y, score]
#   person_detected bytes → array  bool    (T,)
#
# Uso básico:
#   python3 build_lsm_dataset.py --root /ruta/a/MSLwords1 --output lsm_dataset.parquet
#
# Opciones:
#   --device   cpu | cuda | mps          (default: cpu)
#   --mode     performance | balanced | lightweight  (default: balanced)
#   --workers  N                         (default: 1 — solo 1 en GPU, >1 solo con cpu)
#   --resume                             Salta videos cuyo video_id ya esté en el parquet
#   --batch-size N                       Filas a acumular antes de hacer flush a disco (default: 50)
#
# Instalación:
#   pip install rtmlib opencv-python numpy pandas pyarrow tqdm
# ─────────────────────────────────────────────────────────────────────────────

import io
import sys
import argparse
import traceback
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ─────────────────────────────────────────────
#  Constantes COCO-WholeBody
# ─────────────────────────────────────────────
N_KPT  = 133
N_FEAT = 3   # x, y, score

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}

# ─────────────────────────────────────────────
#  Schema Parquet
# ─────────────────────────────────────────────
SCHEMA = pa.schema([
    pa.field("glosa",           pa.string()),
    pa.field("intento",         pa.string()),
    pa.field("video_id",        pa.string()),
    pa.field("video_path",      pa.string()),
    pa.field("fps",             pa.float32()),
    pa.field("total_frames",    pa.int32()),
    pa.field("width",           pa.int32()),
    pa.field("height",          pa.int32()),
    pa.field("keypoints",       pa.binary()),   # (T, 133, 3) float32
    pa.field("person_detected", pa.binary()),   # (T,)        bool
])


# ─────────────────────────────────────────────
#  Helpers de serialización
# ─────────────────────────────────────────────
def to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def from_bytes(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b))


# ─────────────────────────────────────────────
#  Descubrimiento de videos
# ─────────────────────────────────────────────
def find_videos(root: Path) -> list[dict]:
    """
    Retorna lista de dicts con keys: glosa, intento, video_id, video_path (Path).
    Estructura esperada: root/{glosa}/{intento}/*.{mov,mp4,...}
    """
    videos = []
    for glosa_dir in sorted(root.iterdir()):
        if not glosa_dir.is_dir():
            continue
        glosa = glosa_dir.name
        for intento_dir in sorted(glosa_dir.iterdir()):
            if not intento_dir.is_dir():
                continue
            intento = intento_dir.name
            # Buscar el primer archivo de video dentro del directorio intento
            found = None
            for f in sorted(intento_dir.iterdir()):
                if f.suffix.lower() in VIDEO_EXTENSIONS:
                    found = f
                    break
            if found is None:
                # Buscar recursivamente un nivel más (por si hay subcarpeta)
                for f in sorted(intento_dir.rglob("*")):
                    if f.suffix.lower() in VIDEO_EXTENSIONS:
                        found = f
                        break
            if found is not None:
                videos.append({
                    "glosa":      glosa,
                    "intento":    intento,
                    "video_id":   f"{glosa}_{intento}",
                    "video_path": found,
                })
    return videos


# ─────────────────────────────────────────────
#  Procesamiento de un video
# ─────────────────────────────────────────────
def process_video(meta: dict, detector) -> dict | None:
    """
    Procesa un video y devuelve la fila lista para Parquet.
    Retorna None si hay error irrecuperable.
    """
    path = str(meta["video_path"])
    cap  = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pre-alocación si conocemos el número de frames
    if total_frames > 0:
        kpt_tensor = np.zeros((total_frames, N_KPT, N_FEAT), dtype=np.float32)
        det_array  = np.zeros(total_frames, dtype=bool)
        dynamic    = False
    else:
        kpt_list   = []
        det_list   = []
        dynamic    = True

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            keypoints_list, scores_list = detector(frame)
        except Exception:
            # Frame problemático: guardamos ceros y seguimos
            keypoints_list = None

        if keypoints_list is not None and len(keypoints_list) > 0:
            kpts     = keypoints_list[0]
            scores   = scores_list[0]
            detected = True
        else:
            kpts     = None
            scores   = None
            detected = False

        if dynamic:
            row = np.zeros((N_KPT, N_FEAT), dtype=np.float32)
            if detected:
                row[:, :2] = kpts.astype(np.float32)
                row[:,  2] = scores.astype(np.float32)
            kpt_list.append(row)
            det_list.append(detected)
        else:
            if detected:
                kpt_tensor[frame_idx, :, :2] = kpts.astype(np.float32)
                kpt_tensor[frame_idx, :,  2] = scores.astype(np.float32)
            det_array[frame_idx] = detected

        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        return None  # Video vacío o ilegible

    if dynamic:
        kpt_tensor = np.stack(kpt_list, axis=0)
        det_array  = np.array(det_list, dtype=bool)
    else:
        kpt_tensor = kpt_tensor[:frame_idx]
        det_array  = det_array[:frame_idx]

    return {
        "glosa":           meta["glosa"],
        "intento":         meta["intento"],
        "video_id":        meta["video_id"],
        "video_path":      str(meta["video_path"]),
        "fps":             fps_video,
        "total_frames":    frame_idx,
        "width":           vid_w,
        "height":          vid_h,
        "keypoints":       to_bytes(kpt_tensor),
        "person_detected": to_bytes(det_array),
    }


# ─────────────────────────────────────────────
#  Flush parcial a disco
# ─────────────────────────────────────────────
def flush_rows(rows: list[dict], output_path: Path, first_write: bool) -> bool:
    """Escribe las filas acumuladas al Parquet. Retorna nuevo valor de first_write."""
    if not rows:
        return first_write

    table = pa.Table.from_pylist(rows, schema=SCHEMA)

    if first_write or not output_path.exists():
        pq.write_table(table, output_path, compression="zstd", compression_level=3)
        return False
    else:
        # Append: leer + concatenar + reescribir
        # (para datasets grandes considera usar ParquetWriter incremental)
        existing = pq.read_table(output_path)
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, output_path, compression="zstd", compression_level=3)
        return False


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Procesa todo MSLwords1 y genera un Parquet de landmarks."
    )
    p.add_argument("--root",       "-r", required=True,
                   help="Directorio raíz MSLwords1")
    p.add_argument("--output",     "-o", required=True,
                   help="Parquet de salida (ej. lsm_dataset.parquet)")
    p.add_argument("--device",     "-d", default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--mode",       default="balanced",
                   choices=["performance", "balanced", "lightweight"])
    p.add_argument("--resume",     action="store_true",
                   help="Salta videos cuyo video_id ya existe en el parquet")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Filas a acumular antes de flush a disco (default: 50)")
    args = p.parse_args()

    root       = Path(args.root)
    output     = Path(args.output)
    batch_size = args.batch_size

    if not root.exists():
        print(f"ERROR: '{root}' no existe.")
        sys.exit(1)

    # ── Descubrir videos ─────────────────────────────────────
    print(f"Escaneando '{root}'...")
    all_videos = find_videos(root)
    print(f"  → {len(all_videos)} videos encontrados.")

    # ── Resume: filtrar ya procesados ────────────────────────
    done_ids = set()
    if args.resume and output.exists():
        existing = pq.read_table(output, columns=["video_id"])
        done_ids = set(existing["video_id"].to_pylist())
        print(f"  → {len(done_ids)} ya procesados, se omitirán.")

    pending = [v for v in all_videos if v["video_id"] not in done_ids]
    print(f"  → {len(pending)} por procesar.")

    if not pending:
        print("Nada que procesar.")
        sys.exit(0)

    # ── Cargar modelo ────────────────────────────────────────
    print(f"\nCargando modelo RTMPose ({args.mode}) en {args.device}...")
    from rtmlib import Wholebody
    detector = Wholebody(mode=args.mode, backend="onnxruntime", device=args.device)
    print("Modelo listo.\n")

    # ── Procesar ─────────────────────────────────────────────
    errors    = []
    batch     = []
    first_w   = not output.exists()   # ¿es la primera escritura?

    progress = tqdm(pending, unit="video", dynamic_ncols=True)
    for meta in progress:
        progress.set_description(f"{meta['glosa']}/{meta['intento']}")

        try:
            row = process_video(meta, detector)
        except Exception as e:
            errors.append((meta["video_id"], str(e)))
            tqdm.write(f"  ✗ ERROR {meta['video_id']}: {e}")
            continue

        if row is None:
            errors.append((meta["video_id"], "video vacío o ilegible"))
            tqdm.write(f"  ✗ SKIP  {meta['video_id']}: video vacío o ilegible")
            continue

        batch.append(row)

        # Flush periódico para no perder progreso
        if len(batch) >= batch_size:
            first_w = flush_rows(batch, output, first_w)
            batch.clear()
            tqdm.write(f"  💾 flush → {output}  ({output.stat().st_size/1024**2:.1f} MB)")

    # Flush final con lo que queda
    if batch:
        first_w = flush_rows(batch, output, first_w)

    # ── Reporte ──────────────────────────────────────────────
    total_done = len(pending) - len(errors)
    print(f"\n{'─'*60}")
    print(f"✅  Videos procesados : {total_done}")
    print(f"❌  Errores           : {len(errors)}")
    if output.exists():
        mb = output.stat().st_size / 1024**2
        print(f"📦  Parquet final     : {output}  ({mb:.1f} MB)")

    if errors:
        err_log = output.with_suffix(".errors.txt")
        with open(err_log, "w") as f:
            for vid_id, msg in errors:
                f.write(f"{vid_id}\t{msg}\n")
        print(f"⚠️   Log de errores    : {err_log}")


if __name__ == "__main__":
    main()