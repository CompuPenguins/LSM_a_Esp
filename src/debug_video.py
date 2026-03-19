"""
debug_video.py
==============
Diagnóstico completo del problema de carga de video en infer_viz.

Ejecutar en Jupyter:
    %run debug_video.py

O como script:
    python debug_video.py
"""

import io
import sys
from pathlib import Path

# ── Ajusta estos dos valores ──────────────────────────────────────────────────
PARQUET    = "../corpus_LSM_esp/lsm_dataset.parquet"
VIDEO_ROOT = "../corpus_LSM_esp/Mexican sign language dataset/MSLwords1"
VIDEO_ID   = None   # None = tomar el primero del parquet; o pon ej. "106_10106"
# ─────────────────────────────────────────────────────────────────────────────

SEP = "─" * 60

# ══════════════════════════════════════════════════════════════════════════════
# 1. OpenCV
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("1. OpenCV")
print(SEP)
try:
    import cv2
    print(f"  ✓ cv2 version : {cv2.__version__}")
    build = cv2.getBuildInformation()
    # Mostrar soporte de codecs relevantes
    for line in build.splitlines():
        line = line.strip()
        if any(k in line for k in ("FFMPEG", "GStreamer", "Video I/O", "avcodec", "avformat")):
            print(f"  │  {line}")
    HAS_CV2 = True
except ImportError as e:
    print(f"  ✗ cv2 no instalado: {e}")
    print("    → pip install opencv-python  o  pip install opencv-python-headless")
    HAS_CV2 = False

# ══════════════════════════════════════════════════════════════════════════════
# 2. Parquet — leer video_id de muestra
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("2. Parquet")
print(SEP)
try:
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(
        PARQUET,
        columns=["video_id", "glosa", "width", "height", "keypoints"],
    ).reset_index(drop=True)
    print(f"  ✓ Filas      : {len(df)}")
    print(f"  ✓ Columnas   : {list(df.columns)}")

    if VIDEO_ID is not None:
        row = df[df["video_id"] == VIDEO_ID]
        if row.empty:
            print(f"  ✗ video_id '{VIDEO_ID}' no encontrado en el parquet")
            print(f"    Ejemplos disponibles: {df['video_id'].head(5).tolist()}")
            sys.exit(1)
        row = row.iloc[0]
    else:
        row = df.iloc[0]

    vid    = str(row["video_id"])
    W, H   = int(row["width"]), int(row["height"])
    glosa  = str(row["glosa"])
    kpts   = np.load(io.BytesIO(row["keypoints"]))

    print(f"  ✓ video_id   : {vid}")
    print(f"  ✓ glosa      : {glosa}")
    print(f"  ✓ resolución : {W}x{H}")
    print(f"  ✓ keypoints  : shape={kpts.shape}  dtype={kpts.dtype}")

except Exception as e:
    print(f"  ✗ Error leyendo parquet: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Localizar el archivo de video
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("3. Localizar video en disco")
print(SEP)

def _parse_video_id(video_id):
    if "_" in video_id:
        glosa_dir, speaker_dir = video_id.split("_", 1)
    else:
        v = video_id.zfill(5)
        glosa_dir, speaker_dir = v[2:], v
    return glosa_dir, speaker_dir

root = Path(VIDEO_ROOT)
print(f"  VIDEO_ROOT   : {root.resolve()}")
print(f"  Existe       : {root.exists()}")

glosa_dir, speaker_dir = _parse_video_id(vid)
print(f"  glosa_dir    : '{glosa_dir}'")
print(f"  speaker_dir  : '{speaker_dir}'")

canonical = root / glosa_dir / speaker_dir
print(f"  Ruta canónica: {canonical}")
print(f"  Existe       : {canonical.exists()}")

video_path = None
if canonical.exists():
    found = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.MP4", "*.MOV"):
        found.extend(canonical.glob(ext))
    if found:
        video_path = str(sorted(found)[0])
        print(f"  ✓ Archivo    : {video_path}")
        print(f"  Tamaño       : {Path(video_path).stat().st_size / 1024:.1f} KB")
    else:
        print(f"  ✗ Carpeta existe pero sin archivos de video")
        print(f"    Contenido: {list(canonical.iterdir())}")
else:
    print(f"  ✗ Carpeta no existe. Buscando en todo el árbol…")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi"):
            if speaker_dir in str(p):
                video_path = str(p)
                print(f"  ✓ Encontrado (fallback): {video_path}")
                break
    if not video_path:
        print(f"  ✗ No se encontró ningún archivo para video_id '{vid}'")
        # Mostrar qué hay en el root
        subdirs = sorted([d.name for d in root.iterdir() if d.is_dir()])[:10]
        print(f"    Subdirectorios en root (primeros 10): {subdirs}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Abrir y leer frames con OpenCV
# ══════════════════════════════════════════════════════════════════════════════
if video_path and HAS_CV2:
    print()
    print(SEP)
    print("4. Leer frames con OpenCV")
    print(SEP)

    for backend_name, backend_id in [("CAP_ANY", cv2.CAP_ANY), ("CAP_FFMPEG", cv2.CAP_FFMPEG)]:
        print(f"\n  Backend: {backend_name}")
        try:
            cap = cv2.VideoCapture(video_path, backend_id)
        except Exception:
            cap = cv2.VideoCapture(video_path)

        opened = cap.isOpened()
        print(f"  cap.isOpened() : {opened}")

        if opened:
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec  = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])
            print(f"  Frames totales : {total}")
            print(f"  FPS            : {fps:.1f}")
            print(f"  Resolución     : {fw}x{fh}")
            print(f"  Codec (fourcc) : '{codec}'")

            # Intentar leer 3 frames
            frames_ok = 0
            for i in range(3):
                ret, frame = cap.read()
                if ret:
                    frames_ok += 1
                    print(f"  frame[{i}]        : shape={frame.shape}  dtype={frame.dtype}  ok=True")
                else:
                    print(f"  frame[{i}]        : ret=False — no se pudo leer")
            cap.release()

            if frames_ok == 0:
                print(f"\n  ✗ El video se abre pero no se pueden leer frames.")
                print(f"    Causa probable: codec no soportado en esta instalación de OpenCV.")
                print(f"    Soluciones:")
                print(f"      1. pip install opencv-python-headless  (incluye FFMPEG)")
                print(f"      2. conda install -c conda-forge opencv")
                print(f"      3. Convertir el video: ffmpeg -i input.mov -c:v libx264 output.mp4")
            else:
                print(f"\n  ✓ {frames_ok}/3 frames leídos correctamente con {backend_name}")
                break   # este backend funciona, no seguir probando
        else:
            cap.release()

# ══════════════════════════════════════════════════════════════════════════════
# 5. Verificar matplotlib imshow
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("5. Test matplotlib imshow (fondo de video en animación)")
print(SEP)
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    print(f"  ✓ matplotlib  : {matplotlib.__version__}")
    print(f"  Backend activo: {matplotlib.get_backend()}")

    # Crear un frame sintético y mostrarlo brevemente
    dummy = np.zeros((100, 80, 3), dtype=np.uint8)
    dummy[:50, :, 2] = 200   # mitad azul
    dummy[50:, :, 1] = 200   # mitad verde

    fig, ax = plt.subplots(figsize=(2, 2))
    im = ax.imshow(dummy, extent=[0, 1, 1, 0], aspect="auto")
    ax.axis("off")

    # Simular un update
    dummy2 = dummy.copy()
    dummy2[:, :, 0] = 100
    im.set_data(dummy2)
    fig.canvas.draw()
    plt.close(fig)
    print(f"  ✓ imshow + set_data funcionan correctamente")
except Exception as e:
    print(f"  ✗ Error en matplotlib: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Resumen
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("RESUMEN")
print(SEP)
print(f"  OpenCV instalado   : {'✓' if HAS_CV2 else '✗'}")
print(f"  Video en disco     : {'✓' if video_path else '✗'}")
print(f"  video_id probado   : {vid}")
if video_path:
    print(f"  Ruta del video     : {video_path}")
print()