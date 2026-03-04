"""
video_to_landmarks.py
─────────────────────
Procesa un video frame a frame usando la misma lógica que camera.py
y exporta series de tiempo de landmarks a CSV.

Uso:
    python3 video_to_landmarks.py -i video.mp4 -o landmarks.csv
    python3 video_to_landmarks.py -i video.mp4 -o landmarks.csv --no-preview
    python3 video_to_landmarks.py -i video.mp4 -o landmarks.csv --mirror
"""

import cv2
import mediapipe as mp
import csv
import argparse
import sys
import numpy as np

mp_holistic  = mp.solutions.holistic
mp_drawing   = mp.solutions.drawing_utils

FACE_POINTS = [
    33, 133,   # Left eye corners
    362, 263,  # Right eye corners
    61, 291,   # Lip corners
    105, 334,  # Eyebrow midpoints
]

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def lm_to_str(x, y, z) -> str:
    """Serializa un landmark como '[x y z]' (formato np.array legible)."""
    return f"[{round(x,6)} {round(y,6)} {round(z,6)}]"

NONE_LM = "[None None None]"


# ─────────────────────────────────────────────
#  CSV header
# ─────────────────────────────────────────────
def build_csv_header():
    cols = ["frame", "time_s"]

    # Face selected points — 1 col por punto
    for idx in FACE_POINTS:
        cols.append(f"face_lm{idx}")

    # Left hand — 21 landmarks, 1 col cada uno
    for i in range(21):
        cols.append(f"left_hand_lm{i:02d}")

    # Right hand — 21 landmarks, 1 col cada uno
    for i in range(21):
        cols.append(f"right_hand_lm{i:02d}")

    return cols


# ─────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────
def process_video(input_path, output_csv, show_preview, mirror):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: no se puede abrir '{input_path}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video : {input_path}")
    print(f"        {vid_w}×{vid_h}  {fps_video:.2f} fps  {total_frames} frames")
    print(f"CSV   : {output_csv}")
    if show_preview:
        print("Preview activo — ESC / Q para cancelar.")

    csv_file = open(output_csv, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(build_csv_header())

    frame_idx = 0
    cancelled = False

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # ── Fila CSV ─────────────────────────────────────────────
            time_s = frame_idx / fps_video
            row    = [frame_idx, round(time_s, 5)]

            # Face — 1 celda por punto con "[x y z]"
            if results.face_landmarks:
                for idx in FACE_POINTS:
                    lm = results.face_landmarks.landmark[idx]
                    row.append(lm_to_str(lm.x, lm.y, lm.z))
            else:
                row += [NONE_LM] * len(FACE_POINTS)

            # Left hand
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    row.append(lm_to_str(lm.x, lm.y, lm.z))
            else:
                row += [NONE_LM] * 21

            # Right hand
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    row.append(lm_to_str(lm.x, lm.y, lm.z))
            else:
                row += [NONE_LM] * 21

            writer.writerow(row)

            # ── Preview ───────────────────────────────────────────────
            if show_preview:
                if results.face_landmarks:
                    for idx in FACE_POINTS:
                        landmark = results.face_landmarks.landmark[idx]
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                cv2.imshow("Procesando video", image)
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
    p = argparse.ArgumentParser(description="Extrae landmarks de manos y cara a CSV.")
    p.add_argument("--input",      "-i", required=True,       help="Video de entrada")
    p.add_argument("--output",     "-o", required=True,       help="CSV de salida")
    p.add_argument("--no-preview", action="store_true",       help="Sin ventana de preview")
    p.add_argument("--mirror",     "-m", action="store_true", help="Espejo horizontal")
    args = p.parse_args()

    process_video(
        input_path   = args.input,
        output_csv   = args.output,
        show_preview = not args.no_preview,
        mirror       = args.mirror,
    )
