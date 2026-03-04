"""
video_to_landmarks.py
─────────────────────
Procesa un video frame a frame usando la misma lógica que camera.py
(Holistic con pose, mano izquierda y mano derecha) y exporta
series de tiempo de landmarks a CSV.

Uso:
    python3 video_to_landmarks.py -i video.mp4 -o landmarks.csv
    python3 video_to_landmarks.py -i video.mp4 -o landmarks.csv --no-preview
    python3 video_to_landmarks.py -i video.mp4 -o landmarks.csv --no-mirror
"""

import cv2
import mediapipe as mp
import csv
import argparse
import sys

mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic       = mp.solutions.holistic

# Pose tiene 33 landmarks, manos 21 cada una
N_POSE  = 33
N_HAND  = 21


# ─────────────────────────────────────────────
#  CSV header
# ─────────────────────────────────────────────
def build_csv_header():
    cols = ["frame", "time_s"]

    # Pose — 33 landmarks (x, y, z, visibility)
    for i in range(N_POSE):
        for coord in ("x", "y", "z", "vis"):
            cols.append(f"pose_lm{i:02d}_{coord}")

    # Left hand — 21 landmarks
    for i in range(N_HAND):
        for coord in ("x", "y", "z"):
            cols.append(f"left_hand_lm{i:02d}_{coord}")

    # Right hand — 21 landmarks
    for i in range(N_HAND):
        for coord in ("x", "y", "z"):
            cols.append(f"right_hand_lm{i:02d}_{coord}")

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
        print("Preview activo — ESC para cancelar.")

    csv_file = open(output_csv, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(build_csv_header())

    frame_idx = 0
    cancelled = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
    ) as holistic:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            if mirror:
                image = cv2.flip(image, 1)

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # ── Fila CSV ─────────────────────────────────────────────
            time_s = frame_idx / fps_video
            row    = [frame_idx, round(time_s, 5)]

            # Pose
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    row += [round(lm.x, 6), round(lm.y, 6),
                            round(lm.z, 6), round(lm.visibility, 4)]
            else:
                row += [None] * (N_POSE * 4)

            # Left hand
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    row += [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)]
            else:
                row += [None] * (N_HAND * 3)

            # Right hand
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    row += [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)]
            else:
                row += [None] * (N_HAND * 3)

            writer.writerow(row)

            # ── Preview (misma lógica que camera.py) ─────────────────
            if show_preview:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                    )
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                    )

                cv2.imshow("MediaPipe Holistic", image)
                if cv2.waitKey(1) & 0xFF == 27:
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
    p = argparse.ArgumentParser(description="Extrae landmarks de pose y manos a CSV.")
    p.add_argument("--input",     "-i", required=True,        help="Video de entrada")
    p.add_argument("--output",    "-o", required=True,        help="CSV de salida")
    p.add_argument("--no-preview", action="store_true",       help="Sin ventana de preview")
    p.add_argument("--no-mirror",  action="store_true",       help="Sin espejo horizontal")
    args = p.parse_args()

    process_video(
        input_path   = args.input,
        output_csv   = args.output,
        show_preview = not args.no_preview,
        mirror       = not args.no_mirror,
    )
