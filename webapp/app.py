"""
app.py — LSM Recognizer Web Server
===================================
Recibe frames codificados en base64 desde el frontend,
extrae landmarks COCO-WholeBody (133 puntos) con RTMPose
y corre inferencia con el modelo ONNX exportado.

Endpoints:
  GET  /            → index.html
    POST /infer       → {"frames": ["...base64...", ...], "mirror": bool}
                    ← {"glosa": str, "conf": float, "end_prob": float,
                        "top3": [["glosa", conf], ...]}
  GET  /labels      → lista de glosas ordenadas por índice
  GET  /health      → estado del servidor
"""

import json
import os
import io
import base64

import torch
import numpy as np
import cv2
from flask import Flask, jsonify, render_template, request
from rtmlib import Wholebody

app = Flask(__name__)

# ─── Rutas configurables ──────────────────────────────────────────────────────
MODEL_PATH  = os.environ.get("MODEL_PATH",  "model.onnx")
LABELS_PATH = os.environ.get("LABELS_PATH", "glosa_labels.json")
CKPT_PATH   = os.environ.get("CKPT_PATH",   "best_model.ptrom")

# ─── Estado global ────────────────────────────────────────────────────────────
ort_session     = None
input_name      = None
labels: list    = []
rtm_detector    = None  # RTMPose detector

# ─── Constantes COCO-WholeBody ────────────────────────────────────────────
N_KPT  = 133  # 133 keypoints COCO-WholeBody (entrada RTMPose)
N_FEAT = 3    # x, y, score

# Índices a CONSERVAR: cuerpo (0-16) + manos (91-132)
# Se eliminan: pies (17-22) y cara (23-90)
_KEEP_IDX   = list(range(0, 17)) + list(range(91, 133))  # 59 landmarks
N_KPT_MODEL = len(_KEEP_IDX)   # 59
INPUT_DIM   = N_KPT_MODEL * 2  # 118


# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares para landmarks
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_bytes(arr: np.ndarray) -> bytes:
    """Serializa un ndarray a bytes con np.save (portable)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def bytes_to_tensor(b: bytes) -> np.ndarray:
    """Deserializa bytes a ndarray."""
    return np.load(io.BytesIO(b))


def decode_frame_b64(b64_str: str):
    """Decodifica un frame en base64 a numpy array."""
    frame_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


def preprocess_landmarks(kpt_seq: np.ndarray, det_seq: np.ndarray):
    """
    Preprocesa landmarks COCO-WholeBody igual que en entrenamiento:
    - Filtra a 59 landmarks (cuerpo 0-16 + manos 91-132), sin cara ni pies
    - Máscara baja confianza (score < 0.3) → [0, 0]
    - Centrado en punto medio de hombros (índices 5, 6 tras filtrado)
    - Escalado por distancia hombro-cadera media → punto medio caderas

    Args:
        kpt_seq: (T, 133, 3) float32 [x, y, score]
        det_seq: (T,) bool

    Returns:
        (1, TMAX, 118) float32 listo para el modelo ONNX
    """
    TMAX = 200
    T = kpt_seq.shape[0]

    # ── 1. Filtrar solo landmarks útiles (59 de 133) ──────────────────────
    kpts = kpt_seq[:, _KEEP_IDX, :]        # (T, 59, 3)

    # ── 2. Máscara baja confianza ─────────────────────────────────────────
    xy     = kpts[:, :, :2].copy()         # (T, 59, 2)
    scores = kpts[:, :, 2]
    xy[scores < 0.3] = 0.0

    # ── 3. Centrado en punto medio de hombros ─────────────────────────────
    # Tras filtrar, hombros siguen en índices 5 y 6 (dentro de rango 0-16)
    shoulder_mid = (xy[:, 5, :] + xy[:, 6, :]) / 2.0   # (T, 2)
    xy -= shoulder_mid[:, None, :]

    # ── 4. Escalado por distancia hombro-cadera (punto medio caderas) ─────
    # Caderas en índices 11 y 12 (dentro de rango 0-16)
    hip_mid = (xy[:, 11, :] + xy[:, 12, :]) / 2.0      # (T, 2)
    scale   = np.linalg.norm(hip_mid, axis=-1)           # (T,)
    scale   = np.clip(scale, 1e-8, None)
    xy     /= scale[:, None, None]

    # ── 5. Aplanar a (T, 118) y padear a TMAX ────────────────────────────
    landmarks_flat = np.zeros((TMAX, INPUT_DIM), dtype=np.float32)
    for t in range(min(T, TMAX)):
        landmarks_flat[t] = xy[t].reshape(-1)

    return landmarks_flat


# ─────────────────────────────────────────────────────────────────────────────
# Carga de recursos al arrancar
# ─────────────────────────────────────────────────────────────────────────────

def load_resources():
    global ort_session, input_name, labels, rtm_detector

    # ── RTMPose Detector (COCO-WholeBody) ──────────────────────────────────
    try:
        print("Cargando RTMPose...")
        rtm_detector = Wholebody(
            mode="balanced",           # performance | balanced | lightweight
            backend="onnxruntime",
            device="cpu",
        )
        print("✅ RTMPose cargado (COCO-WholeBody 133 keypoints)")
    except Exception as exc:
        print(f"❌ Error cargando RTMPose: {exc}")
        print("   Asegúrate de tener instalado: pip install rtmlib onnxruntime opencv-python")

    # ── Modelo ONNX ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Modelo ONNX no encontrado en '{MODEL_PATH}'")
        print("   Ejecuta primero: python onnx_export.py")
    else:
        try:
            import onnxruntime as ort
            ort_session = ort.InferenceSession(
                MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )
            input_name = ort_session.get_inputs()[0].name
            print(f"✅ Modelo ONNX cargado  ({MODEL_PATH})")
            print(f"   Entrada : {input_name} {ort_session.get_inputs()[0].shape}")
            for o in ort_session.get_outputs():
                print(f"   Salida  : {o.name} {o.shape}")
        except Exception as exc:
            print(f"❌ Error cargando modelo ONNX: {exc}")

    # ── Labels (glosa_labels.json o desde el checkpoint) ──────────────────────
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        # Formato: {"0": "HOLA", "1": "GRACIAS", ...}
        max_idx = max(int(k) for k in raw)
        labels  = [raw.get(str(i), f"clase_{i}") for i in range(max_idx + 1)]
        print(f"✅ Labels cargadas desde '{LABELS_PATH}'  ({len(labels)} glosas)")

    elif os.path.exists(CKPT_PATH):
        print(f"   Intentando extraer labels desde checkpoint '{CKPT_PATH}'...")
        try:
            ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "glosa2idx" in ckpt:
                g2i    = ckpt["glosa2idx"]
                labels = [g for g, _ in sorted(g2i.items(), key=lambda x: x[1])]
                # Guardar para no volver a leer el checkpoint
                idx2g = {str(i): g for i, g in enumerate(labels)}
                with open(LABELS_PATH, "w", encoding="utf-8") as f:
                    json.dump(idx2g, f, ensure_ascii=False, indent=2)
                print(f"✅ Labels extraídas del checkpoint ({len(labels)} glosas)")
                print(f"   Guardadas en '{LABELS_PATH}' para futuros arranques.")
            else:
                print("   Checkpoint sin 'glosa2idx' — se usarán índices numéricos.")
        except Exception as exc:
            print(f"   No se pudieron extraer labels: {exc}")

    if not labels and ort_session is not None:
        # Fallback: índices numéricos
        n = ort_session.get_outputs()[0].shape[-1] or 249
        labels = [f"clase_{i}" for i in range(n)]
        print(f"   Usando {len(labels)} labels ficticias (índices).")


# ─────────────────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": ort_session is not None,
        "num_labels":   len(labels),
    })


@app.route("/labels")
def get_labels():
    return jsonify(labels)


@app.route("/infer", methods=["POST"])
def infer():
    """
    Recibe frames de video y realiza:
    1. Extracción de 133 landmarks COCO-WholeBody con RTMPose
    2. Preprocesamiento (centrado, escalado)
    3. Inferencia con modelo ONNX
    
    Body JSON:
      {
        "frames": ["base64_encoded_frame1", "base64_encoded_frame2", ...],
        "mirror": bool  (opcional)
      }
    
    Response:
      {
        "glosa":    str,
        "conf":     float,
        "end_prob": float,
        "top3":     [["glosa", conf], ...],
        "num_frames": int
      }
    """
    if ort_session is None:
        return jsonify({"error": "Modelo ONNX no cargado"}), 503

    if rtm_detector is None:
        return jsonify({"error": "RTMPose no cargado"}), 503

    data = request.get_json(force=True)
    if not data or "frames" not in data:
        return jsonify({"error": "Falta el campo 'frames'"}), 400

    try:
        frames_b64 = data["frames"]  # list[str]
        mirror = data.get("mirror", False)
        
        if not isinstance(frames_b64, list) or len(frames_b64) == 0:
            return jsonify({"error": "frames debe ser una lista no vacía"}), 400

        # ── Decodificar y extraer landmarks ────────────────────────────────
        kpt_list = []
        det_list = []

        for b64_frame in frames_b64:
            try:
                frame = decode_frame_b64(b64_frame)
                if frame is None:
                    continue

                if mirror:
                    frame = cv2.flip(frame, 1)

                # Inferencia con RTMPose
                keypoints, scores = rtm_detector(frame)

                if keypoints is not None and len(keypoints) > 0:
                    kpts = keypoints[0]      # (133, 2)
                    scr = scores[0]          # (133,)
                    
                    # Construir tensor (133, 3)
                    kpt_with_score = np.stack(
                        [kpts[:, 0], kpts[:, 1], scr],
                        axis=1
                    ).astype(np.float32)
                    kpt_list.append(kpt_with_score)
                    det_list.append(True)
                else:
                    # Sin detección: landmark en ceros
                    kpt_list.append(np.zeros((N_KPT, N_FEAT), dtype=np.float32))
                    det_list.append(False)

            except Exception as e:
                print(f"  ⚠ Error decodificando frame: {e}")
                kpt_list.append(np.zeros((N_KPT, N_FEAT), dtype=np.float32))
                det_list.append(False)

        if not kpt_list:
            return jsonify({"error": "No se pudieron procesar los frames"}), 400

        # ── Stack a tensores (T, 133, 3) y (T,) ───────────────────────────
        kpt_seq = np.stack(kpt_list, axis=0)    # (T, 133, 3)
        det_seq = np.array(det_list, dtype=bool)  # (T,)

        # ── Preprocesamiento (centrado, escalado, normalización) ────────────
        x_preprocessed = preprocess_landmarks(kpt_seq, det_seq)  # (TMAX, 118)
        x = x_preprocessed[np.newaxis, ...]  # (1, TMAX, 118)

        # ── Inferencia ─────────────────────────────────────────────────────
        ort_inputs = {input_name: x}
        outputs = ort_session.run(None, ort_inputs)
        glosa_logits = outputs[0][0]    # (num_classes,)
        trigger_logits = outputs[1][0]  # (1,)

        # Softmax sobre glosa
        ex = np.exp(glosa_logits - glosa_logits.max())
        prob = ex / ex.sum()

        end_prob = float(1 / (1 + np.exp(-trigger_logits[0])))  # sigmoid

        top_idx = int(prob.argmax())
        top_conf = float(prob[top_idx])

        top3_idx = prob.argsort()[::-1][:3]
        top3 = [
            [labels[i] if i < len(labels) else f"clase_{i}", float(prob[i])]
            for i in top3_idx
        ]

        return jsonify({
            "glosa": labels[top_idx] if top_idx < len(labels) else f"clase_{top_idx}",
            "conf": top_conf,
            "end_prob": end_prob,
            "top3": top3,
            "num_frames": len(kpt_list),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Arranque
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("🤟  LSM Recognizer — Servidor Web")
    print("=" * 60)
    load_resources()
    print("\n🚀 Servidor en https://localhost:8443\n" + "=" * 60)
    app.run(host="0.0.0.0", port=8443, debug=False,
        ssl_context=("cert.pem", "key.pem"))