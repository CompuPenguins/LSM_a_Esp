"""
app.py — LSM Recognizer + Translator Web Server
=================================================
Chat endpoint for ESP ↔ LSM translation, plus secondary sign-recognition
and server-side speech-to-text via faster-whisper.

Endpoints:
  GET  /              → index.html
  POST /translate     → {"text": str, "direction": "esp_to_lsm"|"lsm_to_esp"}
                      ← {"translation": str}
  POST /stt           → multipart/form-data  field: audio (webm/ogg/wav)
                      ← {"text": str}
  POST /infer         → {"frames": [...base64...], "mirror": bool}
                      ← {"glosa": str, "conf": float, "end_prob": float,
                          "top3": [...], "num_frames": int}
  GET  /labels        → lista de glosas ordenadas por índice
  GET  /health        → estado del servidor
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
MODEL_PATH      = os.environ.get("MODEL_PATH",      "model.onnx")
LABELS_PATH     = os.environ.get("LABELS_PATH",     "glosa_labels.json")
CKPT_PATH       = os.environ.get("CKPT_PATH",       "best_model.ptrom")
ESP_TO_LSM_DIR  = os.environ.get("ESP_TO_LSM_DIR",  "./translator/esp_to_lsm")
LSM_TO_ESP_DIR  = os.environ.get("LSM_TO_ESP_DIR",  "./translator/lsm_to_esp")

# ─── Estado global ────────────────────────────────────────────────────────────
ort_session     = None
input_name      = None
labels: list    = []
rtm_detector    = None

# Translation models – lazy-loaded once on first use
_translators: dict = {}   # {"esp_to_lsm": (tokenizer, model), "lsm_to_esp": (...)}

# Whisper STT model – lazy-loaded on first /stt call
_whisper_model = None
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")  # tiny|base|small|medium

# ─── Constantes COCO-WholeBody ─────────────────────────────────────────────
N_KPT  = 133
N_FEAT = 3

_KEEP_IDX   = list(range(0, 17)) + list(range(91, 133))  # 59 landmarks
N_KPT_MODEL = len(_KEEP_IDX)
INPUT_DIM   = N_KPT_MODEL * 2   # 118


# ─────────────────────────────────────────────────────────────────────────────
# Translation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_translator(direction: str):
    """Load and cache a Seq2Seq translation model by direction key."""
    if direction in _translators:
        return _translators[direction]

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_dir = ESP_TO_LSM_DIR if direction == "esp_to_lsm" else LSM_TO_ESP_DIR
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Directorio de modelo no encontrado: '{model_dir}'")

    print(f"  Cargando traductor '{direction}' desde '{model_dir}'…")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    _translators[direction] = (tokenizer, model)
    print(f"  ✅ Traductor '{direction}' listo.")
    return tokenizer, model


def translate_text(text: str, direction: str) -> str:
    """Translate *text* using the Seq2Seq model for *direction*."""
    tokenizer, model = _load_translator(direction)
    inputs  = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# STT helpers (faster-whisper, server-side — browser-independent)
# ─────────────────────────────────────────────────────────────────────────────

def _load_whisper():
    """Lazy-load and cache the faster-whisper model."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    from faster_whisper import WhisperModel
    print(f"  Cargando Whisper '{WHISPER_MODEL_SIZE}'…")
    _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    print(f"  ✅ Whisper '{WHISPER_MODEL_SIZE}' listo.")
    return _whisper_model


def transcribe_audio(audio_bytes: bytes, mime: str) -> str:
    """Transcribe raw audio bytes to Spanish text using faster-whisper."""
    import tempfile, pathlib, subprocess
    ext = ".webm"
    if "ogg" in mime:  ext = ".ogg"
    elif "wav" in mime: ext = ".wav"
    elif "mp4" in mime or "mp4a" in mime: ext = ".mp4"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = _load_whisper()
        segments, _ = model.transcribe(
            tmp_path,
            language="es",          # always Spanish — LSM STT doesn't exist
            beam_size=5,
            vad_filter=True,        # skip silent chunks automatically
        )
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)



def decode_frame_b64(b64_str: str):
    frame_data = base64.b64decode(b64_str)
    nparr      = np.frombuffer(frame_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def preprocess_landmarks(kpt_seq: np.ndarray, det_seq: np.ndarray):
    TMAX = 200
    T    = kpt_seq.shape[0]

    kpts   = kpt_seq[:, _KEEP_IDX, :]
    xy     = kpts[:, :, :2].copy()
    scores = kpts[:, :, 2]
    xy[scores < 0.3] = 0.0

    shoulder_mid = (xy[:, 5, :] + xy[:, 6, :]) / 2.0
    xy -= shoulder_mid[:, None, :]

    hip_mid = (xy[:, 11, :] + xy[:, 12, :]) / 2.0
    scale   = np.linalg.norm(hip_mid, axis=-1)
    scale   = np.clip(scale, 1e-8, None)
    xy     /= scale[:, None, None]

    landmarks_flat = np.zeros((TMAX, INPUT_DIM), dtype=np.float32)
    for t in range(min(T, TMAX)):
        landmarks_flat[t] = xy[t].reshape(-1)
    return landmarks_flat


# ─────────────────────────────────────────────────────────────────────────────
# Startup resource loading
# ─────────────────────────────────────────────────────────────────────────────

def load_resources():
    global ort_session, input_name, labels, rtm_detector

    # ── RTMPose ──────────────────────────────────────────────────────────────
    try:
        print("Cargando RTMPose…")
        rtm_detector = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
        print("✅ RTMPose cargado (133 keypoints)")
    except Exception as exc:
        print(f"❌ Error cargando RTMPose: {exc}")

    # ── ONNX sign model ──────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Modelo ONNX no encontrado en '{MODEL_PATH}'")
    else:
        try:
            import onnxruntime as ort
            ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
            input_name  = ort_session.get_inputs()[0].name
            print(f"✅ Modelo ONNX cargado ({MODEL_PATH})")
        except Exception as exc:
            print(f"❌ Error cargando modelo ONNX: {exc}")

    # ── Labels ───────────────────────────────────────────────────────────────
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        max_idx = max(int(k) for k in raw)
        labels  = [raw.get(str(i), f"clase_{i}") for i in range(max_idx + 1)]
        print(f"✅ {len(labels)} labels cargadas")
    elif os.path.exists(CKPT_PATH):
        try:
            ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "glosa2idx" in ckpt:
                g2i    = ckpt["glosa2idx"]
                labels = [g for g, _ in sorted(g2i.items(), key=lambda x: x[1])]
                with open(LABELS_PATH, "w", encoding="utf-8") as f:
                    json.dump({str(i): g for i, g in enumerate(labels)}, f,
                              ensure_ascii=False, indent=2)
                print(f"✅ {len(labels)} labels extraídas del checkpoint")
        except Exception as exc:
            print(f"   No se pudieron extraer labels: {exc}")

    if not labels and ort_session is not None:
        n = ort_session.get_outputs()[0].shape[-1] or 249
        labels = [f"clase_{i}" for i in range(n)]

    # ── Pre-load translation models (optional – comment out to lazy-load) ───
    for direction, model_dir in [("esp_to_lsm", ESP_TO_LSM_DIR), ("lsm_to_esp", LSM_TO_ESP_DIR)]:
        if os.path.isdir(model_dir):
            try:
                _load_translator(direction)
            except Exception as exc:
                print(f"⚠️  Traductor '{direction}' no cargado: {exc}")
        else:
            print(f"⚠️  Directorio de traducción no encontrado: '{model_dir}'")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status":           "ok",
        "model_loaded":     ort_session is not None,
        "num_labels":       len(labels),
        "translators":      list(_translators.keys()),
    })


@app.route("/labels")
def get_labels():
    return jsonify(labels)


@app.route("/translate", methods=["POST"])
def translate():
    """
    Body JSON:
      { "text": str, "direction": "esp_to_lsm" | "lsm_to_esp" }
    Response:
      { "translation": str }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "JSON vacío"}), 400

    text      = data.get("text", "").strip()
    direction = data.get("direction", "")

    if not text:
        return jsonify({"error": "El campo 'text' está vacío"}), 400
    if direction not in ("esp_to_lsm", "lsm_to_esp"):
        return jsonify({"error": "direction debe ser 'esp_to_lsm' o 'lsm_to_esp'"}), 400

    try:
        result = translate_text(text, direction)
        return jsonify({"translation": result})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/stt", methods=["POST"])
def stt():
    """
    Receives a raw audio file (webm/ogg/wav) and returns the Spanish transcript.
    Expects multipart/form-data with field name 'audio'.
    Response: { "text": str }
    """
    if "audio" not in request.files:
        return jsonify({"error": "Falta el campo 'audio'"}), 400

    audio_file = request.files["audio"]
    mime       = audio_file.content_type or "audio/webm"
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return jsonify({"error": "Archivo de audio vacío"}), 400

    try:
        text = transcribe_audio(audio_bytes, mime)
        return jsonify({"text": text})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/infer", methods=["POST"])
def infer():
    if ort_session is None:
        return jsonify({"error": "Modelo ONNX no cargado"}), 503
    if rtm_detector is None:
        return jsonify({"error": "RTMPose no cargado"}), 503

    data = request.get_json(force=True)
    if not data or "frames" not in data:
        return jsonify({"error": "Falta el campo 'frames'"}), 400

    try:
        frames_b64 = data["frames"]
        mirror     = data.get("mirror", False)

        if not isinstance(frames_b64, list) or len(frames_b64) == 0:
            return jsonify({"error": "frames debe ser una lista no vacía"}), 400

        kpt_list, det_list = [], []

        for b64_frame in frames_b64:
            try:
                frame = decode_frame_b64(b64_frame)
                if frame is None:
                    raise ValueError("frame vacío")
                if mirror:
                    frame = cv2.flip(frame, 1)

                keypoints, scores = rtm_detector(frame)

                if keypoints is not None and len(keypoints) > 0:
                    kpts = keypoints[0]
                    scr  = scores[0]
                    kpt_with_score = np.stack(
                        [kpts[:, 0], kpts[:, 1], scr], axis=1
                    ).astype(np.float32)
                    kpt_list.append(kpt_with_score)
                    det_list.append(True)
                else:
                    kpt_list.append(np.zeros((N_KPT, N_FEAT), dtype=np.float32))
                    det_list.append(False)
            except Exception as e:
                print(f"  ⚠ Error decodificando frame: {e}")
                kpt_list.append(np.zeros((N_KPT, N_FEAT), dtype=np.float32))
                det_list.append(False)

        if not kpt_list:
            return jsonify({"error": "No se pudieron procesar los frames"}), 400

        kpt_seq = np.stack(kpt_list, axis=0)
        det_seq = np.array(det_list, dtype=bool)

        x_preprocessed = preprocess_landmarks(kpt_seq, det_seq)
        x               = x_preprocessed[np.newaxis, ...]

        ort_inputs  = {input_name: x}
        outputs     = ort_session.run(None, ort_inputs)
        glosa_logits   = outputs[0][0]
        trigger_logits = outputs[1][0]

        ex      = np.exp(glosa_logits - glosa_logits.max())
        prob    = ex / ex.sum()
        end_prob = float(1 / (1 + np.exp(-trigger_logits[0])))

        top_idx  = int(prob.argmax())
        top_conf = float(prob[top_idx])
        top3_idx = prob.argsort()[::-1][:3]
        top3 = [
            [labels[i] if i < len(labels) else f"clase_{i}", float(prob[i])]
            for i in top3_idx
        ]

        return jsonify({
            "glosa":      labels[top_idx] if top_idx < len(labels) else f"clase_{top_idx}",
            "conf":       top_conf,
            "end_prob":   end_prob,
            "top3":       top3,
            "num_frames": len(kpt_list),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/tts", methods=["POST"])
def tts():
    """
    Body JSON: { "text": str }
    Response:  audio/mpeg stream
    """
    data = request.get_json(force=True)
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "Texto vacío"}), 400
    try:
        from gtts import gTTS
        import io

        # Prepend a short pause so the first word isn't clipped.
        # A leading comma makes gTTS insert ~300 ms of silence at the start.
        padded_text = ", " + text

        tts_obj = gTTS(text=padded_text, lang="es", slow=False)
        buf = io.BytesIO()
        tts_obj.write_to_fp(buf)
        buf.seek(0)

        from flask import Response
        return Response(buf.read(), mimetype="audio/mpeg")
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🤟  LSM Recognizer + Translator — Servidor Web")
    print("=" * 60)
    load_resources()
    print("\n🚀 Servidor en https://localhost:8443\n" + "=" * 60)
    app.run(host="0.0.0.0", port=8443, debug=False,
            ssl_context=("cert.pem", "key.pem"))
