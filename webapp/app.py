"""
app.py — LSM Recognizer Web Server
===================================
Recibe landmarks preprocesados desde el frontend (JS + MediaPipe)
y corre inferencia con el modelo ONNX exportado.

Endpoints:
  GET  /            → index.html
  POST /infer       → {"landmarks": [[f0_0, f0_1, ...], ...], "valid": [...]}
                    ← {"glosa": str, "conf": float, "end_prob": float,
                        "top3": [["glosa", conf], ...]}
  GET  /labels      → lista de glosas ordenadas por índice
  GET  /health      → estado del servidor
"""

import json
import os
import sys
import torch

import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ─── Rutas configurables ──────────────────────────────────────────────────────
MODEL_PATH  = os.environ.get("MODEL_PATH",  "model.onnx")
LABELS_PATH = os.environ.get("LABELS_PATH", "glosa_labels.json")
CKPT_PATH   = os.environ.get("CKPT_PATH",   "best_model.ptrom")

# ─── Estado global ────────────────────────────────────────────────────────────
ort_session  = None
input_name   = None
labels: list = []


# ─────────────────────────────────────────────────────────────────────────────
# Carga de recursos al arrancar
# ─────────────────────────────────────────────────────────────────────────────

def load_resources():
    global ort_session, input_name, labels

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
    Body JSON:
      {
        "landmarks": [[266 floats], ...],   # T frames, ya pre-procesados
        "valid":     [true, true, ..., false, ...]  # TMAX bools
      }
    """
    if ort_session is None:
        return jsonify({"error": "Modelo no cargado"}), 503

    data = request.get_json(force=True)
    if not data or "landmarks" not in data:
        return jsonify({"error": "Falta el campo 'landmarks'"}), 400

    try:
        lm_list = data["landmarks"]   # list[list[float]]  shape (TMAX, 266)
        vl_list = data.get("valid")   # list[bool]          shape (TMAX,)

        x = np.array(lm_list, dtype=np.float32)          # (TMAX, 266)
        x = x[np.newaxis, ...]                            # (1, TMAX, 266)

        ort_inputs = {input_name: x}
        outputs    = ort_session.run(None, ort_inputs)
        glosa_logits   = outputs[0][0]   # (num_classes,)
        trigger_logits = outputs[1][0]   # (1,)

        # Softmax sobre glosa
        ex   = np.exp(glosa_logits - glosa_logits.max())
        prob = ex / ex.sum()

        end_prob = float(1 / (1 + np.exp(-trigger_logits[0])))   # sigmoid

        top_idx  = int(prob.argmax())
        top_conf = float(prob[top_idx])

        top3_idx  = prob.argsort()[::-1][:3]
        top3      = [
            [labels[i] if i < len(labels) else f"clase_{i}", float(prob[i])]
            for i in top3_idx
        ]

        return jsonify({
            "glosa":    labels[top_idx] if top_idx < len(labels) else f"clase_{top_idx}",
            "conf":     top_conf,
            "end_prob": end_prob,
            "top3":     top3,
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
    print("\n🚀 Servidor en http://localhost:5000\n" + "=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)