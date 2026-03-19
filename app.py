from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import onnxruntime as rt
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__, template_folder='webapp')
CORS(app)

# Cargar el modelo ONNX
model_path = 'model.onnx'
if os.path.exists(model_path):
    sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
else:
    sess = None
    print(f"Advertencia: modelo {model_path} no encontrado")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/infer', methods=['POST'])
def infer():
    try:
        if sess is None:
            return jsonify({'error': 'Modelo no cargado'}), 500
        
        # Obtener la imagen del request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Redimensionar a 224x224 (ajusta según tu modelo)
        img = img.resize((224, 224))
        
        # Convertir a numpy array y normalizar
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Cambiar a formato CHW (PyTorch)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Añadir dimensión de batch
        input_data = np.expand_dims(img_array, axis=0)
        
        # Realizar la inferencia
        output = sess.run([output_name], {input_name: input_data})
        
        # Procesar el output
        result = output[0]
        
        return jsonify({
            'success': True,
            'output': result.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    if sess is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    
    return jsonify({
        'inputs': [{'name': i.name, 'shape': i.shape} for i in inputs],
        'outputs': [{'name': o.name, 'shape': o.shape} for o in outputs]
    })

if __name__ == '__main__':
    print("Iniciando aplicación en http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
