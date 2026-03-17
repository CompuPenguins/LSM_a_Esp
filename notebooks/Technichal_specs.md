Especificación Técnica: LSM-Transformer con AimCLR y Morfología Sintética

## 0. Contexto del Dataset (Real)

**Fuente**: `corpus_LSM_esp/lsm_dataset.parquet` (RTMPose 133 keypoints COCO-WholeBody)

| Métrica           | Valor              |
|-------------------|--------------------|
| Videos totales    | 2,447              |
| Glosas únicas     | 249                |
| Señadores/glosa   | 7–11 (media ~9.8)  |
| Duración promedio | 61 frames (2.0 seg @ 30fps) |
| Rango duración    | 9–196 frames (0.3–6.5 seg)  |
| FPS               | 30                 |
| Resolución        | ~746 × 530 píx     |
| Landmarks/frame   | 133 × 3 (x, y, score) = 399 valores |
| Formato data      | bytes (np.ndarray serializado con np.save) |

**Estructura Parquet**:
```
glosa, intento, video_id, fps, total_frames, width, height,
keypoints (bytes: float32 (T, 133, 3)), person_detected (bytes: bool (T,))
```

**Nota Crítica**: La variabilidad en duración (9–196 frames) requiere:
- Padding/truncamiento a una longitud máxima de referencia (ej. 200 frames).
- Masking temporal para ignorar frames de padding durante atención del Transformer.

---

## 1. Definición del Input (Landmark-Only)

El modelo no procesa video. Recibe una secuencia temporal de landmarks extraída del archivo Parquet.

    Formato: Tensor (Batch, Seconds, 133, 3) donde Seconds ≤ Tmax=200 frames
    (Coordenadas X, Y y Score de confianza para 133 keypoints COCO-WholeBody).

    Normalización: Centrado de coordenadas respecto a los hombros (puntos 5–6 de COCO-WholeBody, "shoulder midpoint") y escalado unitario para eliminar variaciones de distancia a la cámara.
    
    **Preprocessing Recomendado**:
    ```python
    # Pseudocódigo: normalización robusto con detección
    def preprocess(keypoints, person_detected, score_thresh=0.3):
        T, 133, 3 = keypoints.shape
        # 1. Mascara de confianza: marcar landmarks con score < umbral
        mask = keypoints[..., 2] < score_thresh  # (T, 133)
        
        # 2. Centrado respecto a hombros (kpts 5, 6) si ambos detectados
        shoulder_midpoint = (keypoints[:, 5, :2] + keypoints[:, 6, :2]) / 2  # (T, 2)
        keypoints_centered = keypoints[:, :, :2] - shoulder_midpoint[:, None, :]
        
        # 3. Escalado: norma L2 de los hombros a cadera (kpts 11, 12)
        scale = np.linalg.norm(keypoints[:, 11:13, :2] - shoulder_midpoint[:, None, :], axis=(1,2))
        scale = np.clip(scale, 1e-8, None)  # evitar división por cero
        keypoints_scaled = keypoints_centered / scale[:, None, None]
        
        # 4. Reconstruir con scores y masking
        output = np.zeros_like(keypoints)
        output[:, :, :2] = keypoints_scaled
        output[:, :, 2] = keypoints[:, :, 2]
        output[mask] = 0  # opcional: anular landmarks bajo confianza
        
        return output, mask
    ```

2. Arquitectura del Modelo: Hybrid Temporal Transformer

Se utilizará un codificador Transformer para capturar la semántica de la seña a lo largo de toda la secuencia temporal.

**Tratamiento de Longitudes Variables**: Dado el rango 9–196 frames:
- **Tmax = 200 frames** elegido como umbral (cubre >99% de los videos).
- **Padding**: Rellenar secuencias cortas con ceros hasta Tmax.
- **Attention Mask**: Pasar máscara de padding al Transformer para ignorar frames añadidos.

**Componentes Principales**:

1. **Flat Embedding** (Espacial)
   - Projectión lineal: 399 valores (133×3) → dmodel = 512
   - Output: (B, Tmax, 512)

2. **Positional Encoding** (Temporal)
   - Inyección de información temporal mediante funciones senoidales.
   - Formula estándar: PE(t, 2i) = sin(t / 10000^(2i/dmodel)), PE(t, 2i+1) = cos(...)
   - Permite que el Transformer distinga el orden de los movimientos.

3. **Transformer Core**
   - 6 capas de Encoder Multi-Head Attention
   - 8 cabezas de atención
   - FFN hidden = 2048
   - Dropout = 0.1

4. **Salidas Multi-tarea (Heads)**
   - **Clasificador de Glosas**: 
     - Input: CLS token (especial) o promedio de secuencia (T_valid tokens)
     - Layers: Linear(512) → ReLU → Linear(512) → ReLU → Linear(249)
     - Output: logits para 249 clases
   
   - **Detector de Finalización (Early Trigger)**: 
     - Input: tokens últimos N frames (ej. últimos 30 frames)
     - Layers: Linear(512*N) → ReLU → Linear(128) → ReLU → Linear(1)
     - Output: probabilidad binaria (seña terminada: p>0.85)

## 3. Pipeline de Aumentación Anatómica (Morfología Sintética)

Para simular diferentes usuarios y mejorar la generalización ante solo ~9.8 señadores/glosa, el DataLoader aplicará transformaciones que conservan proporciones físicas y relaciones esqueletales.

**Rationale**: Con 2,447 videos totales y 249 glosas, la augmentación es crítica para evitar sobrentrenamiento. Las aumentaciones anatómicas son más realistas que rotaciones/shears aleatorios.

### A. Variación de Complexión (Anchura de Clavícula)

- **Pivotes**: Puntos 5 (Hombro Izq) y 6 (Hombro Der) de COCO-WholeBody
- **Escala**: sbuild ∈ [0.90, 1.10] (simula variedad corporal: fornido ↔ delgado)
- **Algoritmo**:
  ```python
  shoulder_midpoint = (kpts[:, 5, :2] + kpts[:, 6, :2]) / 2
  shoulder_dist = norm(kpts[:, 6, :2] - kpts[:, 5, :2])
  
  # Desplazar lateralmente brazos/manos si hombros se ensanchan
  left_arm_slice = slice(7, 11)   # COCO body: left_shoulder to left_wrist
  right_arm_slice = slice(12, 16)
  
  dx_shoulder = (kpts[:, 6, :2] - kpts[:, 5, :2]) * (sbuild - 1) / 2
  kpts[:, left_arm_slice, :2] -= dx_shoulder[:, None, :]
  kpts[:, right_arm_slice, :2] += dx_shoulder[:, None, :]
  ```

### B. Variación de Extremidades (Escalado Isótropo de Manos)

- **Pivotes**: Muñecas (Puntos 91 [izq], 112 [der] en COCO WholeBody de 133 kpts)
- **Escala**: shand ∈ [0.95, 1.05] (simula tamaño de manos: pequeño ↔ grande)
- **Puntos afectados**: 
  - Mano izquierda: índices 92–112 (puntos 92–112 de COCO-Holistic son los 21 puntos de mano)
  - Mano derecha: índices 113–133
- **Algoritmo**:
  ```python
  # Mano izquierda
  wrist_l = kpts[:, 91, :2]  # pivot
  hand_l_indices = slice(92, 113)
  offset = kpts[:, hand_l_indices, :2] - wrist_l[:, None, :]
  kpts[:, hand_l_indices, :2] = wrist_l[:, None, :] + offset * shand
  
  # Análogo para mano derecha (wrist 112, índices 113–133)
  ```

### C. Ruido Gaussiano Leve (Score-Aware)

- **Amplitud**: σ ∈ [0, 2 píxeles] (realista para manos con mocap, no excesivo)
- **Máscara**: Aplicar solo a keypoints con score > 0.5 para evitar romper "indetectables"
  ```python
  noise = np.random.normal(0, σ, kpts.shape[:-1] + (2,))
  mask = (kpts[:, :, 2] > 0.5)[:, :, None]
  kpts[:, :, :2] += noise * mask
  ```

## 4. Marco de Entrenamiento con AimCLR (Regularización Self-Supervised)

Se utilizará la metodología AimCLR como regularización (no como pre-entrenamiento completo, para acelerar convergencia):

### Aumentaciones Extremas (Contrastive Views)

Para la misma secuencia de seña, se generan dos vistas usando transformaciones agresivas:

1. **Temporal Flip**: Invertir la secuencia en tiempo (~50% de prob)
   - Algunos movimientos son simétricos (ej. "ir"), otros no lo son (ej. "pasado" requiere contexto temporal)
   - Propósito: forzar que el modelo use contexto temporal aprendido, no solo patrones de inicio/fin

2. **Axis Masking**: Enmascarar aleatoriamente coordenadas (x o y)
   ```python
   mask_axis = np.random.choice([0, 1])  # masquear eje X o Y
   view[:, :, mask_axis] = 0  # o interpolar desde vecinos
   ```

3. **Gaussian Blur Temporal**: Suavizar secuencia mediante convolución 1D
   ```python
   kernel = gaussian_kernel1d(sigma=1.5, order=0, radius=3)
   view_blurred = convolve1d(view, kernel, axis=0)
   ```

### Pérdida D3M (Detail-preserving Divergence Minimization)

Minimizar divergencia entre representaciones de vista original y aumentada:

```
L_D3M = KL(p_original || p_augmented) + KL(p_augmented || p_original)
       donde p = softmax(logits / τ)   (τ ≈ 0.1, temperatura)
```

Propósito: el Transformer debe estar **invariante** a aumentaciones extremas (semántica intacta) pero **sensible** a cambios semánticos reales.

### EADM (Energy-Based Attention Dropout Mask)

Durante entrenamiento, análisis de atención:
- Calcular suma de pesos de atención por landmark (133 canales)
- Dropout estructurado: aquellos landmarks con >80 percentil de atención son descartados aleatoriamente (p=0.3)
  ```python
  attn_energy = torch.sum(attention_weights, dim=1)  # (B, 133)
  high_energy_mask = attn_energy > percentile(attn_energy, 80)
  dropout_mask = high_energy_mask & (torch.rand(...) < 0.3)
  embeddings = embeddings * (~dropout_mask)
  ```
- Propósito: evitar que el modelo dependa excesivamente de un solo dedo o movimiento; forzar redundancia.

**Coeficiente de Pérdida**: λ_AimCLR ∈ [0.1, 0.5] (balance con pérdida de clasificación principal)

## 5. Estrategia de Entrenamiento

### Split de Datos

Dado que cada glosa tiene 7–11 intentos de diferentes señadores:
- **Train**: 70% videos (~1,713 videos) – garantizar representación de múltiples señadores por glosa
- **Val**: 15% videos (~367 videos) – para early stopping y validación
- **Test**: 15% videos (~367 videos) – evaluación final, **NO se toca durante entrenamiento**

**Protocolo**: Split por **glosa + señador** (si metadata lo permite) para evitar *leakage*.  
Si no hay metadata de señador explícita, usar `stratified k-fold` por glosa dentro de cada subset.

### Hiperparámetros de Entrenamiento

| Parámetro                | Valor              |
|-----|-----|
| Optimizer              | AdamW (lr=1e-4, β₁=0.9, β₂=0.999) |
| Scheduler              | CosineAnnealingLR (T_max=100 épocas) |
| Batch Size             | 32 (según GPU disponible)  |
| Epochs                 | 100–150            |
| Loss Weight            | α=1.0 (glosa), β=0.2 (early-trigger), λ=0.3 (AimCLR) |
| Gradient Clipping      | max_norm=1.0       |
| Early Stopping         | Paciencia=20 épocas (Val Macro-F1) |

### Métricas de Validación

- **Top-1 Accuracy** (glosa)
- **Top-3 Accuracy** (glosa)  
- **Macro-F1** (por glosa, importante dado desbalance)
- **AUC-ROC** (early-trigger)

---

## 6. Estrategia de Tiempo Real e Inferencia

### Runtime Offline (Batch)

- **Exportación a ONNX**: Una vez el modelo converja (val macro-F1 saturada), exportar a ONNX para portabilidad
  ```python
  model.eval()
  dummy_input = torch.randn(1, 200, 399)
  torch.onnx.export(model, dummy_input, "lsm_transformer.onnx", 
                    input_names=["batch"], output_names=["glosa_logits", "end_trigger"])
  ```

- **TensorRT** (opcional, si se dispone de Jetson): compilar ONNX a TensorRT para optimización

### Runtime Online (Webcam)

- **Captura**: OpenCV + **MediaPipe Holistic** (balance velocidad vs precisión para webcam)
- **Ventana Deslizante**: Mantener *buffer* de últimos **200 frames** (∼6.7 seg @ 30fps)
  ```python
  frame_buffer = deque(maxlen=200)
  
  while capture:
      ret, frame = cap.read()
      if ret:
          landmarks = mediapipe_inference(frame)  # <1ms
          frame_buffer.append(landmarks)
          
          if len(frame_buffer) == 200:
              # Preprocesar + inferencia Transformer
              x_norm = preprocess(np.array(frame_buffer))
              logits, end_prob = model.onnx_infer(x_norm)
              
              if end_prob > 0.85:
                  glosa_pred = np.argmax(logits)
                  print(f"Glosa: {ID_TO_GLOSA[glosa_pred]} (conf: {softmax(logits)[glosa_pred]:.3f})")
                  frame_buffer.clear()  # limpiar para nueva seña
  ```

- **Latencia Objetivo**: <5ms inferencia Transformer (<20ms incluyendo pre/post-process) para 30 FPS

- **Criterio de Disparo**:
  - `end_prob > 0.85` AND `top-1 confidence > 0.7` → emitir predicción
  - Cooldown de 500ms entre predicciones para evitar duplicados

---

## 7. Entregables Esperados

1. **Módulo de Datos** (`src/dataset.py`)
   - Clase `LSMDataset` cargando desde parquet
   - Funciones de preprocesamiento (normalización, padding)
   - Augmentations (anatómicas + AimCLR)

2. **Modelo** (`src/model.py`)
   - Clase `LSMTransformer` con dos heads (glosa + end-trigger)
   - Componentes: Embedding, PositionalEncoding, TransformerEncoder, MLPHeads

3. **Entrenamiento** (`src/train.py`)
   - Loop de entrenamiento con early stopping
   - Logging de métricas (W&B o TensorBoard)
   - Checkpoint saving por val macro-F1

4. **Inferencia** (`src/inference.py`)
   - Función de inferencia batch
   - ONNX export
   - Webcam pipeline (opcional)

5. **Notebook de Validación** (`notebooks/benchmark.ipynb`)
   - Comparativa: Transformer vs Baselines (LSTM/CNN)
   - Análisis de confusión
   - Visualización de atención (t-SNE/UMAP de representaciones)

6. **README Técnico** (`README_TECHNICAL.md`)
   - Reproducción step-by-step
   - Comando de entrenamiento
   - Descargar modelo pre-entrenado (si aplica)
---

## 8. Checklist de Implementación

Orden recomendado para desarrollo:

- [ ] **Fase 0**: Estructurar codebase (`src/`, configs, logging)
- [ ] **Fase 1**: Implementar `LSMDataset` con augmentations anatómicas
- [ ] **Fase 2**: Implementar `LSMTransformer` (embedding + encoder + heads)
- [ ] **Fase 3**: Loop de entrenamiento básico (pérdida de glosa + early stopping)
- [ ] **Fase 4**: Integrar AimCLR (pérdida D3M + EADM)
- [ ] **Fase 5**: Validación y benchmark (Transformer vs LSTM baseline)
- [ ] **Fase 6**: Exportación a ONNX + prueba de inferencia batch
- [ ] **Fase 7**: Pipeline de webcam (MediaPipe + buffer + inferencia online)
- [ ] **Fase 8**: Documentación y reproducibilidad

---

## 9. Resumen de Ventajas

✓ **Eficiencia Computacional**: 133×3=399 valores/frame << píxeles crudos; latencia <5ms  
✓ **Robustez a Variabilidad**: Augmentación anatómica compensa bajo # señadores (~10/glosa)  
✓ **Captura Temporal**: Transformer entiende orden de movimientos (no solo "qué" sino "cuándo")  
✓ **Escalabilidad**: Estructura limpia permite agregar más lenguas o tareas (traducción a texto)  
✓ **Debugabilidad**: Atención visualizable; augmentations están bajo control

---

## 10. Referencias Técnicas

- COCO-WholeBody: 133 keypoints (pose 17 + feet 4×2 + face 67 + hands 21×2)
- Transformer Encoder: Vaswani et al., "Attention Is All You Need" (2017)
- AimCLR: (adaptación de contrastive learning para acciones corporales)
- RTMPose: https://github.com/open-mmlab/mmpose (offline extraction, high-quality)
- MediaPipe Holistic: Google Research (real-time, webcam-friendly)