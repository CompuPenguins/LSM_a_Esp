Se tiene que Usar `python3.11` (Si no no funciona mediapipe). Dentro del repositorio:
`pyenv local 3.11.7`
Y luego crear el entorno.
`python -m venv .venv`
`source .venv/bin/activate`
`pip install pandas opencv-python mediapipe matplotlib ipykernel`
(o nadota, nomas cv y mediapipe)

Los modelos .safetensors van en el directorio translators, cada uno en su respectivo directorio segun nombre. Para usarlos se ejecuta el script `translator_model.py` seguido de un String en la forma "esp: \<texto\>" o "lsm: \<texto\>", segun la forma de la oracion. Solo pueden con oraciones simples.

# LSM_a_Esp

Proyecto para pasar de **Lengua de Señas Mexicana (LSM)** a lenguaje natural en español usando una representación por **landmarks**.

## Requisito de Python

Usar **Python 3.11** (MediaPipe no funciona bien fuera de esta versión en este proyecto).

Ejemplo con pyenv:

1. `pyenv local 3.11.7`
2. `python -m venv .venv`
3. `source .venv/bin/activate`

## Instalación de dependencias

Instala dependencias directas (manuales) con:

`pip install -r requirements.txt`


## Dependencias directas incluidas

- `mediapipe==0.10.9` (captura/landmarks en tiempo real)
- `opencv-python==4.13.0.92` (lectura de video y cámara)
- `numpy==1.26.4` (tensores y operaciones numéricas)
- `pandas==2.2.3` (tablas y análisis)
- `pyarrow==17.0.0` (lectura/escritura parquet)
- `rtmlib==0.0.13` (RTMPose/RTMW para extracción offline)
- `onnxruntime==1.20.1` (backend de inferencia para RTMPose)
- `tqdm==4.67.1` (barras de progreso)
- `matplotlib==3.9.2` (visualización EDA)
- `seaborn==0.13.2` (visualización EDA)
- `ipykernel==6.29.5` (uso de notebooks en VS Code/Jupyter)

## Scripts principales

- `build_lsm_dataset.py`: recorre `MSLwords1/` y genera parquet consolidado con landmarks.
- `video_to_landmarks.py`: extrae landmarks de uno o varios videos a parquet.
- `camera.py`: demo de tiempo real con MediaPipe Holistic.
- `interpolate_landmarks.py` / `interpolate_gaps.py`: interpolación de gaps cortos (flujo legado CSV).
- `notebooks/eda_lsm_dataset.ipynb`: EDA del dataset de landmarks.

## Comandos útiles

Construcción de dataset desde carpeta raíz de videos:

`python build_lsm_dataset.py --root MSLwords1 --output corpus_LSM_esp/lsm_dataset.parquet --device cpu --mode balanced`

Extracción para video(s) puntuales:

`python video_to_landmarks.py -i ruta_video.mp4 -o salida.parquet --no-preview`

Demo de webcam:

`python camera.py`
