Se tiene que Usar `python3.11` (Si no no funciona mediapipe). Dentro del repositorio:
`pyenv local 3.11.7`
Y luego crear el entorno.
`python -m venv .venv`
`source .venv/bin/activate`
`pip install pandas opencv-python mediapipe matplotlib ipykernel`
(o nadota, nomas cv y mediapipe)

Los modelos .safetensors van en el directorio translators, cada uno en su respectivo directorio segun nombre. Para usarlos se ejecuta el script `translator_model.py` seguido de un String en la forma "esp: \<texto\>" o "lsm: \<texto\>", segun la forma de la oracion. Solo pueden con oraciones simples.
