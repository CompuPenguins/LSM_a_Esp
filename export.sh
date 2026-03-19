#!/bin/bash
cd /home/tumbadoboy/UNIVERSIDAD/LSM_a_Esp
eval "$(conda shell.bash hook)"
conda activate lsm
python onnx_export.py
