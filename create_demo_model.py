#!/usr/bin/env python3
"""
Script para crear un modelo ONNX dummy para demostración.
Este script crea un modelo simple que puede ser usado para probar la aplicación web.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import os

def create_simple_onnx_model(output_path='model.onnx'):
    """Crear un modelo ONNX simple para demostración"""
    
    print("Creando modelo ONNX de demostración...")
    
    # Define el número de clases (ajusta según tu modelo)
    num_classes = 100
    
    # Crear un grafo ONNX simple
    # Entrada: (batch_size, 3, 224, 224) para imágenes
    # Salida: (batch_size, num_classes) predicciones
    
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1, 3, 224, 224]
    )
    
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1, num_classes]
    )
    
    # Crear un nodo Identity simple (sin procesamiento real)
    reshape_const = helper.make_tensor(
        name='shape',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[num_classes],
    )
    
    # Crear constante de weights para simular procesamiento
    weights_data = np.random.randn(num_classes).astype(np.float32)
    weights = helper.make_tensor(
        name='weights',
        data_type=TensorProto.FLOAT,
        dims=[num_classes],
        vals=weights_data,
    )
    
    # Nodo que suma la entrada (para simular procesamiento)
    reduce_sum = helper.make_node(
        'ReduceSum',
        inputs=['input'],
        outputs=['reduced'],
        axes=[1, 2, 3],
        keepdims=True,
    )
    
    # Expandir dimensiones
    expand = helper.make_node(
        'Squeeze',
        inputs=['reduced'],
        outputs=['squeezed'],
    )
    
    # Repetir para crear num_classes valores
    tile_repeats = helper.make_tensor(
        name='tile_repeats',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[num_classes],
    )
    
    tile = helper.make_node(
        'Tile',
        inputs=['squeezed', 'tile_repeats'],
        outputs=['tiled'],
    )
    
    # Añadir pesos aleatorios
    add = helper.make_node(
        'Add',
        inputs=['tiled', 'weights'],
        outputs=['output'],
    )
    
    # Crear el grafo
    graph = helper.make_graph(
        [reduce_sum, expand, tile, add],
        'DemoModel',
        [input_tensor],
        [output_tensor],
        [weights, tile_repeats],
    )
    
    # Crear el modelo
    model = helper.make_model(
        graph,
        producer_name='demo_producer',
        opset_imports=[helper.make_opsetid('', 11)],
    )
    
    # Guardar el modelo
    onnx.save(model, output_path)
    print(f"✅ Modelo ONNX de demostración guardado en: {output_path}")
    print(f"   Entrada: (1, 3, 224, 224)")
    print(f"   Salida: (1, {num_classes})")
    print("\nEste es un modelo de demostración.")
    print("Para usar el modelo real, ejecuta:")
    print("   python onnx_export.py")

if __name__ == '__main__':
    output_file = 'model.onnx'
    
    if os.path.exists(output_file):
        print(f"El archivo {output_file} ya existe.")
        response = input("¿Sobrescribir? (s/n): ")
        if response.lower() != 's':
            print("Operación cancelada.")
            exit(0)
    
    try:
        create_simple_onnx_model(output_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
