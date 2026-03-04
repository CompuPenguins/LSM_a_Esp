"""
interpolate_landmarks.py
────────────────────────
Post-procesa el CSV generado por video_to_landmarks.py e interpola
los gaps de landmarks (manos / cara) que duran poco tiempo.

Estrategia:
  - Carga el CSV con pandas.
  - Para cada columna de coordenadas (x, y, z) usa interpolación
    lineal, pero solo rellena gaps de hasta MAX_GAP frames consecutivos.
    Gaps más largos se dejan como NaN (detección genuinamente ausente).

Uso:
    python3 interpolate_landmarks.py -i landmarks.csv -o landmarks_interp.csv
    python3 interpolate_landmarks.py -i landmarks.csv -o landmarks_interp.csv --max-gap 5
"""

import pandas as pd
import numpy as np
import argparse
import sys


def interpolate_landmarks(input_csv, output_csv, max_gap):
    print(f"Leyendo  : {input_csv}")
    df = pd.read_csv(input_csv)

    # Columnas de coordenadas — todas las que terminan en _x, _y, _z
    coord_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
    print(f"Columnas a interpolar : {len(coord_cols)}")

    original_nan = df[coord_cols].isna().sum().sum()

    for col in coord_cols:
        df[col] = (
            df[col]
            .interpolate(method="linear", limit=max_gap, limit_direction="both")
        )

    filled_nan = original_nan - df[coord_cols].isna().sum().sum()
    remaining  = df[coord_cols].isna().sum().sum()

    df.to_csv(output_csv, index=False)

    print(f"NaN originales  : {original_nan}")
    print(f"NaN rellenados  : {filled_nan}  (gaps ≤ {max_gap} frames)")
    print(f"NaN restantes   : {remaining}  (gaps > {max_gap} frames, dejados como NaN)")
    print(f"Guardado en     : {output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Interpola gaps cortos en CSV de landmarks.")
    p.add_argument("--input",   "-i", required=True, help="CSV de entrada (video_to_landmarks output)")
    p.add_argument("--output",  "-o", required=True, help="CSV de salida con interpolación")
    p.add_argument("--max-gap", "-g", type=int, default=3,
                   help="Máximo de frames consecutivos a interpolar (default: 3)")
    args = p.parse_args()

    interpolate_landmarks(args.input, args.output, args.max_gap)