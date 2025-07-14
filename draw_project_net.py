#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:22:20 2025

@author: cristiantobar
"""
import os
import re
import pandas as pd
from graphviz import Digraph

# Ruta de entrada y salida
input_folder = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/datos_schedules_construccion"
output_folder = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/data_understanding/nets"

# Asegurar que exista la carpeta de salida
os.makedirs(output_folder, exist_ok=True)

# Función para interpretar relaciones como transiciones
def parse_relation(rel_str):
    match = re.match(r"(\d+)(FS|SS)([+-]\d+)?(d|w)?", rel_str)
    if match:
        pred_id, rel_type, lag, unit = match.groups()
        lag = lag or ''
        unit = unit or ''
        label = f"{rel_type}{lag}{unit}"
        return pred_id, label
    return None, None

# Diccionario para guardar matrices Pre y Post
matrices_por_proyecto = {}

# Procesar todos los archivos Excel
for file in os.listdir(input_folder):
    if file.endswith(".xlsx"):
        try:
            file_path = os.path.join(input_folder, file)
            df = pd.read_excel(file_path, sheet_name="Baseline Schedule", header=1)

            # Eliminar la primera fila de datos reales si está vacía
            df = df.iloc[1:].reset_index(drop=True)
            
            # Filtrar las filas que serán eliminadas (aquellas donde ambas columnas sean NaN)
            df[df[['Predecessors', 'Successors']].isna().all(axis=1)]
            df = df.reset_index(drop=True)
            
            # Identificar nodos de inicio y fin
            start_nodes = df[pd.isna(df['Predecessors'])]['ID'].astype(str).tolist()
            end_nodes = df[pd.isna(df['Successors'])]['ID'].astype(str).tolist()

            # Crear grafo estilo metro
            dot = Digraph(format='pdf')
            dot.attr(rankdir='LR')

            # Añadir nodos de lugar
            for _, row in df.iterrows():
                act_id = str(row["ID"])
                act_name = row["Name"]
                label = f"{act_id}. {act_name}"
                if act_id in start_nodes:
                    dot.node(f"P{act_id}", label, shape='circle', style='filled', fillcolor='gold')
                elif act_id in end_nodes:
                    dot.node(f"P{act_id}", label, shape='circle', style='filled', fillcolor='red')
                else:
                    dot.node(f"P{act_id}", label, shape='circle', style='filled', fillcolor='lightblue')

            # Añadir transiciones
            for _, row in df.iterrows():
                act_id = str(row["ID"])
                act_duration = str(row["Duration"])
                if pd.isna(row["Predecessors"]):
                    continue
                preds = str(row["Predecessors"]).split(";")
                for rel in preds:
                    pred_id, label = parse_relation(rel.strip())
                    if pred_id is not None:
                        trans_id = f"T_{pred_id}_{act_id}_{label}"
                        trans_label = f"{label}\\n({act_duration})"
                        dot.node(trans_id, trans_label, shape='box', style='filled', fillcolor='lightgreen')
                        dot.edge(f"P{pred_id}", trans_id)
                        dot.edge(trans_id, f"P{act_id}")

            # Guardar con el nombre del archivo
            filename_base = os.path.splitext(file)[0].replace(" ", "_")
            output_path = os.path.join(output_folder, f"{filename_base}_petri_net")
            dot.render(output_path, cleanup=True)

            print(f"✅ PDF generado: {output_path}.pdf")
        except Exception as e:
            print(f"⚠️ Error procesando {file}: {e}")
