#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 09:25:46 2025

@author: cristiantobar
"""

import os
import re
import pandas as pd
from graphviz import Digraph
import numpy as np
import sympy as sp
from itertools import combinations
import matplotlib.pyplot as plt
from collections import defaultdict
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger, LpStatus, value

# Ruta de entrada y salida
#input_folder = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/sifones_test"
input_folder = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/datos_schedules_construccion" 
output_folder = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/data_understanding/nets"

# Ruta del archivo excel consolidado de los proyectos
ruta_consolidado_proj = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/DSLIB_Analysis_Scheet.xlsx"
df_consolidado_proj   = pd.read_excel(ruta_consolidado_proj, sheet_name='all_data_combining')  # Carga todas las hojas 

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

def calcular_componentes_test(C, transitions, places):
    C_sym = sp.Matrix(C)
    # -------------------------
    # T-INVARIANTES
    # -------------------------
    nullspace_T = C_sym.nullspace()
    t_invariantes_enteros = []
    t_componentes_nombres = []

    for vec in nullspace_T:
        # Convertir a coeficientes enteros
        lcm = sp.lcm([term.q for term in vec])
        vec_int = [int(lcm * term) for term in vec]

        # Incluir solo si todos los valores son ≥ 0 y alguno > 0
        if all(v >= 0 for v in vec_int) and any(v > 0 for v in vec_int):
            t_invariantes_enteros.append(vec_int)

            if transitions:
                indices = [i for i, val in enumerate(vec_int) if val != 0]
                t_componentes_nombres.append([transitions[i] for i in indices])
    
    # Matriz binaria (1 si transición participa, 0 si no)
    t_invariantes_binarios = []
    for vec in t_invariantes_enteros:
        bin_vec = [1 if v != 0 else 0 for v in vec]
        t_invariantes_binarios.append(bin_vec)
            
    # -------------------------
    # P-INVARIANTES
    # -------------------------
    C_sym_T = C_sym.T
    nullspace_P = C_sym_T.nullspace()
    p_componentes_nombres = []

    for vec in nullspace_P:
        lcm = sp.lcm([term.q for term in vec])
        vec_int = [int(lcm * term) for term in vec]

        if all(v >= 0 for v in vec_int) and any(v > 0 for v in vec_int):
            if places:
                indices = [i for i, val in enumerate(vec_int) if val != 0]
                p_componentes_nombres.append([places[i] for i in indices])
    

    
    return {
        "matriz_C": C,
        "t_invariantes_enteros": t_invariantes_enteros,
        "t_invariantes_binarios": t_invariantes_binarios,
        "t_componentes_nombres": t_componentes_nombres,
        "p_componentes_nombres": p_componentes_nombres
    }


# Extraer indicadores
def extraer_indicadores_por_proyecto(matrices_por_proyecto):

    resumen = []
    for nombre, datos in matrices_por_proyecto.items():
        pre = datos["Pre"]
        post = datos["Post"]
        places = datos["places"]
        transitions = datos["transitions"]
        
        C = post - pre
        componentes = calcular_componentes_test(C, transitions, places)

        entradas = pre.sum(axis=1)
        salidas = post.sum(axis=1)
        resumen.append({
            "proyecto": nombre,
            "n_places": len(places),
            "n_transitions": len(transitions),
            "n_t_componentes": len(componentes["t_invariantes_binarios"]),
            "n_p_componentes": len(componentes["p_componentes_nombres"]),
            #"n_sifones": len(sif_trap["sifones"]),
            #"n_trampas": len(sif_trap["trampas"]),
            "norm_frobenius_C": np.linalg.norm(post - pre, ord='fro'),
            "solo_entrada_places": int(np.sum((entradas > 0) & (salidas == 0))),
            "solo_salida_places": int(np.sum((salidas > 0) & (entradas == 0))),
            "intermedios_places": int(np.sum((entradas > 0) & (salidas > 0)))
        })
    return pd.DataFrame(resumen)

# Diccionario para guardar matrices Pre y Post
matrices_por_proyecto = {}
indicadores_estructurales = {}

# Este bloque solo funcionará si se cargan archivos Excel al entorno
for file in os.listdir(input_folder):
    if file.endswith(".xlsx"): 
        try:
            file_path = os.path.join(input_folder, file)
            df = pd.read_excel(file_path, sheet_name="Baseline Schedule", header=1)
            df = df.iloc[1:].reset_index(drop=True)
    
            # Filtrar las filas que serán eliminadas (aquellas donde ambas columnas sean NaN)
            df = df[~df[['Predecessors', 'Successors']].isna().all(axis=1)]
            df = df.reset_index(drop=True)
                        
            places = {}
            transitions = {}
            pre_matrix = []
            post_matrix = []
            place_indices = {}
            transition_indices = {}
            place_counter = 0
            transition_counter = 0
            # Crear grafo
            dot = Digraph(format='pdf')
            dot.attr(rankdir='LR')
                        
            start_nodes = df[pd.isna(df['Predecessors'])]['ID'].astype(str).tolist()
            end_nodes = df[pd.isna(df['Successors'])]['ID'].astype(str).tolist()
    
            # Añadir lugares
            for _, row in df.iterrows():
                act_id = str(row["ID"])
                act_name = row["Name"]
                label = f"P{act_id}\n{act_name}"
                place_id = f"P{act_id}"
                place_indices[place_id] = place_counter
                place_counter += 1
                if act_id in start_nodes:
                    dot.node(place_id, label, shape='circle', style='filled', fillcolor='gold')
                elif act_id in end_nodes:
                    dot.node(place_id, label, shape='circle', style='filled', fillcolor='red')
                else:
                    dot.node(place_id, label, shape='circle', style='filled', fillcolor='lightblue')
    
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
                        tid = f"T_{pred_id}_{act_id}_{label}"
                        label_txt = f"T{transition_counter}\n{label}\\n({act_duration})"
                        transition_indices[tid] = transition_counter
                        transition_counter += 1
                        dot.node(tid, label_txt, shape='box', style='filled', fillcolor='lightgreen')
                        dot.edge(f"P{pred_id}", tid)
                        dot.edge(tid, f"P{act_id}")
                        
            # Crear matrices vacías
            pre = np.zeros((len(place_indices), len(transition_indices)), dtype=int)
            post = np.zeros((len(place_indices), len(transition_indices)), dtype=int)
            
            # if file == 'C2023-01 House renovation.xlsx':
            #         # Rellenar matrices Pre y Post
            #         for t_id, t_idx in transition_indices.items():
                        
            #                 match = re.match(r"T_(\d+)_(\d+)_", t_id)
                            
            #                 if match:
            #                     p_from = f"P{match.group(1)}"
            #                     p_to = f"P{match.group(2)}"
                                
            #                 try:
            #                     if p_from in place_indices:
            #                         pre[place_indices[p_from], t_idx] = 1
            #                     if p_to in place_indices:
            #                         post[place_indices[p_to], t_idx] = 1
            #                 except Exception as e:
            #                      breakpoint()
                
            # Rellenar matrices Pre y Post
            for t_id, t_idx in transition_indices.items():
                match = re.match(r"T_(\d+)_(\d+)_", t_id)
                if match:
                    p_from = f"P{match.group(1)}"
                    p_to = f"P{match.group(2)}"
                    if p_from in place_indices:
                        pre[place_indices[p_from], t_idx] = 1
                    if p_to in place_indices:
                        post[place_indices[p_to], t_idx] = 1
            
            matrices_por_proyecto[file] = {
                "Pre": pre,
                "Post": post,
                "places": list(place_indices.keys()),
                "transitions": list(transition_indices.keys())
            }
    
            df_indicadores = extraer_indicadores_por_proyecto(matrices_por_proyecto)
            # Guardar gráfico
            #filename_base = os.path.splitext(file)[0].replace(" ", "_")
            #output_path = os.path.join(output_folder, f"{filename_base}_petri_net")
            #dot.render(output_path, cleanup=True)
        except Exception as e:
            print(f"⚠️ Error procesando {file}: {e}")
            
            
