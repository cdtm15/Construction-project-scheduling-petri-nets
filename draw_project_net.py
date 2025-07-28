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
import numpy as np
import sympy as sp
from itertools import combinations
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler               # Para normalización
from scipy.cluster.hierarchy import linkage, fcluster        # Para clustering jerárquico
import matplotlib.pyplot as plt                              # Para visualización
import seaborn as sns                                        # Para gráficos más estéticos
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier


# Ruta de entrada y salida
input_folder = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/sifones_test"
#input_folder = "/Users/cristiantobar/Library/CloudStorage/OneDrive-unicauca.edu.co/doctorado_cristian/doctorado_cristian/procesamiento_datos/experimentos_schedulings/datos_schedules_construccion" 
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

# Función mejorada: T- y P-componentes como subconjuntos reales
def calcular_componentes_C(pre, post, places, transitions):
    C = post - pre
    C_sym = sp.Matrix(C)

    nullspace_T = np.array(C_sym.nullspace()).astype(object)
    t_componentes = []
    for vec in nullspace_T:
        indices = [i for i, val in enumerate(vec) if val != 0]
        if indices:
            t_componentes.append([transitions[i] for i in indices])
    
    C_sym_T = C_sym.T
    nullspace_P = np.array(C_sym_T.nullspace()).astype(object)
    p_componentes = []
    for vec in nullspace_P:
        indices = [i for i, val in enumerate(vec) if val != 0]
        if indices:
            p_componentes.append([places[i] for i in indices])

    return {
        "matriz_C": C.tolist(),
        "t_componentes": t_componentes,
        "p_componentes": p_componentes
    }

def calcular_sifones_trampas(pre, post, places):
    n_places, n_trans = pre.shape
    resultados = {
        "sifones": [],
        "trampas": []
    }

    for i in range(n_places):
        input_transitions = set(j for j in range(n_trans) if post[i, j] > 0)
        output_transitions = set(j for j in range(n_trans) if pre[i, j] > 0)

        if input_transitions and not output_transitions:
            resultados["trampas"].append(places[i])
        if output_transitions and not input_transitions:
            resultados["sifones"].append(places[i])

    return resultados

def calcular_sifones_trampas_formales(pre, post, places, max_subset_size= 5):
    n_places, n_trans = pre.shape
    resultados = {"sifones": [], "trampas": []}
    conteo_por_tamano = defaultdict(lambda: {"sifones": 0, "trampas": 0})

    print(f"Evaluando conjuntos de lugares (máximo tamaño = {max_subset_size})...")

    
    for k in range(1, n_places + 1):
    #for k in range(1, min(max_subset_size + 1, n_places + 1)):
        total_combs = len(list(combinations(range(n_places), k)))
        print(f"  - Subconjuntos de tamaño {k}: {total_combs} combinaciones")
        
        #for subset in combinations(range(n_places), k):
        for idx, subset in enumerate(combinations(range(n_places), k), start=1):

            S = set(subset)

            entrada_trans = {j for i in S for j in range(n_trans) if pre[i, j] > 0}
            salida_trans = {j for i in S for j in range(n_trans) if post[i, j] > 0}

            # Sifón: transiciones que consumen desde S también deben producir en S
            if entrada_trans <= salida_trans:
                resultados["sifones"].append([places[i] for i in S])
                conteo_por_tamano[k]["sifones"] += 1

            # Trampa: transiciones que producen en S también deben consumir desde S
            if salida_trans <= entrada_trans:
                resultados["trampas"].append([places[i] for i in S])
                conteo_por_tamano[k]["trampas"] += 1
            
            if idx % 500 == 0:
                print(f"    > Procesadas {idx} combinaciones de tamaño {k}...")
            
    print("✅ Cálculo de sifones y trampas completado.")
        
    # Métricas de resumen
    n_sifones = len(resultados["sifones"])
    n_trampas = len(resultados["trampas"])
    tamano_prom_sifon = np.mean([len(s) for s in resultados["sifones"]]) if resultados["sifones"] else 0
    tamano_prom_trampa = np.mean([len(t) for t in resultados["trampas"]]) if resultados["trampas"] else 0
    tamano_max_sifon = max([len(s) for s in resultados["sifones"]]) if resultados["sifones"] else 0
    tamano_max_trampa = max([len(t) for t in resultados["trampas"]]) if resultados["trampas"] else 0

    # Visualización
    tamanos = sorted(conteo_por_tamano.keys())
    sifon_vals = [conteo_por_tamano[k]["sifones"] for k in tamanos]
    trampa_vals = [conteo_por_tamano[k]["trampas"] for k in tamanos]

    plt.figure(figsize=(10, 5))
    plt.plot(tamanos, sifon_vals, marker='o', label='Sifones')
    plt.plot(tamanos, trampa_vals, marker='s', label='Trampas')
    plt.title("Número de subconjuntos detectados como sifones o trampas por tamaño")
    plt.xlabel("Tamaño del subconjunto")
    plt.ylabel("Cantidad detectada")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "sifones": resultados["sifones"],
        "trampas": resultados["trampas"],
        "n_sifones": n_sifones,
        "n_trampas": n_trampas,
        "tamano_prom_sifon": tamano_prom_sifon,
        "tamano_prom_trampa": tamano_prom_trampa,
        "tamano_max_sifon": tamano_max_sifon,
        "tamano_max_trampa": tamano_max_trampa,
        "conteo_por_tamano": conteo_por_tamano
    }
    
    return resultados


# Extraer indicadores
def extraer_indicadores_por_proyecto(matrices_por_proyecto):
    resumen = []
    for nombre, datos in matrices_por_proyecto.items():
        pre = datos["Pre"]
        post = datos["Post"]
        places = datos["places"]
        transitions = datos["transitions"]
        
        componentes = calcular_componentes_C(pre, post, places, transitions)
        #sif_trap = calcular_sifones_trampas_formales(pre, post, places)
        
        
        entradas = pre.sum(axis=1)
        salidas = post.sum(axis=1)
        resumen.append({
            "proyecto": nombre,
            "n_places": len(places),
            "n_transitions": len(transitions),
            "n_t_componentes": len(componentes["t_componentes"]),
            "n_p_componentes": len(componentes["p_componentes"]),
            #"n_sifones": len(sif_trap["sifones"]),
            #"n_trampas": len(sif_trap["trampas"]),
            "norm_frobenius_C": np.linalg.norm(post - pre, ord='fro'),
            "solo_entrada_places": int(np.sum((entradas > 0) & (salidas == 0))),
            "solo_salida_places": int(np.sum((salidas > 0) & (entradas == 0))),
            "intermedios_places": int(np.sum((entradas > 0) & (salidas > 0)))
        })
    return pd.DataFrame(resumen)

def bootstrap_rf_ct(X_encoded, y, uncert_source, lit):
    
    # === 4. Random Forest para feature importance + bootstrap ===
    feature_counts_rf = defaultdict(int)
    feature_importance_values = defaultdict(list)

    n_iterations = 1000

    for i in range(n_iterations):
        X_resampled, y_resampled = resample(X_encoded, y, replace=True, random_state=42+i)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_resampled, y_resampled)
        
        for idx, col in enumerate(X_encoded.columns):
            imp = rf.feature_importances_[idx]
            if imp > 0:
                feature_counts_rf[col] += 1
            feature_importance_values[col].append(imp)

    # Consolidar en DataFrame
    df_importancia_rf = pd.DataFrame({
        'Feature': list(feature_counts_rf.keys()),
        'Frequency': [feature_counts_rf[k] for k in feature_counts_rf.keys()],
        'MeanImportance': [np.mean(feature_importance_values[k]) for k in feature_counts_rf.keys()],
        'StdImportance': [np.std(feature_importance_values[k]) for k in feature_counts_rf.keys()]
    }).sort_values(by='Frequency', ascending=False)
    
    # Ordenar por importancia
    df_plot = df_importancia_rf.sort_values(by='MeanImportance', ascending=True)
    
    return df_plot

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
            # # Aplicar a un proyecto
            # pre = matrices_por_proyecto[file]["Pre"]
            # post = matrices_por_proyecto[file]["Post"]
            # places = matrices_por_proyecto[file]["places"]
            # transitions = matrices_por_proyecto[file]["transitions"]
            
            # # Calcular componentes y estructuras
            # componentes = calcular_componentes_C(pre, post, places, transitions)
            # estructuras = calcular_sifones_trampas(pre, post, places)

            # indicadores_estructurales[file] = {
            #     "matriz_C": componentes["matriz_C"],
            #     "T_componentes": componentes["t_componentes"],
            #     "P_componentes": componentes["p_componentes"],
            #     "Sifones": estructuras["sifones"],
            #     "Trampas": estructuras["trampas"]
            # }
            
            df_indicadores = extraer_indicadores_por_proyecto(matrices_por_proyecto)
            
            # Guardar gráfico
            #filename_base = os.path.splitext(file)[0].replace(" ", "_")
            #output_path = os.path.join(output_folder, f"{filename_base}_petri_net")
            #dot.render(output_path, cleanup=True)

        except Exception as e:
            print(f"⚠️ Error procesando {file}: {e}")


# Paso 1: Preparar los datos
df_clean = df_indicadores.drop(columns=['proyecto'])
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_clean)

# Paso 2: Calcular el linkage (si no lo has hecho ya)
Z = linkage(scaled, method='ward')

# Paso 3: Graficar el dendrograma
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
plt.title("📊 Dendrograma (Método del Codo)")
plt.xlabel("Proyectos (agrupados)")
plt.ylabel("Distancia (Ward)")
plt.tight_layout()
plt.show()

# Rango de valores de clusters a probar
rango_clusters = range(2, 10)
scores = []

for k in rango_clusters:
    etiquetas = fcluster(Z, t=k, criterion='maxclust')
    score = silhouette_score(scaled, etiquetas)
    scores.append(score)
    print(f"Clusters: {k} → Silhouette Score: {score:.4f}")

# Graficar los resultados
plt.figure(figsize=(8, 5))
plt.plot(rango_clusters, scores, marker='o')
plt.title("🔍 Silhouette Score por número de clusters")
plt.xlabel("Número de clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

df_clean = df_indicadores.drop('proyecto', axis=1)
# Lista de variables a graficar
variables = list(df_clean.columns)

# # Normalización
# scaler = MinMaxScaler()
# scaled = scaler.fit_transform(df_clean)

# # Clustering jerárquico
# Z = linkage(scaled, method='ward')
clusters = fcluster(Z, t=2, criterion='maxclust')

# Asignar al df_clean
df_clean["cluster"] = clusters
df_indicadores["cluster"] = clusters

# Paleta de colores personalizada para los clusters
colores_clusters = sns.color_palette("Set2", n_colors=4)

# Crear figura con subplots
fig, axes = plt.subplots(4, 2, figsize=(14, 14))
axes = axes.flatten()

for i, var in enumerate(variables):
    sns.boxplot(data=df_clean, x='cluster', y=var, ax=axes[i], palette=colores_clusters)
    axes[i].set_title(f"Distribución de {var} por cluster")
    axes[i].set_xlabel("Cluster")
    axes[i].set_ylabel(var)

plt.tight_layout()
plt.show()

# Seleccionar los dos primeros proyectos por cluster
df_analizados = df_indicadores[['proyecto', 'cluster']].groupby('cluster').head(2).reset_index(drop=True)

# Seleccionar solo columnas necesarias
df_todos = df_indicadores[['proyecto', 'cluster']].sort_values(by='cluster').reset_index(drop=True)


