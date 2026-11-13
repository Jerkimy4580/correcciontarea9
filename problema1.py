"""
=============================================================================
Problema 1: Análisis de Fuerzas en una Estructura (3x3)
=============================================================================
Descripción:
    Una estructura plana horizontal tiene 9 nodos dispuestos en una cuadrícula
    3x3. La matriz `fuerzas_nodos` guarda la fuerza vertical aplicada en cada
    nodo (en Newtons; positivo = hacia abajo, negativo = hacia arriba).

    La estructura se sostiene en 3 apoyos (nodos) con reacciones verticales
    desconocidas. Con las leyes de la estática (equilibrio de fuerzas y de
    momentos) se plantea un sistema 3x3 y se resuelve para hallar la reacción
    en CADA apoyo.

    Ecuaciones de equilibrio estático (cuerpo rígido en el plano):
        1) ΣFz = 0   ->  ΣR  = ΣF
        2) ΣMx = 0   ->  ΣR·y = ΣF·y      (momentos respecto al eje X)
        3) ΣMy = 0   ->  ΣR·x = ΣF·x      (momentos respecto al eje Y)
    Con 3 apoyos hay 3 incógnitas y 3 ecuaciones: estructura isostática.
=============================================================================
"""

import numpy as np

print("=" * 65)
print("PROBLEMA 1: ANÁLISIS DE FUERZAS EN UNA ESTRUCTURA")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. Datos de entrada
# ---------------------------------------------------------------------------
# Matriz 3x3 de fuerzas aplicadas (N). Fila i = posición en Y, columna j = posición en X.
fuerzas_nodos = np.array([
    [ 12.5,  -3.2,   8.1],
    [  5.4,  18.0,  -2.5],
    [ -4.1,   9.6,  22.4],
])

ESPACIADO_M = 2.0  # distancia entre nodos vecinos (metros)

# Apoyos como (fila, columna). No son colineales, así que el sistema tiene solución única.
apoyos = [(0, 0), (0, 2), (2, 1)]

# ---------------------------------------------------------------------------
# 2. Coordenadas de cada nodo (matrices del mismo tamaño que fuerzas_nodos)
# ---------------------------------------------------------------------------
n_filas, n_cols = fuerzas_nodos.shape
coord_x, coord_y = np.meshgrid(np.arange(n_cols) * ESPACIADO_M,
                               np.arange(n_filas) * ESPACIADO_M)
# coord_x[i, j] = x del nodo (i, j) | coord_y[i, j] = y del nodo (i, j)

print("Matriz de fuerzas aplicadas en los nodos (N):\n", fuerzas_nodos)

# ---------------------------------------------------------------------------
# 3. Lado derecho del sistema: fuerza total y momentos de las cargas
#    (operaciones vectorizadas sobre toda la matriz, sin ciclos)
# ---------------------------------------------------------------------------
fuerza_total = fuerzas_nodos.sum()
momento_x_cargas = (fuerzas_nodos * coord_y).sum()   # Σ F·y
momento_y_cargas = (fuerzas_nodos * coord_x).sum()   # Σ F·x
vector_cargas = np.array([fuerza_total, momento_x_cargas, momento_y_cargas])

# ---------------------------------------------------------------------------
# 4. Matriz de coeficientes: una columna por apoyo (incógnita R_k)
#    fila 0: coeficientes de ΣFz | fila 1: brazos en Y | fila 2: brazos en X
# ---------------------------------------------------------------------------
matriz_equilibrio = np.array([
    [1.0] * len(apoyos),
    [coord_y[a] for a in apoyos],
    [coord_x[a] for a in apoyos],
])

# ---------------------------------------------------------------------------
# 5. Resolver  A · R = b  (reacciones en los apoyos)
# ---------------------------------------------------------------------------
reacciones_apoyos = np.linalg.solve(matriz_equilibrio, vector_cargas)

# Matriz 3x3 de reacciones por nodo (0 en los nodos sin apoyo)
reacciones_nodos = np.zeros_like(fuerzas_nodos)
for reaccion, apoyo in zip(reacciones_apoyos, apoyos):
    reacciones_nodos[apoyo] = reaccion

# ---------------------------------------------------------------------------
# 6. Verificación del equilibrio: las cargas y las reacciones deben anularse
# ---------------------------------------------------------------------------
residuo_fuerza = reacciones_nodos.sum() - fuerza_total
residuo_mx = (reacciones_nodos * coord_y).sum() - momento_x_cargas
residuo_my = (reacciones_nodos * coord_x).sum() - momento_y_cargas

# ---------------------------------------------------------------------------
# 7. Resultados
# ---------------------------------------------------------------------------
print("\n[Resultados del Análisis Estructural]")
print(f"-> Fuerza total aplicada: {fuerza_total:.2f} N")
print(f"-> Sumatoria de fuerzas por columna (eje X): {fuerzas_nodos.sum(axis=0)}")
print(f"-> Carga acumulada por fila (eje Y):        {fuerzas_nodos.sum(axis=1)}")
print("\nReacciones en los apoyos (N):")
for reaccion, (i, j) in zip(reacciones_apoyos, apoyos):
    sentido = "hacia arriba (compresión)" if reaccion >= 0 else "hacia abajo (tracción)"
    print(f"   Nodo ({i},{j}) en x={coord_x[i, j]:.0f} m, y={coord_y[i, j]:.0f} m -> {reaccion:8.2f} N  {sentido}")
print("\nMatriz de reacciones por nodo (N):\n", np.round(reacciones_nodos, 2))
print("\nVerificación de equilibrio (deben ser ≈ 0):")
print(f"   ΣFz = {residuo_fuerza:.2e} | ΣMx = {residuo_mx:.2e} | ΣMy = {residuo_my:.2e}")
print("-> Equilibrio estático comprobado." if np.allclose([residuo_fuerza, residuo_mx, residuo_my], 0)
      else "-> ERROR: no se cumple el equilibrio.")
