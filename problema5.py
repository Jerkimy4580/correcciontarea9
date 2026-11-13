"""
=============================================================================
Problema 5: Transformación de Coordenadas en un Sistema Cartesiano
=============================================================================
Descripción:
    Los puntos se guardan en una matriz N x 2 (cada fila = un punto [x, y]).
    Se aplican transformaciones lineales con matrices:
      A) Rotación de 90° respecto al origen (matriz 2x2).
      B) Transformación compuesta escalado + rotación + traslación, usando
         coordenadas homogéneas (matriz 3x3), porque la traslación NO se puede
         expresar con una matriz 2x2.
    Se grafican los puntos antes y después de cada transformación.
=============================================================================
"""

import os

import matplotlib.pyplot as plt
import numpy as np

print("=" * 65)
print("PROBLEMA 5: TRANSFORMACIÓN DE COORDENADAS CARTESIANAS")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. Puntos originales (vértices de un triángulo), matriz N x 2
# ---------------------------------------------------------------------------
puntos = np.array([
    [2.0, 1.0],
    [4.0, 1.0],
    [3.0, 5.0],
])
print("Coordenadas originales [x, y]:\n", puntos)

# ---------------------------------------------------------------------------
# 2. Transformación A: rotación de 90° respecto al origen
# ---------------------------------------------------------------------------
ANGULO_GRADOS = 90.0
theta = np.radians(ANGULO_GRADOS)
matriz_rotacion = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)],
])

# Los puntos están en FILAS, por eso se multiplica por la transpuesta:
# (R · p)ᵀ = pᵀ · Rᵀ. Aplica la rotación a los N puntos con UN solo producto matricial.
puntos_rotados = puntos @ matriz_rotacion.T

print(f"\nMatriz de rotación de {ANGULO_GRADOS:.0f}°:\n", np.round(matriz_rotacion, 2) + 0.0)
print("Puntos rotados:\n", np.round(puntos_rotados, 2) + 0.0)


# ---------------------------------------------------------------------------
# 3. Transformación B: escalado + rotación + traslación (coordenadas homogéneas)
# ---------------------------------------------------------------------------
def matriz_escalado(factor):
    return np.array([[factor, 0.0, 0.0], [0.0, factor, 0.0], [0.0, 0.0, 1.0]])


def matriz_rotacion_homogenea(angulo_rad):
    c, s = np.cos(angulo_rad), np.sin(angulo_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def matriz_traslacion(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


# Se componen aplicando primero escalado, luego rotación y al final traslación.
# Con vectores columna el orden se escribe de derecha a izquierda: T · R · S
matriz_compuesta = (matriz_traslacion(6.0, 2.0)
                    @ matriz_rotacion_homogenea(theta)
                    @ matriz_escalado(1.5))

# Se agrega una columna de unos (x, y, 1) para poder usar la matriz 3x3
puntos_homogeneos = np.hstack([puntos, np.ones((len(puntos), 1))])
puntos_compuestos = (puntos_homogeneos @ matriz_compuesta.T)[:, :2]

print("\nMatriz compuesta (traslación · rotación · escalado), 3x3:\n",
      np.round(matriz_compuesta, 2) + 0.0)
print("Puntos tras la transformación compuesta:\n", np.round(puntos_compuestos, 2) + 0.0)


# ---------------------------------------------------------------------------
# 4. Verificación: la rotación no deforma la figura (conserva las distancias)
# ---------------------------------------------------------------------------
def longitudes_lados(vertices):
    """Longitud de cada lado del triángulo (distancia entre vértices consecutivos)."""
    return np.linalg.norm(vertices - np.roll(vertices, -1, axis=0), axis=1)


print("\n[Verificación]")
print("Lados originales:", np.round(longitudes_lados(puntos), 3))
print("Lados rotados:   ", np.round(longitudes_lados(puntos_rotados), 3))
print("-> La rotación conserva las distancias." if
      np.allclose(longitudes_lados(puntos), longitudes_lados(puntos_rotados))
      else "-> ERROR: la rotación deformó la figura.")
print("-> El escalado 1.5 multiplica los lados por 1.5:",
      np.allclose(longitudes_lados(puntos_compuestos), 1.5 * longitudes_lados(puntos)))


# ---------------------------------------------------------------------------
# 5. Visualización antes y después
# ---------------------------------------------------------------------------
def dibujar(eje, original, transformado, titulo):
    """Dibuja ambos triángulos (cerrados) con sus vértices numerados."""
    for datos, color, etiqueta in [(original, "tab:blue", "Original"),
                                   (transformado, "tab:red", "Transformado")]:
        cerrado = np.vstack([datos, datos[0]])     # repite el 1er punto para cerrar
        eje.plot(cerrado[:, 0], cerrado[:, 1], "-o", color=color, label=etiqueta)
        for k, (x, y) in enumerate(datos, start=1):
            eje.annotate(f"P{k}", (x, y), textcoords="offset points", xytext=(6, 6))
    eje.axhline(0, color="black", linewidth=0.8)
    eje.axvline(0, color="black", linewidth=0.8)
    eje.set_aspect("equal")
    eje.grid(alpha=0.3)
    eje.set_title(titulo)
    eje.legend()


fig, (eje_a, eje_b) = plt.subplots(1, 2, figsize=(13, 6))
dibujar(eje_a, puntos, puntos_rotados, f"A) Rotación de {ANGULO_GRADOS:.0f}° respecto al origen")
dibujar(eje_b, puntos, puntos_compuestos, "B) Escalado 1.5 + rotación 90° + traslación (6, 2)")
fig.suptitle("Problema 5: puntos antes y después de la transformación")
fig.tight_layout()
ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultado_problema5.png")
fig.savefig(ruta, dpi=120)
print(f"-> Imagen guardada en: {ruta}")
plt.show()
