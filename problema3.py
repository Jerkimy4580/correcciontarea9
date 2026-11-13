"""
=============================================================================
Problema 3: Análisis de Imágenes Médicas en 3D
=============================================================================
Descripción:
    Un volumen médico es un tensor 3D de forma (capas, alto, ancho). Se genera
    un volumen sintético (5 capas de 32x32 píxeles) con un "órgano" circular,
    se le agrega ruido y se aplica un filtro de promedio 3x3 a cada capa.
    Se muestra la mejora de forma numérica (error cuadrático medio) y visual.
=============================================================================
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

print("=" * 65)
print("PROBLEMA 3: ANÁLISIS DE IMÁGENES MÉDICAS EN 3D (SUAVIZADO)")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. Volumen sintético: imagen limpia + ruido
# ---------------------------------------------------------------------------
CAPAS, ALTO, ANCHO = 5, 32, 32
rng = np.random.default_rng(10)

# Imagen "ideal": fondo oscuro (60) y un órgano circular brillante (160)
# cuyo radio cambia con la profundidad (más grande en la capa central).
yy, xx = np.mgrid[0:ALTO, 0:ANCHO]
volumen_limpio = np.full((CAPAS, ALTO, ANCHO), 60.0)
for z in range(CAPAS):
    radio = 5 + 3 * (2 - abs(z - 2))                       # 5, 8, 11, 8, 5
    mascara = (yy - 16) ** 2 + (xx - 16) ** 2 <= radio ** 2
    volumen_limpio[z][mascara] = 160.0

# Ruido gaussiano en todo el volumen + 30 píxeles "quemados" (255) de ruido extremo
volumen_ruidoso = volumen_limpio + rng.normal(0, 20, volumen_limpio.shape)
indices_ruido = tuple(rng.integers(0, s, 30) for s in volumen_limpio.shape)
volumen_ruidoso[indices_ruido] = 255.0
volumen_ruidoso = np.clip(volumen_ruidoso, 0, 255)


# ---------------------------------------------------------------------------
# 2. Filtro de promedio 3x3: versión vectorizada (la usada en la solución)
# ---------------------------------------------------------------------------
def suavizar_vectorizado(volumen):
    """Promedio 3x3 de cada píxel de TODAS las capas a la vez.

    - np.pad(mode='edge') replica los bordes para que también se filtren.
    - sliding_window_view crea una "vista" de ventanas 3x3 sin copiar datos
      (no consume memoria extra): forma (capas, alto, ancho, 3, 3).
    - mean(axis=(3, 4)) promedia cada ventana.
    """
    relleno = np.pad(volumen, ((0, 0), (1, 1), (1, 1)), mode="edge")
    ventanas = sliding_window_view(relleno, (3, 3), axis=(1, 2))
    return ventanas.mean(axis=(3, 4))


# Versión con ciclos: SOLO para comparar el tiempo de ejecución
def suavizar_con_ciclos(volumen):
    relleno = np.pad(volumen, ((0, 0), (1, 1), (1, 1)), mode="edge")
    resultado = np.empty_like(volumen)
    for z in range(volumen.shape[0]):
        for i in range(volumen.shape[1]):
            for j in range(volumen.shape[2]):
                resultado[z, i, j] = relleno[z, i:i + 3, j:j + 3].mean()
    return resultado


t0 = time.perf_counter()
volumen_suavizado = suavizar_vectorizado(volumen_ruidoso)
t_vec = time.perf_counter() - t0

t0 = time.perf_counter()
volumen_ciclos = suavizar_con_ciclos(volumen_ruidoso)
t_ciclos = time.perf_counter() - t0

# Ambas versiones deben dar exactamente el mismo resultado
assert np.allclose(volumen_suavizado, volumen_ciclos)

# ---------------------------------------------------------------------------
# 3. Medición de la mejora: error cuadrático medio respecto a la imagen limpia
# ---------------------------------------------------------------------------
mse_antes = np.mean((volumen_ruidoso - volumen_limpio) ** 2)
mse_despues = np.mean((volumen_suavizado - volumen_limpio) ** 2)

print(f"Volumen: {CAPAS} capas de {ALTO}x{ANCHO} píxeles (tensor 3D {volumen_ruidoso.shape})")
print("\n[Resultados del Procesamiento de Imagen Médica]")
print(f"-> Error cuadrático medio ANTES del filtro:   {mse_antes:8.2f}")
print(f"-> Error cuadrático medio DESPUÉS del filtro: {mse_despues:8.2f}")
print(f"-> Reducción del error: {100 * (1 - mse_despues / mse_antes):.1f} %")
print(f"-> Tiempo vectorizado: {t_vec * 1000:.2f} ms | con ciclos: {t_ciclos * 1000:.2f} ms "
      f"(≈ {t_ciclos / t_vec:.0f}x más lento)")

# ---------------------------------------------------------------------------
# 4. Visualización: capa central antes y después
# ---------------------------------------------------------------------------
CAPA = 2
fig, ejes = plt.subplots(1, 3, figsize=(13, 4.5))
imagenes = [(volumen_limpio[CAPA], "Imagen ideal (sin ruido)"),
            (volumen_ruidoso[CAPA], f"Con ruido (MSE = {mse_antes:.0f})"),
            (volumen_suavizado[CAPA], f"Suavizada 3x3 (MSE = {mse_despues:.0f})")]
for eje, (img, titulo) in zip(ejes, imagenes):
    eje.imshow(img, cmap="gray", vmin=0, vmax=255)
    eje.set_title(titulo)
    eje.axis("off")
fig.suptitle(f"Problema 3: filtro de promedio sobre la capa {CAPA} del volumen")
fig.tight_layout()
ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultado_problema3.png")
fig.savefig(ruta, dpi=120)
print(f"-> Imagen guardada en: {ruta}")
plt.show()
