"""
=============================================================================
Problema 4: Manejo de Datos en una Tabla de Sensores
=============================================================================
Descripción:
    Tabla de temperaturas en una matriz 5x5:
        filas    (eje 0) = 5 momentos en el tiempo (T1..T5)
        columnas (eje 1) = 5 sensores (S1..S5)
    Se calculan promedio y desviación estándar por sensor (columnas) y por
    momento (filas) y se visualizan los resultados.
=============================================================================
"""

import os

import matplotlib.pyplot as plt
import numpy as np

print("=" * 65)
print("PROBLEMA 4: MANEJO DE DATOS EN UNA TABLA DE SENSORES")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. Datos (temperaturas simuladas en °C, reproducibles con semilla)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(5)
temperaturas = np.round(rng.uniform(18.5, 32.0, size=(5, 5)), 2)
etiquetas_sensores = [f"S{i}" for i in range(1, 6)]
etiquetas_tiempo = [f"T{i}" for i in range(1, 6)]

print("Tabla de temperaturas (°C)  [filas: tiempo | columnas: sensores]\n",
      temperaturas)

# ---------------------------------------------------------------------------
# 2. Estadísticas por eje (una sola llamada calcula todos los valores)
#    axis=0 -> colapsa las filas    -> un valor POR SENSOR (columna)
#    axis=1 -> colapsa las columnas -> un valor POR MOMENTO (fila)
# ---------------------------------------------------------------------------
promedio_sensor = temperaturas.mean(axis=0)
desviacion_sensor = temperaturas.std(axis=0)       # desviación poblacional (ddof=0)
promedio_tiempo = temperaturas.mean(axis=1)
desviacion_tiempo = temperaturas.std(axis=1)

print("\n[Resultados por sensor (columnas)]")
for nombre, m, s in zip(etiquetas_sensores, promedio_sensor, desviacion_sensor):
    print(f"   {nombre}: promedio = {m:6.2f} °C | desviación estándar = {s:5.2f} °C")
print("\n[Resultados por momento temporal (filas)]")
for nombre, m, s in zip(etiquetas_tiempo, promedio_tiempo, desviacion_tiempo):
    print(f"   {nombre}: promedio = {m:6.2f} °C | desviación estándar = {s:5.2f} °C")

sensor_mas_estable = etiquetas_sensores[int(np.argmin(desviacion_sensor))]
print(f"\n-> Sensor más estable (menor desviación): {sensor_mas_estable}")

# ---------------------------------------------------------------------------
# 3. Visualización
# ---------------------------------------------------------------------------
fig, (eje_mapa, eje_sensor, eje_tiempo) = plt.subplots(1, 3, figsize=(17, 4.8))

# (a) Mapa de calor de la tabla completa, con el valor escrito en cada celda
imagen = eje_mapa.imshow(temperaturas, cmap="YlOrRd")
eje_mapa.set_xticks(range(5), etiquetas_sensores)
eje_mapa.set_yticks(range(5), etiquetas_tiempo)
for i in range(5):
    for j in range(5):
        eje_mapa.text(j, i, f"{temperaturas[i, j]:.1f}", ha="center", va="center", fontsize=9)
eje_mapa.set_title("Tabla de datos (°C)")
fig.colorbar(imagen, ax=eje_mapa, label="°C")

# (b) Promedio por sensor con barra de error = desviación estándar
eje_sensor.bar(etiquetas_sensores, promedio_sensor, yerr=desviacion_sensor,
               capsize=5, color="tab:blue")
eje_sensor.set_title("Promedio ± desviación por sensor")
eje_sensor.set_ylabel("Temperatura (°C)")

# (c) Promedio por momento con barra de error = desviación estándar
eje_tiempo.errorbar(etiquetas_tiempo, promedio_tiempo, yerr=desviacion_tiempo,
                    fmt="-o", capsize=5, color="tab:red")
eje_tiempo.set_title("Promedio ± desviación por momento")
eje_tiempo.set_ylabel("Temperatura (°C)")
eje_tiempo.grid(alpha=0.3)

fig.suptitle("Problema 4: análisis de la tabla de sensores 5x5")
fig.tight_layout()
ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultado_problema4.png")
fig.savefig(ruta, dpi=120)
print(f"-> Imagen guardada en: {ruta}")
plt.show()
