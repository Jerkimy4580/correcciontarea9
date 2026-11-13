"""
=============================================================================
Problema 2: Simulación de Fluido en una Cuadrícula Tridimensional (3x3x3)
=============================================================================
Descripción:
    Arreglo 4D de forma (3, 3, 3, 3):
        ejes 0,1,2 -> posición (z, y, x) de la celda en el volumen 3x3x3
        eje 3      -> 3 propiedades físicas de cada celda:
                      índice 0 = Presión (kPa)
                      índice 1 = Temperatura (K)
                      índice 2 = Velocidad (m/s)

    Se simula la propagación de una onda de presión: en cada paso de tiempo
    se actualiza TODAS las celdas según el promedio de sus 6 celdas vecinas
    (arriba, abajo, izquierda, derecha, delante, detrás).
=============================================================================
"""

import numpy as np

print("=" * 65)
print("PROBLEMA 2: SIMULACIÓN DE FLUIDO EN UNA CUADRÍCULA 3D")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. Estado inicial del fluido
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)   # generador con semilla: resultados reproducibles
volumen_fluido = np.zeros((3, 3, 3, 3))
volumen_fluido[..., 0] = rng.uniform(100.0, 105.0, (3, 3, 3))  # Presión (kPa)
volumen_fluido[..., 1] = rng.uniform(290.0, 310.0, (3, 3, 3))  # Temperatura (K)
volumen_fluido[..., 2] = rng.uniform(0.0, 5.0, (3, 3, 3))      # Velocidad (m/s)

# Perturbación: un pulso de sobrepresión en la celda central (la "fuente" de la onda)
volumen_fluido[1, 1, 1, 0] += 20.0

# Coeficiente de mezcla con los vecinos por propiedad (entre 0 y 1).
# Se aplica por broadcasting sobre el último eje (3 propiedades).
#   presión: 0.5 (la onda se propaga rápido) | temperatura: 0.2 | velocidad: 0.3
COEF_MEZCLA = np.array([0.5, 0.2, 0.3])
PASOS_TIEMPO = 4


# ---------------------------------------------------------------------------
# 2. Paso de simulación (vectorizado, sin ciclos sobre las celdas)
# ---------------------------------------------------------------------------
def promedio_vecinos(volumen):
    """Promedio de las 6 celdas vecinas de TODAS las celdas a la vez.

    np.pad con mode='edge' replica la celda del borde: las celdas de la frontera
    (que tienen menos de 6 vecinos reales) usan su propio valor como vecino
    faltante, equivalente a una pared sin flujo.
    """
    p = np.pad(volumen, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="edge")
    suma = (p[:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1] +     # vecinos en z (abajo/arriba)
            p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1] +     # vecinos en y
            p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:])      # vecinos en x
    return suma / 6.0


def paso_simulacion(volumen):
    """Nuevo estado = (1 - k)·estado propio + k·promedio de vecinas, por propiedad."""
    return (1.0 - COEF_MEZCLA) * volumen + COEF_MEZCLA * promedio_vecinos(volumen)


# ---------------------------------------------------------------------------
# 3. Ejecución de la simulación
# ---------------------------------------------------------------------------
print("Presión (kPa) en la capa central (z=1) -- estado inicial:\n",
      np.round(volumen_fluido[1, :, :, 0], 2))

estado = volumen_fluido.copy()
presion_centro = [estado[1, 1, 1, 0]]
presion_esquina = [estado[0, 0, 0, 0]]
for _ in range(PASOS_TIEMPO):
    estado = paso_simulacion(estado)       # se actualizan las 27 celdas
    presion_centro.append(estado[1, 1, 1, 0])
    presion_esquina.append(estado[0, 0, 0, 0])

print(f"\nPresión (kPa) en la capa central (z=1) tras {PASOS_TIEMPO} pasos:\n",
      np.round(estado[1, :, :, 0], 2))

print("\n[Evolución de la onda de presión]")
print("Paso | Presión centro (fuente) | Presión esquina (0,0,0)")
for t, (pc, pe) in enumerate(zip(presion_centro, presion_esquina)):
    print(f"  {t}  |        {pc:7.2f}          |        {pe:7.2f}")
print("-> El pulso del centro se atenúa y llega a las celdas lejanas (la esquina "
      "sube su presión):\n   la información se propaga celda a celda por vecindad.")
