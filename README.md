# Aplicaciones Prácticas de Arreglos Multidimensionales en Ingeniería

Repositorio académico con cinco soluciones en **Python** a problemas de ingeniería, resueltas con arreglos multidimensionales (matrices y tensores) usando **NumPy**, y con visualizaciones en **Matplotlib**.

Cada archivo es independiente: se puede ejecutar solo, en cualquier orden.

---

## 1. Requisitos del entorno

| Herramienta | Requisito |
|---|---|
| Lenguaje | Python 3.8 o superior |
| Cálculo con arreglos | NumPy (>= 1.20, necesario para `sliding_window_view`) |
| Gráficos | Matplotlib (>= 3.3) |
| Terminal | CMD, PowerShell o terminal integrada de VS Code |

El archivo `requirements.txt` contiene las dos dependencias:

```text
numpy>=1.20.0
matplotlib>=3.3.0
```

Instalación:

```bash
python -m pip install -r requirements.txt
```

---

## 2. Estructura del repositorio

```text
├── problema1.py                 # Equilibrio estático de una estructura 3x3
├── problema2.py                 # Propagación de presión en fluido 3x3x3x3
├── problema3.py                 # Suavizado de volumen médico 5x32x32
├── problema4.py                 # Tabla de sensores 5x5 + gráficos
├── problema5.py                 # Transformaciones de puntos + gráficos
├── resultado_problema3.png      # Gráfico generado por problema3.py
├── resultado_problema4.png      # Gráfico generado por problema4.py
├── resultado_problema5.png      # Gráfico generado por problema5.py
├── requirements.txt
└── README.md
```

| Archivo | Problema | Arreglo utilizado | Forma (`shape`) |
|---|---|---|---|
| `problema1.py` | Fuerzas en una estructura | Matriz 2D de fuerzas + sistema de ecuaciones 3x3 | `(3, 3)` |
| `problema2.py` | Fluido en cuadrícula 3D | Arreglo 4D: 3 ejes espaciales + 1 eje de propiedades | `(3, 3, 3, 3)` |
| `problema3.py` | Imágenes médicas 3D | Tensor 3D (capas, alto, ancho) | `(5, 32, 32)` |
| `problema4.py` | Tabla de sensores | Matriz 2D (tiempo x sensor) | `(5, 5)` |
| `problema5.py` | Transformación de coordenadas | Matriz N x 2 de puntos, matrices 2x2 y 3x3 | `(3, 2)`, `(2, 2)`, `(3, 3)` |

> **Nota sobre el problema 2:** el arreglo es `(3, 3, 3, 3)`: un volumen de 3x3x3 celdas y **3** propiedades por celda (presión, temperatura y velocidad), tal como pide el enunciado.

---

## 3. Descripción de cada solución, enfoque y eficiencia

### Problema 1: Análisis de fuerzas en una estructura

**Qué se hace.** La matriz 3x3 representa la fuerza vertical (N) aplicada en cada nodo. La estructura se apoya en 3 nodos (no alineados), y se calculan las **reacciones en cada apoyo** con las leyes de la estática:

- ΣFz = 0 (equilibrio de fuerzas)
- ΣMx = 0 y ΣMy = 0 (equilibrio de momentos)

Esto da un sistema lineal de 3 ecuaciones con 3 incógnitas (las reacciones), que se resuelve con `np.linalg.solve`. Al final se **verifica** que las cargas y las reacciones se anulen (residuos ≈ 0).

**Por qué este enfoque.** La versión anterior solo calculaba `-suma(fuerzas)`, que es la reacción global, no las reacciones por nodo. Para repartir la carga entre los apoyos hay que considerar también la **posición** de cada fuerza (momentos), por eso se construyen las matrices de coordenadas con `np.meshgrid`.

**Cómo influye en la eficiencia.**
- Las sumas ponderadas (`Σ F·x`, `Σ F·y`) se hacen con una operación sobre toda la matriz (`(fuerzas * coord_x).sum()`), sin ciclos.
- Resolver el sistema con `np.linalg.solve` usa álgebra lineal optimizada (LAPACK) y es más preciso y rápido que invertir la matriz a mano.
- El mismo código sirve para cualquier tamaño de malla y cualquier posición de apoyos, cambiando solo los datos.

### Problema 2: Simulación de fluido en una cuadrícula 3D

**Qué se hace.** El arreglo `(3, 3, 3, 3)` guarda presión, temperatura y velocidad de cada celda. Se crea un pulso de sobrepresión en la celda central y se simulan 4 pasos de tiempo. En cada paso **se actualizan las 27 celdas**: cada una mezcla su valor con el promedio de sus 6 vecinas (arriba, abajo, izquierda, derecha, delante, detrás):

`nuevo = (1 - k) · propio + k · promedio_vecinas`

El coeficiente `k` es distinto por propiedad (presión 0.5, temperatura 0.2, velocidad 0.3), porque la presión se propaga más rápido que las otras.

**Por qué este enfoque.** La versión anterior actualizaba solo la celda central usando el promedio de **todo** el volumen, no de sus vecinas. Ahora la función `promedio_vecinos` obtiene los vecinos de todas las celdas a la vez con **cortes desplazados** del arreglo (`p[:-2]`, `p[2:]`, etc.). En los bordes se usa `np.pad(mode="edge")`, que equivale a una pared sin flujo.

**Cómo influye en la eficiencia.**
- Se evitan 3 ciclos anidados (z, y, x) y las condiciones para no salirse del arreglo: la actualización completa son unas pocas operaciones vectorizadas.
- Usar `k` como arreglo de 3 elementos y aplicarlo por *broadcasting* actualiza las 3 propiedades a la vez, sin repetir código.
- Crece bien: la misma función funciona con una malla de 100x100x100 sin cambiarla.

### Problema 3: Análisis de imágenes médicas en 3D

**Qué se hace.** Se genera un volumen sintético de 5 capas de 32x32 píxeles con un "órgano" circular, se le agrega ruido (gaussiano y píxeles extremos de 255) y se aplica un **filtro de promedio 3x3** a cada capa. La mejora se muestra de dos formas:

1. **Numérica:** error cuadrático medio (MSE) contra la imagen limpia, antes y después (se reduce alrededor de 69 %).
2. **Visual:** `resultado_problema3.png` con la capa central ideal, ruidosa y suavizada.

**Por qué este enfoque.** Con imágenes de 4x4 (versión anterior) casi no hay estructura que preservar ni mejora que mostrar. Al tener una imagen "ideal" conocida se puede **medir** si el filtro realmente mejora.

**Cómo influye en la eficiencia.**
- `sliding_window_view` crea una *vista* de todas las ventanas 3x3 **sin copiar datos**, así que no gasta memoria extra, y `mean(axis=(3, 4))` procesa todas las capas a la vez.
- El script incluye también una versión con 3 ciclos anidados y compara los tiempos: la vectorizada es unas **30 veces más rápida** en este tamaño, y la diferencia crece con volúmenes grandes. El `assert np.allclose(...)` comprueba que ambas dan exactamente el mismo resultado.
- Los bordes también se filtran (con `np.pad`), en vez de quedar sin procesar.

### Problema 4: Manejo de datos en una tabla de sensores

**Qué se hace.** Matriz 5x5: filas = momentos (T1 a T5), columnas = sensores (S1 a S5). Se calcula el promedio y la desviación estándar por sensor (`axis=0`) y por momento (`axis=1`), y se genera `resultado_problema4.png` con 3 gráficos:

1. Mapa de calor de la tabla completa.
2. Promedio por sensor con barras de error (desviación).
3. Promedio por momento con barras de error (desviación).

**Por qué este enfoque.** El eje elige la dirección del cálculo: `axis=0` colapsa las filas (un resultado por columna) y `axis=1` colapsa las columnas (un resultado por fila). La visualización que faltaba permite ver de un vistazo qué sensor es más estable (menor desviación).

**Cómo influye en la eficiencia.**
- Una sola llamada (`mean(axis=0)`) reemplaza un ciclo por columna con acumuladores; el código es más corto, tiene menos errores y se ejecuta en código compilado de NumPy.
- Los resultados son arreglos de tamaño 5, así que se pasan directo a Matplotlib sin conversiones.

### Problema 5: Transformación de coordenadas cartesianas

**Qué se hace.** Los puntos están en una matriz N x 2. Se aplican dos transformaciones y se grafican antes y después en `resultado_problema5.png`:

- **A) Rotación de 90°** con una matriz 2x2.
- **B) Escalado 1.5 + rotación 90° + traslación (6, 2)** con una única matriz 3x3 en **coordenadas homogéneas**.

**Por qué este enfoque.** La traslación no se puede escribir como matriz 2x2, por eso se agrega una columna de unos a los puntos `(x, y, 1)` y se usan matrices 3x3. Así rotación, escalado y traslación se **combinan en una sola matriz** multiplicándolas (`T @ R @ S`, de derecha a izquierda). Además, el script verifica que la rotación conserva las distancias entre vértices y que el escalado las multiplica por 1.5.

**Cómo influye en la eficiencia.**
- `puntos @ matriz.T` transforma los N puntos con **un solo producto matricial**; con 3 o con 3 millones de puntos el código es el mismo.
- Combinar varias transformaciones en una sola matriz reduce el trabajo: se calcula la matriz compuesta una vez (3x3) y se aplica una sola vez a todos los puntos, en lugar de recorrer los puntos una vez por cada transformación.

---

## 4. Enfoque metodológico general

| Técnica | Dónde se usa | Beneficio |
|---|---|---|
| **Vectorización** (operar sobre arreglos completos) | Todos | Menos código, sin ciclos de Python, ejecución en C |
| **Reducción por ejes** (`axis=`) | P1, P4 | Un cálculo por fila o columna en una sola llamada |
| **Broadcasting** | P2 | Operar arreglos de distinta forma sin repetir datos |
| **Vistas sin copia** (`sliding_window_view`, cortes) | P2, P3 | Ahorro de memoria |
| **Álgebra lineal optimizada** (`solve`, `@`) | P1, P5 | Rapidez y precisión numérica |
| **Verificación de resultados** | P1, P3, P5 | Comprobar que la solución es correcta, no solo que corre |

---

## 5. Instalación y ejecución

```bash
git clone https://github.com/Jerkimy4580/correcciontarea9.git
cd correcciontarea9
python -m pip install -r requirements.txt

python problema1.py
python problema2.py
python problema3.py
python problema4.py
python problema5.py
```

Los problemas 3, 4 y 5 abren una ventana con el gráfico y además lo guardan como imagen `.png` en la misma carpeta.

---

## 6. Tecnologías utilizadas

- **Python:** lenguaje de las soluciones.
- **NumPy:** arreglos multidimensionales, álgebra lineal y estadística.
- **Matplotlib:** visualización de resultados.
- **Git y GitHub:** control de versiones y entrega.
