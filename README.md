# Modelo Predictivo de la Crocancia de Rodajas de Papa

Proyecto académico de la asignatura **Modelación** (Reto 1) de la Facultad de Ingeniería de la **Universidad Católica Luis Amigó** (Medellín, Colombia). Aplicación de escritorio en Python que modela un fenómeno físico real: la **crocancia de rodajas de papa** (pico de amplitud sonora en decibelios, dB) en función de la **temperatura de horneado** (°C). El proyecto integra recolección de datos experimentales, análisis de datos, implementación de **8 modelos de regresión y Machine Learning** (incluida una red neuronal desarrollada desde cero), evaluación con métricas de error y generación automatizada de reportes.

---

## Tabla de contenido

- [Resumen del proyecto](#resumen-del-proyecto)
- [Contexto académico](#contexto-académico)
- [Competencias demostradas](#competencias-demostradas)
- [Tecnologías y librerías](#tecnologías-y-librerías)
- [Implementación: modelos](#implementación-modelos)
- [Conjunto de datos](#conjunto-de-datos)
- [Metodología experimental](#metodología-experimental)
- [Evaluación de los modelos](#evaluación-de-los-modelos)
- [Entregables](#entregables)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación y uso](#instalación-y-uso)
- [Autores](#autores)
- [Referencias externas](#referencias-externas)

## Resumen del proyecto

El reto consistió en responder una pregunta práctica: **¿la crocancia de unas rodajas de papa horneadas depende de la temperatura de cocción y se puede modelar con datos?** Para ello se diseñó un experimento con una freidora de aire y el micrófono de un teléfono móvil, se registraron **150 mediciones** (temperatura vs. decibelios) y se construyó una aplicación de escritorio que permite:

- Cargar la base de datos experimental desde un archivo CSV.
- Seleccionar y entrenar **8 modelos** de regresión y Machine Learning.
- Calcular y comparar la calidad de cada modelo mediante **MSE** y **RMSE**.
- Visualizar el ajuste del modelo sobre los datos reales y una **comparativa global** de todos los modelos.
- Exportar un **reporte PDF** con la ecuación/coeficientes, el RMSE y las gráficas.

Se implementaron tanto modelos clásicos (regresión lineal y polinomial de forma cerrada con NumPy) como modelos de Machine Learning (KNN, SVR, árbol de decisión y random forest con scikit-learn) y una **red neuronal con activación sigmoide y descenso de gradiente desarrollada desde cero**, con preprocesado de características polinomiales y escalado.

## Contexto académico

- Institución: Universidad Católica Luis Amigó, Facultad de Ingeniería.
- Asignatura: Modelación.
- Tipo de trabajo: Reto 1 (proyecto integrador de la asignatura).
- Participación: desarrollo integral del proyecto (diseño del experimento, recolección de datos, modelación, aplicación y documentación).
- Fecha: marzo de 2026.

## Competencias demostradas

- Programación en Python para análisis de datos (NumPy, Pandas).
- Modelos de regresión y Machine Learning supervisado (scikit-learn).
- Redes neuronales implementadas desde cero: normalización de datos, función de activación sigmoide y descenso de gradiente con NumPy.
- Diseño de interfaces gráficas de escritorio (Tkinter) con gráficas embebidas (Matplotlib).
- Generación automatizada de reportes PDF (ReportLab).
- Tratamiento de datos experimentales y evaluación comparativa de modelos con MSE y RMSE.

## Tecnologías y librerías

| Tecnología / librería | Uso |
| --------------------- | --- |
| Python 3 | Lenguaje principal del proyecto |
| NumPy | Cálculo numérico y modelos clásicos de forma cerrada |
| Pandas | Lectura y procesamiento del CSV de datos |
| Matplotlib | Generación de gráficas y visualización |
| scikit-learn | Modelos ML (KNN, SVR, árbol, random forest) y preprocesado (PolynomialFeatures, StandardScaler) |
| Tkinter | Interfaz gráfica de escritorio |
| ReportLab | Generación del reporte PDF |
| Pillow | Carga del logo institucional en la interfaz |
| PyTorch | Importado en el script (actualmente sin uso funcional; requiere instalación para que la aplicación arranque) |

## Implementación: modelos

La aplicación permite seleccionar entre 8 modelos desde la interfaz, con el grado del polinomio ajustable (1-5) para los modelos polinómicos:

| # | Modelo | Implementación | Detalle / hiperparámetros |
| - | ------ | -------------- | ------------------------- |
| 1 | Varianza | Regresión lineal de forma cerrada: m = Cov(X,Y)/Var(X), b = media(Y) - m * media(X) | - |
| 2 | Mínimos cuadrados | Regresión polinomial con NumPy (np.polyfit + np.poly1d) | Grado 1-5 (selector) |
| 3 | Polyfit | Regresión polinomial equivalente a mínimos cuadrados (np.polyfit) | Grado 1-5 (selector) |
| 4 | KNN | KNeighborsRegressor sobre características polinomiales escaladas | k = 3 |
| 5 | SVR | SVR con kernel RBF sobre características polinomiales escaladas | C = 1e8, gamma = 0.01, epsilon = 1e-7 |
| 6 | Árbol | DecisionTreeRegressor sobre características polinomiales escaladas | Profundidad máxima = 5 |
| 7 | Random Forest | RandomForestRegressor sobre características polinomiales escaladas | 100 árboles |
| 8 | Red Neuronal | Red neuronal implementada desde cero con NumPy: entrada polinomial normalizada (x, x², x³), activación sigmoide y descenso de gradiente | 5000 épocas, tasa de aprendizaje = 0.1 |

Nota: los modelos de Machine Learning (KNN, SVR, árbol y random forest) se entrenan con características polinomiales de grado 3 normalizadas con StandardScaler. La red neuronal se entrena manualmente con NumPy.

## Conjunto de datos

El archivo `datosmodelacion.csv` contiene **150 registros** medidos experimentalmente:

| Columna | Descripción |
| ------- | ----------- |
| x_C | Temperatura de horneado en grados Celsius (°C). Rango aproximado: 80-185 °C |
| y_dB | Crocancia de la rodaja de papa: pico máximo de amplitud sonora en decibelios (dB) |

## Metodología experimental

- Rodajas de papa de variedad Kennebec o gallega, de aproximadamente 2 mm de grosor y con un poco de aceite vegetal.
- Cocción en freidora de aire precalentada durante unos 15 minutos.
- Medición de la crocancia rompiendo la rodaja a máximo 2 cm del micrófono del teléfono; el pico de amplitud (dB) se registra como la crocancia.
- Aplicaciones de medición: NIOSH (iPhone) y Sound Meter (Android).

## Evaluación de los modelos

Para cada modelo ejecutado se calculan, con n = número de muestras:

- **MSE** (Error cuadrático medio): MSE = (1/n) · Σ (yᵢ − ŷᵢ)²
- **RMSE** (Raíz del error cuadrático medio): RMSE = √MSE

Estas métricas permiten comparar objetivamente los 8 modelos sobre el mismo conjunto de datos.

## Entregables

| Entregable | Descripción |
| ---------- | ----------- |
| Aplicación de escritorio | Interfaz Tkinter para cargar datos, entrenar modelos, comparar y exportar reportes |
| Gráficas | `grafica_modelo.png` (modelo vs. datos) y `comparacion_modelos.png` (comparativa global) |
| Reporte PDF | Modelo utilizado, ecuación/coeficientes, RMSE y gráficas (generado con ReportLab) |
| Dataset | 150 mediciones experimentales (temperatura vs. decibelios) |
| Informe académico | Documento completo del proyecto (disponible de forma externa, ver referencias) |

## Estructura del repositorio

| Archivo | Descripción |
| ------- | ----------- |
| `Main.py` | Aplicación de escritorio: interfaz gráfica, carga de datos, modelos y generación del reporte |
| `datosmodelacion.csv` | Base de datos del experimento (150 mediciones) |
| `logo.png` | Logo institucional mostrado en la interfaz |
| `README.md` | Documentación del proyecto (este archivo) |

## Instalación y uso

**Requisitos:** Python 3.x y las librerías indicadas en la sección de tecnologías.

```bash
git clone https://github.com/Daanikku/crispness-predictive-model.git
cd crispness-predictive-model
pip install numpy pandas matplotlib pillow scikit-learn reportlab torch
```

Ejecutar la aplicación:

```bash
python Main.py
```

Flujo de uso dentro de la aplicación:

1. Cargar el archivo CSV (`datosmodelacion.csv`) con el botón **Cargar CSV**.
2. Seleccionar el modelo y el grado del polinomio (1-5).
3. Ejecutar el modelo seleccionado, o ejecutar y comparar todos los modelos.
4. Generar el reporte PDF con el botón **Guardar PDF**.

## Autores

- Daniel Andres Cardona Muñoz
- Juan Jose Echeverri Quintero
- Tomas Saldarriaga Posso

Facultad de Ingeniería, Universidad Católica Luis Amigó, Medellín, Colombia.

## Referencias externas

- Drive del proyecto (base de datos, código e informe académico completo): [enlace](https://drive.google.com/drive/folders/1o_91SFqbsdi3_gpdzcGi4yVrULr2m7aQ?usp=drive_link)