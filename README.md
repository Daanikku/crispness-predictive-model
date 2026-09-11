# 🥔 crispness-predictive-model

**Modelo predictivo de la crocancia de rodajas de papa en función de la temperatura de horneado.**

Aplicación de escritorio desarrollada en **Python** que, mediante una interfaz gráfica (Tkinter), carga una base de datos experimental y entrena/visualiza **8 modelos de regresión y Machine Learning** para predecir la *crocancia* de una rodaja de papa —medida como el pico máximo de amplitud sonora en **decibelios (dB)** al romperla— a partir de la **temperatura de cocción en grados Celsius (°C)**.

Proyecto académico del **Reto 1** de la asignatura *Modelación* de la Facultad de Ingeniería de la **Universidad Católica Luis Amigó** (Medellín, Colombia, marzo 2026).

---

## 📑 Índice

- [Características](#-características)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Requisitos e instalación](#-requisitos-e-instalación)
- [Uso](#-uso)
- [Conjunto de datos](#-conjunto-de-datos)
- [Modelos implementados](#-modelos-implementados)
- [Métricas de error](#-métricas-de-error)
- [Reporte en PDF](#-reporte-en-pdf)
- [Metodología experimental](#-metodología-experimental)
- [Archivos generados](#-archivos-generados)
- [Autores](#-autores)
- [Anexos](#-anexos)

---

## ⭐ Características

- 🖥️ Interfaz gráfica amigable desarrollada con **Tkinter** e integración de gráficas **Matplotlib** dentro de la ventana.
- 📂 Carga de datos desde cualquier archivo **CSV** mediante un selector de archivos.
- 🧮 **8 modelos** de regresión y Machine Learning seleccionables desde un desplegable.
- 🎚️ Ajuste del **grado del polinomio (1–5)** para los modelos de regresión polinomial.
- 📊 Visualización del modelo ajustado superpuesto sobre los datos reales, y **comparación de todos los modelos** en una única gráfica.
- 📈 Cálculo del error con **MSE** y **RMSE** para cada modelo.
- 📄 Generación de un **reporte PDF** con la ecuación/coeficientes, el RMSE y las gráficas del modelo.

## 🗂️ Estructura del proyecto

| Archivo                  | Descripción                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `Main.py`                | Aplicación de escritorio: interfaz gráfica, carga de datos, modelos de regresión, red neuronal y generación del reporte PDF. |
| `datosmodelacion.csv`    | Base de datos del experimento: **150 mediciones** (temperatura en °C vs. crocancia en dB). |
| `logo.png`               | Logo institucional de la universidad mostrado en la interfaz.                |
| `Modelacion Reto 1.md`   | Informe académico completo: resumen, estrategia experimental, requisitos, variables de control, justificación y conclusiones. |
| `README.md`              | Este documento.                                                             |

## ⚙️ Requisitos e instalación

### Requisitos

- **Python 3.x**
- Bibliotecas:

| Biblioteca      | Uso                                            |
|-----------------|------------------------------------------------|
| `numpy`         | Operaciones numéricas y modelos explícitos.    |
| `pandas`        | Lectura y procesamiento del CSV.               |
| `matplotlib`    | Generación de gráficas.                        |
| `Pillow`        | Carga del logo institucional.                  |
| `scikit-learn`  | KNN, SVR, Árbol, Random Forest y preprocesado. |
| `reportlab`     | Generación del reporte PDF.                    |
| `torch`         | Importada en el script (sin uso funcional hoy, pero **requiere instalación** para que la app arranque). |
| `tkinter`       | Interfaz gráfica (incluida con Python).        |

### Instalación

```bash
git clone https://github.com/Daanikku/crispness-predictive-model.git
cd crispness-predictive-model
pip install numpy pandas matplotlib pillow scikit-learn reportlab torch
```

## 🚀 Uso

1. Ejecutar la aplicación desde la carpeta del proyecto:

   ```bash
   python Main.py
   ```

2. Pulsar el botón **«Cargar CSV»** y seleccionar `datosmodelacion.csv` (o cualquier CSV con dos columnas: `x_C` y `y_dB`).

3. Elegir el **modelo** en el desplegable y el **grado del polinomio** (1–5).

4. Usar los botones de la ventana:

   | Botón                          | Acción                                                                 |
   |--------------------------------|------------------------------------------------------------------------|
   | **Ejecutar Modelo**            | Entrena y grafica el modelo seleccionado, e imprime su RMSE.           |
   | **Ejecutar todos los modelos** | Entrena los 8 modelos y muestra la gráfica comparativa.                |
   | **Comparar todos los modelos** | Dibuja sobre los datos reales los modelos ya ejecutados.               |
   | **Guardar PDF**                | Genera un reporte PDF con el último modelo ejecutado.                  |

## 📊 Conjunto de datos

`datosmodelacion.csv` contiene **150 registros** recopilados experimentalmente por los integrantes del equipo:

| Columna | Descripción                                                                    |
|---------|--------------------------------------------------------------------------------|
| `x_C`   | Temperatura de horneado en grados Celsius (°C). Rango aproximado: 80–185 °C.   |
| `y_dB`  | Crocancia de la rodaja de papa: pico máximo de amplitud sonora en decibelios (dB). |

## 🧠 Modelos implementados

| # | Modelo         | Implementación                                                                   | Hiperparámetros/Detalle        |
|---|----------------|----------------------------------------------------------------------------------|--------------------------------|
| 1 | Varianza       | Regresión lineal de forma cerrada (`m = Cov(X,Y)/Var(X)`, `b = ȳ − m·x̄`).       | —                              |
| 2 | Mín. cuadrados | Regresión polinomial mediante `np.polyfit` + `np.poly1d`.                        | Grado 1–5 (selector)           |
| 3 | Polyfit        | Regresión polinomial equivalente a mínimos cuadrados (`np.polyfit`).             | Grado 1–5 (selector)           |
| 4 | KNN            | `KNeighborsRegressor` sobre características polinomiales escaladas.              | k = 3                          |
| 5 | SVR            | `SVR` con kernel RBF sobre características polinomiales escaladas.               | C = 10⁸, γ = 0.01, ε = 10⁻⁷   |
| 6 | Árbol          | `DecisionTreeRegressor` sobre características polinomiales escaladas.            | Profundidad máx. = 5           |
| 7 | Random Forest  | `RandomForestRegressor` sobre características polinomiales escaladas.            | 100 árboles                    |
| 8 | Red Neuronal   | Red neuronal **implementada desde cero con NumPy**: entrada polinomial normalizada (x, x², x³), activación *sigmoide* y descenso de gradiente; salida normalizada por min–max. | 5000 épocas, lr = 0.1          |

> ℹ️ Los modelos de Machine Learning (KNN, SVR, Árbol y Random Forest) se entrenan con características polinomiales de **grado 3** normalizadas con `StandardScaler`. La red neuronal entrena manualmente con NumPy (no utiliza el `torch` importado en el encabezado).

## 📈 Métricas de error

Para cada modelo ejecutado se calculan, con `n` = número de muestras:

- **MSE** (Error Cuadrático Medio): `MSE = (1/n) · Σ (yᵢ − ŷᵢ)²`
- **RMSE** (Raíz del Error Cuadrático Medio): `RMSE = √MSE`

## 📄 Reporte en PDF

Al pulsar **«Guardar PDF»**, la aplicación genera con `reportlab` un reporte que incluye:

- Modelo utilizado.
- Ecuación / coeficientes del modelo.
- RMSE obtenido.
- Gráfica del modelo vs. datos reales (`grafica_modelo.png`).
- Gráfica comparativa de todos los modelos (`comparacion_modelos.png`).

## 🧪 Metodología experimental

- **Rodajas de papa**: variedad **Kennebec o gallega**, de ~2 mm de grosor, untadas con un poco de aceite vegetal.
- **Cocción**: freidora de aire precalentada durante ~**15 minutos**.
- **Medición**: cada rodaja se rompe con los dedos a **máximo 2 cm** del micrófono del celular; el pico de amplitud (dB) registrado se considera la crocancia.
- **Aplicaciones de medición**: NIOSH (iPhone) y Sound Meter (Android), con los micrófonos de fábrica de un iPhone 13 y un Samsung S21 Ultra.

*Detalles completos —requisitos, variables de control, justificación y conclusiones— en [`Modelacion Reto 1.md`](./Modelacion%20Reto%201.md).*

## 📁 Archivos generados por la aplicación

| Archivo                      | Descripción                                                        |
|------------------------------|--------------------------------------------------------------------|
| `grafica_modelo.png`         | Gráfica del último modelo ejecutado frente a los datos reales.     |
| `comparacion_modelos.png`    | Gráfica comparativa de todos los modelos ejecutados.               |
| `*.pdf`                      | Reporte de modelamiento (guardado en la ruta elegida por el usuario). |

## 👥 Autores

- **Daniel Andres Cardona Muñoz**
- **Juan Jose Echeverri Quintero**
- **Tomas Saldarriaga Posso**

Facultad de Ingeniería · Universidad Católica Luis Amigó · Medellín, Colombia

## 🔗 Anexos

- Informe académico completo: [`Modelacion Reto 1.md`](./Modelacion%20Reto%201.md)
- Drive del proyecto (base de datos, código e informe): [Enlace](https://drive.google.com/drive/folders/1o_91SFqbsdi3_gpdzcGi4yVrULr2m7aQ?usp=drive_link)