# ==============================
# LIBRERIAS
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

# ==============================
# VARIABLES GLOBALES
# ==============================

X_real = None
Y_real = None

X = None
Y = None

X_poly_scaled = None

modelo_nn = None

ultimo_modelo = None
ultimo_rmse = None

pred_varianza = None
pred_minimos = None
pred_polyfit = None
pred_knn = None
pred_svr = None
pred_arbol = None
pred_rf = None
pred_nn = None

# ==============================
# EJECUTAR MODELO
# ==============================

def ejecutar_modelo():

    global X_real, Y_real
    global pred_varianza
    global pred_minimos
    global pred_polyfit
    global pred_knn
    global pred_svr
    global pred_arbol
    global pred_rf
    global pred_nn

    if X_real is None:
        print("Primero debes cargar un CSV")
        return

    modelo = selector_modelo.get()
    grado = int(selector_grado.get())

    print("Modelo seleccionado:",modelo)

    if modelo == "Varianza":

        m,b,Y_pred = LR_varianza(X_real,Y_real)

        pred_varianza = Y_pred

        ecuacion = f"y = {m:.6f}x + {b:.6f}"

        print("Ecuacion:",ecuacion)

    elif modelo == "Minimos Cuadrados":

        coef,Y_pred = Minimos_Cuadrados(X_real,Y_real,grado)

        pred_minimos = Y_pred

        print("Coeficientes:",coef)

    elif modelo == "Polyfit":

        coef,modelo_poly,Y_pred = Modelo_Polyfit(X_real,Y_real,grado)

        pred_polyfit = Y_pred

        print("Ecuacion:",modelo_poly)

    elif modelo == "KNN":

        modelo_knn = KNeighborsRegressor(n_neighbors=3)

        modelo_knn.fit(X_poly_scaled,Y_real)

        Y_pred = modelo_knn.predict(X_poly_scaled)

        pred_knn = Y_pred

        print("Modelo KNN ejecutado")


    elif modelo == "SVR":

        modelo = SVR(kernel="rbf", C=100000000, gamma=0.01, epsilon=0.0000001)

        modelo.fit(X_poly_scaled,Y)

        Y_pred = modelo.predict(X_poly_scaled)

        pred_svr = Y_pred

        print("Modelo SVR ejecutado")


    elif modelo == "Arbol":

        modelo_arbol = DecisionTreeRegressor(max_depth=5)

        modelo_arbol.fit(X_poly_scaled,Y_real)

        Y_pred = modelo_arbol.predict(X_poly_scaled)

        pred_arbol = Y_pred

        print("Modelo Arbol ejecutado")


    elif modelo == "Random Forest":

        modelo_rf = RandomForestRegressor(n_estimators=100)

        modelo_rf.fit(X_poly_scaled,Y_real)

        Y_pred = modelo_rf.predict(X_poly_scaled)

        pred_rf = Y_pred

        print("Modelo Random Forest ejecutado")

    elif modelo == "Red Neuronal":

        Y_pred = modelo_red_neuronal(X_real,Y_real)

        pred_nn = Y_pred

        print("Red neuronal entrenada")
    
    orden = np.argsort(X_real)

    X_plot = X_real[orden]
    Y_plot = Y_pred[orden]


    # Grafica
    plt.figure(figsize=(8,5))

    plt.scatter(X_real,Y_real,label="Datos reales")

    plt.plot(X_plot,Y_plot,color="red",label="Modelo")

    plt.legend()

    plt.grid()

    plt.savefig("grafica_modelo.png", dpi=300, bbox_inches="tight")

    plt.close()

    mostrar_grafica(X_real,Y_real,Y_plot)

    # Calcular error
    MSE,RMSE = Cal_error(Y_real,Y_pred)

    print("RMSE:",RMSE)

    global ultimo_modelo, ultimo_rmse

    ultimo_modelo = modelo
    ultimo_rmse = RMSE

def graficar_todos_modelos():

    if X_real is None:
        print("Primero debes cargar los datos")
        return

    orden = np.argsort(X_real)

    X_plot = X_real[orden]

    plt.figure(figsize=(10,6))

    plt.scatter(X_real,Y_real,color="black",label="Datos reales")

    if pred_varianza is not None:
        plt.plot(X_plot,pred_varianza[orden],label="Varianza")

    if pred_minimos is not None:
        plt.plot(X_plot,pred_minimos[orden],label="Minimos cuadrados")

    if pred_polyfit is not None:
        plt.plot(X_plot,pred_polyfit[orden],label="Polyfit")

    if pred_knn is not None:
        plt.plot(X_plot,pred_knn[orden],label="KNN")

    if pred_svr is not None:
        plt.plot(X_plot,pred_svr[orden],label="SVR")

    if pred_arbol is not None:
        plt.plot(X_plot,pred_arbol[orden],label="Arbol")

    if pred_rf is not None:
        plt.plot(X_plot,pred_rf[orden],label="Random Forest")

    if pred_nn is not None:
        plt.plot(X_plot,pred_nn[orden],label="Red Neuronal")

    plt.legend()

    plt.title("Comparación de Modelos")

    plt.grid()

    plt.savefig("comparacion_modelos.png",dpi=300,bbox_inches="tight")

    plt.show()

def ejecutar_todos_modelos():

    global pred_varianza
    global pred_minimos
    global pred_polyfit
    global pred_knn
    global pred_svr
    global pred_arbol
    global pred_rf
    global pred_nn

    if X_real is None:
        print("Primero debes cargar los datos")
        return

    grado = int(selector_grado.get())

    # VARIANZA
    m,b,pred_varianza = LR_varianza(X_real,Y_real)

    # MINIMOS CUADRADOS
    coef,pred_minimos = Minimos_Cuadrados(X_real,Y_real,grado)

    # POLYFIT
    coef,modelo_poly,pred_polyfit = Modelo_Polyfit(X_real,Y_real,grado)

    # KNN
    modelo_knn = KNeighborsRegressor(n_neighbors=3)
    modelo_knn.fit(X_poly_scaled,Y_real)
    pred_knn = modelo_knn.predict(X_poly_scaled)

    # SVR
    modelo_svr = SVR(kernel="rbf", C=100000000, gamma=0.01, epsilon=0.0000001)
    modelo_svr.fit(X_poly_scaled,Y)
    pred_svr = modelo_svr.predict(X_poly_scaled)

    # ARBOL
    modelo_arbol = DecisionTreeRegressor(max_depth=5)
    modelo_arbol.fit(X_poly_scaled,Y_real)
    pred_arbol = modelo_arbol.predict(X_poly_scaled)

    # RANDOM FOREST
    modelo_rf = RandomForestRegressor(n_estimators=100)
    modelo_rf.fit(X_poly_scaled,Y_real)
    pred_rf = modelo_rf.predict(X_poly_scaled)

    # RED NEURONAL
    pred_nn = modelo_red_neuronal(X_real,Y_real)

    print("Todos los modelos fueron ejecutados")

    graficar_todos_modelos()

# ==============================
# CARGAR CSV
# ==============================

def cargar_csv():

    global X_real, Y_real, X, Y, X_poly_scaled

    archivo = filedialog.askopenfilename(
        title="Seleccionar archivo CSV",
        filetypes=[("CSV files","*.csv")]
    )

    datos = pd.read_csv(archivo)

    datos_array = np.array(datos)

    X_real = datos_array[:,0]
    Y_real = datos_array[:,1]

    X = X_real.reshape(-1,1)
    Y = Y_real

    # Preparar datos ML
    poly = PolynomialFeatures(degree=3, include_bias=False)
    scaler = StandardScaler()

    X_poly = poly.fit_transform(X)
    X_poly_scaled = scaler.fit_transform(X_poly)

    print("Datos cargados correctamente")

# ==============================
# FUNCION DE ERROR
# ==============================

def Cal_error(Y_real,Y_model):

    n = len(Y_real)

    MSE = np.sum((Y_real - Y_model)**2)/n
    RMSE = np.sqrt(MSE)

    return(MSE,RMSE)

# ==============================
# GENERAR PDF
# ==============================

def generar_pdf(modelo,ecuacion,rmse,archivo):

    archivo = archivo

    c = canvas.Canvas(archivo, pagesize=letter)

    c.setFont("Helvetica",12)

    c.drawString(100,750,"Reporte de Modelamiento")

    c.drawString(100,720,f"Modelo utilizado: {modelo}")

    c.drawString(100,690,f"Ecuacion / coeficientes:")

    c.drawString(100,670,str(ecuacion))

    c.drawString(100,640,f"RMSE: {rmse}")

    c.drawImage("grafica_modelo.png",100,350,width=400,height=250)

    c.drawImage("comparacion_modelos.png",100,350,width=400,height=250)

    c.save()

    print("PDF generado:",archivo)

def guardar_pdf():

    if ultimo_modelo is None:
        print("Primero debes ejecutar un modelo")
        return

    archivo = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files","*.pdf")],
        title="Guardar reporte"
    )

    if archivo == "":
        return

    generar_pdf(ultimo_modelo,"Modelo generado",ultimo_rmse,archivo)

    print("PDF guardado correctamente")

# ==============================
# VENTANA PRINCIPAL
# ==============================

ventana = tk.Tk()

ventana.title("Modelamiento Crocantez Papa")

ventana.geometry("1000x800")

ventana.configure(bg="#E8EEF2")

boton_cargar = tk.Button(
    ventana,
    text="Cargar CSV",
    command=cargar_csv,
    bg="#4F6D7A",
    fg="white",
    width=20
)

boton_cargar.pack(pady=10)



label_modelo = tk.Label(ventana, text="Seleccionar modelo")
label_modelo.pack()

selector_modelo = ttk.Combobox(
    ventana,
    values=[
        "Varianza",
        "Minimos Cuadrados",
        "Polyfit",
        "KNN",
        "SVR",
        "Arbol",
        "Random Forest",
        "Red Neuronal"
    ]
)

selector_modelo.pack()

label_grado = tk.Label(ventana, text="Grado del polinomio")
label_grado.pack()

selector_grado = tk.Spinbox(
    ventana,
    from_=1,
    to=5
)

selector_grado.pack()

frame_grafica = tk.Frame(ventana)
frame_grafica.pack(fill="both", expand=True)

boton_ejecutar = tk.Button(
    ventana,
    text="Ejecutar Modelo",
    command=ejecutar_modelo
)

boton_ejecutar.pack(pady=20)

boton_ejecutar_todos = tk.Button(
    ventana,
    text="Ejecutar todos los modelos",
    command=ejecutar_todos_modelos
)

boton_ejecutar_todos.pack(pady=10)

boton_comparar = tk.Button(
    ventana,
    text="Comparar todos los modelos",
    command=graficar_todos_modelos
)

boton_comparar.pack(pady=10)

boton_pdf = tk.Button(
    ventana,
    text="Guardar PDF",
    command=guardar_pdf
)

boton_pdf.pack(pady=10)

def mostrar_grafica(X,Y,Y_pred):

    for widget in frame_grafica.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(5,4), dpi=100)

    ax = fig.add_subplot(111)

    ax.scatter(X,Y,label="Datos")

    ax.plot(X,Y_pred,label="Modelo")

    ax.set_title("Comparación Modelo vs Datos")

    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ==============================
# LOGO UNIVERSIDAD
# ==============================

import os

ruta_actual = os.path.dirname(__file__)
ruta_logo = os.path.join(ruta_actual, "logo.png")

imagen_logo = Image.open(ruta_logo)
imagen_logo = imagen_logo.resize((120,120))  # tamaño del logo

logo = ImageTk.PhotoImage(imagen_logo)

label_logo = tk.Label(
    ventana,
    image=logo,
    bg="#E8EEF2"
)

label_logo.pack(anchor="nw", padx=10, pady=10)

# ==============================
# Metodo Varianza
# ==============================

def LR_varianza(X,Y):

    Var_X = np.var(X)
    Cov_XY = np.cov(X,Y)

    m = Cov_XY[0,1] / Var_X
    b = np.mean(Y) - m*np.mean(X)

    Y_pred = m*X + b

    return m,b,Y_pred

# ==============================
# Minimos cuadrados
# ==============================

def Minimos_Cuadrados(X,Y,grado):

    coef = np.polyfit(X,Y,grado)

    modelo = np.poly1d(coef)

    Y_pred = modelo(X)

    return coef,Y_pred

# ==============================
# Polyfit
# ==============================

def Modelo_Polyfit(X,Y,grado):

    coef = np.polyfit(X,Y,grado)

    modelo = np.poly1d(coef)

    Y_pred = modelo(X)

    return coef,modelo,Y_pred

# ==============================
# RED NEURONAL 
# ==============================

def modelo_red_neuronal(X,Y):

    X = np.array(X)
    Y = np.array(Y)

    # =============================
    # NORMALIZACION
    # =============================

    X_mean = np.mean(X)
    X_std = np.std(X)

    X_norm = (X - X_mean) / X_std

    Y_min = np.min(Y)
    Y_max = np.max(Y)

    Y_real = (Y - Y_min) / (Y_max - Y_min)

    X2 = X_norm**2
    X3 = X_norm**3

    # =============================
    # PARAMETROS INICIALES
    # =============================

    rng = np.random.default_rng()

    b0 = rng.normal(0,0.1)
    b1 = rng.normal(0,0.1)
    b2 = rng.normal(0,0.1)
    b3 = rng.normal(0,0.1)

    # Sigmoide
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    # =============================
    # ENTRENAMIENTO
    # =============================

    epochs = 5000
    lr = 0.1
    N = len(X_norm)

    for step in range(epochs):

        b0_g = 0
        b1_g = 0
        b2_g = 0
        b3_g = 0
        error = 0

        for i in range(N):

            z = b0 + b1*X_norm[i] + b2*X2[i] + b3*X3[i]

            Yp = sigmoid(z)

            diff = Y_real[i] - Yp

            error += diff**2

            grad = -(2/N) * diff * Yp * (1-Yp)

            b0_g += grad
            b1_g += grad * X_norm[i]
            b2_g += grad * X2[i]
            b3_g += grad * X3[i]

        # actualizar pesos
        b0 -= lr*b0_g
        b1 -= lr*b1_g
        b2 -= lr*b2_g
        b3 -= lr*b3_g

    # =============================
    # PREDICCION
    # =============================

    Y_pred_norm = sigmoid(b0 + b1*X_norm + b2*X2 + b3*X3)

    Y_pred = Y_pred_norm*(Y_max-Y_min) + Y_min

    print("Coeficientes finales:")
    print("b0 =",b0)
    print("b1 =",b1)
    print("b2 =",b2)
    print("b3 =",b3)

    return Y_pred


label_autores = tk.Label(
    ventana,
    text="Generado por Juan Echeverri, Tomas Saldarriaga y Daniel Cardona",
    bg="#E8EEF2",
    fg="#333333",
    font=("Arial",9)
)

label_autores.pack(side="bottom", pady=10)
ventana.mainloop()