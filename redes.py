"""
Lógica de las tres redes neuronales de la Práctica 4 (sin interfaz).

1. Reconocimiento de patrones (clasificación) – Mantenimiento predictivo (AI4I 2020, UCI 601)
2. Ajuste de funciones (regresión)             – Potencia de planta de ciclo combinado (UCI 294)
3. Series de tiempo (clasificación)            – Válvula de un sistema hidráulico (UCI 447)

Cada función entrenar_*() reproduce el código de los cuadernos de Colab (mismos datos, divisiones,
parámetros y semilla) y devuelve un diccionario con la red entrenada y sus resultados.
"""
import io
import urllib.request
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

tf.config.experimental.enable_op_determinism()      # Resultados reproducibles


class Progreso(tf.keras.callbacks.Callback):
    """Informa a la interfaz el avance del entrenamiento al final de cada época."""

    def __init__(self, avisar, total):
        super().__init__()
        self.avisar, self.total = avisar, total

    def on_epoch_end(self, epoch, logs=None):
        if self.avisar:
            self.avisar(epoch + 1, self.total, logs or {})


def resumen_capas(red):
    """Tabla con las capas de la red (equivalente a red.summary())."""
    def forma(c):
        try:
            return str(tuple(c.output.shape)).replace("None", "lote")
        except Exception:
            return "—"
    filas = [{"Capa": c.name, "Tipo": c.__class__.__name__, "Forma de salida": forma(c),
              "Parámetros": c.count_params()} for c in red.layers]
    return pd.DataFrame(filas)


# ============================================================ 1. CLASIFICACIÓN
P1 = dict(capas=[64, 32], dropout=0.2, lr=1e-3, epocas=200, lote=64)
TIPOS = {"L": "Baja (L)", "M": "Media (M)", "H": "Alta (H)"}


@lru_cache(maxsize=1)
def datos_clasificacion():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=601)
    X = d.data.features.copy()
    y = d.data.targets["Machine failure"].values
    X.columns = [c.split(" [")[0] for c in X.columns]
    X = pd.get_dummies(X, columns=["Type"], dtype=float)
    return X, y


def entrenar_clasificacion(avisar=None):
    tf.keras.utils.set_random_seed(42)
    X, y = datos_clasificacion()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_va, X_ev, y_va, y_ev = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    escalador = StandardScaler().fit(X_tr)
    rangos = X_tr.agg(["min", "max"])
    X_tr, X_va, X_ev = (escalador.transform(a) for a in (X_tr, X_va, X_ev))
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)

    red = tf.keras.Sequential([tf.keras.Input(shape=(X_tr.shape[1],))])
    for n in P1["capas"]:
        red.add(tf.keras.layers.Dense(n, activation="relu"))
        red.add(tf.keras.layers.Dropout(P1["dropout"]))
    red.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    red.compile(optimizer=tf.keras.optimizers.Adam(P1["lr"]), loss="binary_crossentropy",
                metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"])

    h = red.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=P1["epocas"], batch_size=P1["lote"],
                class_weight={0: w[0], 1: w[1]}, verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=15,
                                                            restore_best_weights=True),
                           Progreso(avisar, P1["epocas"])])

    p_va = red.predict(X_va, verbose=0).ravel()
    umbrales = np.arange(0.1, 0.96, 0.05)
    umbral = float(umbrales[np.argmax([f1_score(y_va, p_va >= t) for t in umbrales])])

    p_ev = red.predict(X_ev, verbose=0).ravel()
    y_pred = (p_ev >= umbral).astype(int)
    return dict(red=red, escalador=escalador, columnas=list(X.columns), rangos=rangos, umbral=umbral,
                historial=h.history, epocas=len(h.history["loss"]), y_ev=y_ev, p_ev=p_ev, y_pred=y_pred,
                mc=confusion_matrix(y_ev, y_pred), auc=roc_auc_score(y_ev, p_ev),
                reporte=classification_report(y_ev, y_pred, target_names=["Sin falla", "Falla"],
                                              output_dict=True),
                capas=resumen_capas(red), n=len(y), pct_fallas=y.mean())


def diagnosticar(res, tipo, temp_aire, temp_proceso, rpm, torque, desgaste):
    """Probabilidad de falla de una máquina a partir de sus lecturas."""
    fila = pd.DataFrame([[temp_aire, temp_proceso, rpm, torque, desgaste,
                          tipo == "H", tipo == "L", tipo == "M"]], columns=res["columnas"], dtype=float)
    return float(res["red"].predict(res["escalador"].transform(fila), verbose=0)[0, 0])


# ============================================================ 2. REGRESIÓN
P2 = dict(capas=[64, 64], lr=1e-3, epocas=500, lote=64)
VARIABLES_P2 = ["Temperatura (°C)", "Vacío de escape (cm Hg)", "Presión ambiente (mbar)", "Humedad relativa (%)"]


@lru_cache(maxsize=1)
def datos_regresion():
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=294)
    X = d.data.features.copy()
    X.columns = VARIABLES_P2
    y = d.data.targets.iloc[:, 0].values.reshape(-1, 1)
    return X, y


def entrenar_regresion(avisar=None):
    tf.keras.utils.set_random_seed(42)
    X, y = datos_regresion()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_va, X_ev, y_va, y_ev = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    rangos = X_tr.agg(["min", "max"])
    esc_x, esc_y = StandardScaler().fit(X_tr), StandardScaler().fit(y_tr)
    X_tr, X_va, X_ev = (esc_x.transform(a) for a in (X_tr, X_va, X_ev))

    red = tf.keras.Sequential([tf.keras.Input(shape=(X_tr.shape[1],))])
    for n in P2["capas"]:
        red.add(tf.keras.layers.Dense(n, activation="relu"))
    red.add(tf.keras.layers.Dense(1, activation="linear"))
    red.compile(optimizer=tf.keras.optimizers.Adam(P2["lr"]), loss="mse", metrics=["mae"])

    h = red.fit(X_tr, esc_y.transform(y_tr), validation_data=(X_va, esc_y.transform(y_va)),
                epochs=P2["epocas"], batch_size=P2["lote"], verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=40,
                                                            restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15),
                           Progreso(avisar, P2["epocas"])])

    def predecir(Xn):
        return esc_y.inverse_transform(red.predict(Xn, verbose=0)).ravel()

    metricas = pd.DataFrame([
        {"Conjunto": nombre, "R²": r2_score(yr, yp), "MAE (MW)": mean_absolute_error(yr, yp),
         "RMSE (MW)": mean_squared_error(yr, yp) ** 0.5, "MAPE": mean_absolute_percentage_error(yr, yp),
         "Muestras": len(yr)}
        for nombre, (Xc, yr) in {"Entrenamiento": (X_tr, y_tr), "Validación": (X_va, y_va),
                                 "Evaluación": (X_ev, y_ev)}.items()
        for yp in [predecir(Xc)]])
    y_ev, p_ev = y_ev.ravel(), predecir(X_ev)
    return dict(red=red, esc_x=esc_x, esc_y=esc_y, rangos=rangos, historial=h.history,
                epocas=len(h.history["loss"]), metricas=metricas, y_ev=y_ev, p_ev=p_ev,
                r2=r2_score(y_ev, p_ev), r=np.corrcoef(y_ev, p_ev)[0, 1], mae=mean_absolute_error(y_ev, p_ev),
                cortes=np.percentile(y_tr, [100 / 3, 200 / 3]), y_hist=y_tr.ravel(),
                capas=resumen_capas(red), n=len(y))


def estimar(res, temperatura, vacio, presion, humedad):
    """Potencia estimada (MW) y nivel de generación para unas condiciones de operación."""
    fila = pd.DataFrame([[temperatura, vacio, presion, humedad]], columns=VARIABLES_P2)
    mw = float(res["esc_y"].inverse_transform(res["red"].predict(res["esc_x"].transform(fila), verbose=0))[0, 0])
    c1, c2 = res["cortes"]
    return mw, ("Bajo" if mw < c1 else "Medio" if mw < c2 else "Alto")


# ============================================================ 3. SERIES DE TIEMPO
P3 = dict(filtros=[32, 64, 64], lstm=64, dropout=0.3, lr=1e-3, epocas=100, lote=32)
URL_P3 = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
SENSORES = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2"]
ESTADOS = np.array(["Óptima (100 %)", "Retraso leve (90 %)", "Retraso severo (80 %)", "Falla casi total (73 %)"])


@lru_cache(maxsize=1)
def datos_series():
    """Descarga el dataset (73 MB) y arma las series: (ciclos, 600 pasos, 9 sensores) a 10 Hz."""
    archivo = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL_P3).read()))

    def leer(nombre):
        ruta = next(n for n in archivo.namelist() if n.split("/")[-1] == f"{nombre}.txt")
        return pd.read_csv(archivo.open(ruta), sep="\t", header=None).values

    series = []
    for s in SENSORES:
        datos = leer(s)
        if datos.shape[1] == 6000:                                   # 100 Hz -> 10 Hz
            datos = datos.reshape(len(datos), 600, 10).mean(axis=2)
        series.append(datos.astype("float32"))
    X = np.stack(series, axis=-1)
    y = pd.Series(leer("profile")[:, 1]).map({100: 0, 90: 1, 80: 2, 73: 3}).values
    return X, y


def entrenar_series(avisar=None):
    tf.keras.utils.set_random_seed(42)
    X, y = datos_series()
    ciclos = np.arange(len(y))
    X_tr, X_tmp, y_tr, y_tmp, c_tr, c_tmp = train_test_split(X, y, ciclos, test_size=0.3, stratify=y,
                                                             random_state=42)
    X_va, X_ev, y_va, y_ev, c_va, c_ev = train_test_split(X_tmp, y_tmp, c_tmp, test_size=0.5,
                                                          stratify=y_tmp, random_state=42)
    media, desv = X_tr.mean(axis=(0, 1)), X_tr.std(axis=(0, 1))
    X_ev_crudo = X_ev
    X_tr, X_va, X_ev = ((a - media) / desv for a in (X_tr, X_va, X_ev))

    red = tf.keras.Sequential([tf.keras.Input(shape=X_tr.shape[1:])])
    for f in P3["filtros"]:
        red.add(tf.keras.layers.Conv1D(f, 5, padding="same", activation="relu"))
        red.add(tf.keras.layers.MaxPooling1D(2))
    red.add(tf.keras.layers.LSTM(P3["lstm"]))
    red.add(tf.keras.layers.Dropout(P3["dropout"]))
    red.add(tf.keras.layers.Dense(len(ESTADOS), activation="softmax"))
    red.compile(optimizer=tf.keras.optimizers.Adam(P3["lr"]), loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    h = red.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=P3["epocas"], batch_size=P3["lote"],
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                            restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
                           Progreso(avisar, P3["epocas"])])

    p_ev = red.predict(X_ev, verbose=0)
    y_pred = p_ev.argmax(axis=1)
    ejemplos = {k: X[c_tr[y_tr == k][0], :, 0] for k in range(len(ESTADOS))}   # PS1 de un ciclo por estado
    return dict(red=red, historial=h.history, epocas=len(h.history["loss"]), X_ev=X_ev, X_ev_crudo=X_ev_crudo,
                y_ev=y_ev, c_ev=c_ev, p_ev=p_ev, y_pred=y_pred, mc=confusion_matrix(y_ev, y_pred),
                reporte=classification_report(y_ev, y_pred, target_names=ESTADOS, output_dict=True),
                ejemplos=ejemplos, capas=resumen_capas(red), n=len(y))


def diagnosticar_ciclo(res, i):
    """Probabilidades de cada estado de la válvula para el ciclo i del conjunto de evaluación."""
    return res["red"].predict(res["X_ev"][i:i + 1], verbose=0)[0]
