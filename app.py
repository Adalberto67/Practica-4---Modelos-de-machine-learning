"""
Práctica 4 · Modelos de machine learning — Página interactiva (Streamlit)
Alumno: Adalberto Enai Pech Us · Sistemas Inteligentes

Tres redes neuronales en una sola página: entrenarlas, ver sus resultados y probarlas con valores
elegidos por el usuario. La lógica de las redes está en redes.py y el diseño en estilo.py.
Ejecutar localmente:  streamlit run app.py
"""
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import RocCurveDisplay

import estilo as es
import redes

st.set_page_config(page_title="Práctica 4 · Redes neuronales", page_icon="🧠", layout="wide")
st.markdown(es.CSS, unsafe_allow_html=True)
es.estilo_graficas()


# ------------------------------------------------------------------ Modelos compartidos
@st.cache_resource
def almacen():
    """Redes ya entrenadas (compartidas entre visitantes) y un candado para no entrenar dos veces a la vez."""
    return {"modelos": {}, "candado": threading.Lock()}


ALM = almacen()

REDES = {
    1: dict(menu="1 · Clasificación", tipo="Reconocimiento de patrones · Clasificación",
            titulo="¿Fallará esta máquina?", funcion=redes.entrenar_clasificacion, tiempo="1 a 2 minutos",
            problema="Mantenimiento predictivo de una fresadora industrial. A partir de las lecturas de sus "
                     "sensores, la red clasifica si la máquina presentará una falla.",
            dataset="AI4I 2020 Predictive Maintenance · UCI (10,000 registros, 3.4 % con falla)",
            url="https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
            red="Perceptrón multicapa 8 → 64 → 32 → 1 (salida sigmoide)",
            referencia="En Colab: ROC-AUC = 0.975 · exactitud = 0.972"),
    2: dict(menu="2 · Regresión", tipo="Ajuste de funciones · Regresión",
            titulo="¿Cuánta energía generará la planta?", funcion=redes.entrenar_regresion, tiempo="1 a 3 minutos",
            problema="Planta eléctrica de ciclo combinado. A partir de las condiciones ambientales, la red estima "
                     "la potencia eléctrica (MW) que generará la planta.",
            dataset="Combined Cycle Power Plant · UCI (9,568 registros horarios reales)",
            url="https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant",
            red="Perceptrón multicapa 4 → 64 → 64 → 1 (salida lineal)",
            referencia="En Colab: R² = 0.945 · error medio = 2.90 MW"),
    3: dict(menu="3 · Series de tiempo", tipo="Series de tiempo · Clasificación de secuencias",
            titulo="¿En qué estado está la válvula?", funcion=redes.entrenar_series, tiempo="3 a 8 minutos",
            problema="Monitoreo de condición de un sistema hidráulico. La red analiza cómo evolucionan 9 sensores "
                     "durante un ciclo de trabajo de 60 segundos e identifica el estado de la válvula de control.",
            dataset="Condition Monitoring of Hydraulic Systems · UCI (2,205 ciclos de un banco de pruebas real)",
            url="https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems",
            red="Convolucional 1D (32-64-64) + LSTM (64) → 4 estados (softmax)",
            referencia="En Colab: exactitud = 0.988 · F1 promedio = 0.985"),
}
PARAMETROS = {
    1: {"Capas ocultas": "64 y 32 neuronas (ReLU)", "Dropout": "0.2", "Optimizador": "Adam (tasa 0.001)",
        "Lote / épocas máx.": "64 / 200", "Detención temprana": "Paciencia 15 (AUC de validación)",
        "Desbalance": "Pesos de clase + umbral óptimo por F1"},
    2: {"Capas ocultas": "64 y 64 neuronas (ReLU)", "Salida": "Lineal (MW)", "Optimizador": "Adam (tasa 0.001)",
        "Lote / épocas máx.": "64 / 500", "Detención temprana": "Paciencia 40 + reducción de tasa",
        "Normalización": "Entradas y salida"},
    3: {"Convolución 1D": "32, 64 y 64 filtros (tamaño 5) + MaxPooling", "Recurrente": "LSTM de 64 unidades",
        "Dropout": "0.3", "Optimizador": "Adam (tasa 0.001)", "Lote / épocas máx.": "32 / 100",
        "Entrada": "600 pasos de tiempo × 9 sensores"},
}


def modelo(k):
    return ALM["modelos"].get(k)


def mostrar(styler):
    st.markdown(es.html(styler), unsafe_allow_html=True)


def tabla_simple(diccionario, c1, c2, titulo):
    mostrar(es.tabla(pd.DataFrame({c1: list(diccionario), c2: list(diccionario.values())}), titulo))


# ------------------------------------------------------------------ Barra lateral
with st.sidebar:
    st.markdown("### 🧠 Práctica 4")
    st.caption("Modelos de machine learning · Sistemas Inteligentes")
    pagina = st.radio("Ir a", ["Inicio"] + [r["menu"] for r in REDES.values()], label_visibility="collapsed")
    st.divider()
    st.markdown("**Estado de las redes**")
    for k, r in REDES.items():
        st.markdown(f"{'🟢' if modelo(k) else '⚪'} {r['menu']} — {'entrenada' if modelo(k) else 'sin entrenar'}")
    st.divider()
    st.caption("Adalberto Enai Pech Us\n\nPython · TensorFlow/Keras · Streamlit")


# ------------------------------------------------------------------ Inicio
def pagina_inicio():
    st.markdown("""<div class="portada"><div class="etq">Práctica 4 · Sistemas Inteligentes</div>
        <h1>Tres redes neuronales aplicadas a la industria</h1>
        <p>Entrena cada red con datos reales del UCI Machine Learning Repository, revisa sus resultados y
        pruébala con tus propios valores. No necesitas instalar nada.</p></div>""", unsafe_allow_html=True)

    st.markdown("#### ¿Cómo se usa?")
    for n, texto in [("1", "<b>Elige una red</b> en el menú de la izquierda."),
                     ("2", "En la pestaña <b>Entrenar</b>, pulsa el botón y observa el entrenamiento época por época."),
                     ("3", "En <b>Resultados</b> revisa las métricas, tablas y gráficas obtenidas con datos de prueba."),
                     ("4", "En <b>Probar la red</b> elige tus propios valores y obtén la predicción al instante.")]:
        st.markdown(f'<div class="paso"><b class="n">{n}</b><div>{texto}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Las tres redes")
    for col, (k, r) in zip(st.columns(3, gap="medium"), REDES.items()):
        with col:
            st.markdown(f"""<div class="tarjeta">
                <span class="chip" style="background:{es.COLOR[k]}1A; color:{es.COLOR[k]}">{r['tipo'].split(' · ')[-1]}</span><br>
                <span class="num" style="background:{es.COLOR[k]}">{k}</span><h4>{r['titulo']}</h4>
                <p>{r['problema']}</p><p><b>Red:</b> {r['red']}</p>
                <p class="nota">Datos: {r['dataset']}</p></div>""", unsafe_allow_html=True)

    with st.expander("Detalles del procedimiento"):
        st.markdown(
            "- Los datos se dividen en **70 % entrenamiento, 15 % validación y 15 % evaluación**; las métricas se "
            "calculan con los datos de evaluación, que la red nunca vio al entrenar.\n"
            "- Se fija la **semilla aleatoria (42)**, por lo que el entrenamiento es reproducible. Los valores pueden "
            "variar ligeramente respecto a Colab por diferencias de hardware o de versión de las bibliotecas.\n"
            "- Una red entrenada queda disponible para todos los visitantes hasta que el servidor se reinicie.\n"
            "- Código generado con apoyo de inteligencia artificial generativa (Claude, de Anthropic).")


# ------------------------------------------------------------------ Componentes comunes
def encabezado(k):
    r = REDES[k]
    st.markdown(f'<span class="chip" style="background:{es.COLOR[k]}1A; color:{es.COLOR[k]}">{r["tipo"]}</span>',
                unsafe_allow_html=True)
    st.markdown(f"## {r['titulo']}")
    st.markdown(r["problema"])


def pestana_entrenar(k):
    r = REDES[k]
    izq, der = st.columns([1.1, 1], gap="large")
    with izq:
        st.markdown("##### Datos")
        st.markdown(f"[{r['dataset']}]({r['url']}) — se descargan automáticamente al entrenar.")
        st.markdown("##### Arquitectura y parámetros ajustados")
        st.markdown(f"**{r['red']}**")
        tabla_simple(PARAMETROS[k], "Parámetro", "Valor", "Configuración de la red")
    with der:
        st.markdown("##### Entrenamiento")
        res = modelo(k)
        if res:
            st.success(f"Red entrenada en **{res['epocas']} épocas** ({res['segundos']:.0f} s). "
                       "Revisa las pestañas **Resultados** y **Probar la red**.")
            pulsado = st.button("Volver a entrenar", key=f"reentrenar{k}")
        else:
            st.info(f"La red aún no está entrenada. Tiempo aproximado: **{r['tiempo']}**.")
            pulsado = st.button("▶ Entrenar red neuronal", type="primary", key=f"entrenar{k}")
        if pulsado:
            entrenar(k)
        if res:
            mostrar(es.tabla(res["capas"], "Capas de la red entrenada"))
            st.caption(f"Parámetros entrenables: {int(res['capas']['Parámetros'].sum()):,}")


def entrenar(k):
    with ALM["candado"]:
        barra = st.progress(0.0, text="Descargando y preparando los datos…")
        t0 = time.time()

        def avisar(epoca, total, logs):
            barra.progress(min(epoca / total, 1.0),
                           text=f"Época {epoca} · pérdida {logs.get('loss', 0):.4f} · "
                                f"pérdida de validación {logs.get('val_loss', 0):.4f}")

        res = REDES[k]["funcion"](avisar)
        res["segundos"] = time.time() - t0
        barra.progress(1.0, text="Entrenamiento terminado")
        ALM["modelos"][k] = res
    st.rerun()


def sin_entrenar():
    st.warning("Primero entrena la red en la pestaña **① Entrenar**.")


def curvas(ax_perdida, ax_metrica, h, metrica, nombre):
    ax_perdida.plot(h["loss"], label="Entrenamiento"); ax_perdida.plot(h["val_loss"], label="Validación")
    ax_perdida.set(title="Error de entrenamiento", xlabel="Época", ylabel="Pérdida"); ax_perdida.legend()
    if ax_metrica is not None:
        ax_metrica.plot(h[metrica], label="Entrenamiento"); ax_metrica.plot(h[f"val_{metrica}"], label="Validación")
        ax_metrica.set(title=f"{nombre} por época", xlabel="Época", ylabel=nombre); ax_metrica.legend()


def historial(clave, columnas_formato, titulo, colorear=None):
    """Tabla con todas las pruebas que ha hecho el usuario en esta sesión."""
    filas = st.session_state.get(clave, [])
    if not filas:
        return
    sty = es.tabla(pd.DataFrame(filas), titulo).format(columnas_formato)
    if colorear:
        sty = sty.map(colorear[1], subset=[colorear[0]])
    mostrar(sty)
    if st.button("Limpiar historial", key=f"limpiar_{clave}"):
        st.session_state[clave] = []
        st.rerun()


# ------------------------------------------------------------------ 1. Clasificación
EJ1 = {"Operación normal": ("M", 298.1, 308.6, 1551, 42.8, 0),
       "Torque alto y herramienta desgastada": ("L", 298.9, 309.1, 1410, 65.7, 191),
       "Sobrecarga (potencia > 9 kW)": ("L", 302.5, 311.8, 1300, 68.0, 220)}
CAMPOS1 = ["p1_tipo", "p1_ta", "p1_tp", "p1_rpm", "p1_tq", "p1_dw"]


def cargar_ej1():
    if st.session_state.ej1 in EJ1:
        for c, v in zip(CAMPOS1, EJ1[st.session_state.ej1]):
            st.session_state[c] = v


def red_1():
    encabezado(1)
    t1, t2, t3 = st.tabs(["① Entrenar", "② Resultados", "③ Probar la red"])
    with t1:
        pestana_entrenar(1)
    res = modelo(1)
    with t2:
        if not res:
            sin_entrenar()
        else:
            rep = res["reporte"]
            c = st.columns(4)
            c[0].metric("ROC-AUC", f"{res['auc']:.3f}")
            c[1].metric("Exactitud", f"{rep['accuracy']:.3f}")
            c[2].metric("Recall de fallas", f"{rep['Falla']['recall']:.3f}")
            c[3].metric("Umbral de decisión", f"{res['umbral']:.2f}")
            st.caption(REDES[1]["referencia"])
            clases = (pd.DataFrame(rep).T.drop("accuracy")
                      .rename(index={"macro avg": "Promedio macro", "weighted avg": "Promedio ponderado"})
                      .rename_axis("Clase").reset_index()
                      .rename(columns={"precision": "Precisión", "recall": "Recall", "f1-score": "F1-score",
                                       "support": "Muestras"}))
            mc = res["mc"]
            matriz = pd.DataFrame({"Clase real": ["Sin falla", "Falla"], "Predicho: Sin falla": mc[:, 0],
                                   "Predicho: Falla": mc[:, 1]})
            a, b = st.columns([1.3, 1], gap="large")
            with a:
                mostrar(es.tabla(clases, "Métricas por clase (evaluación)")
                        .format({"Precisión": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "Muestras": "{:.0f}"}))
            with b:
                mostrar(es.tabla(matriz, "Matriz de confusión")
                        .apply(es.colores_matriz, axis=None, subset=list(matriz.columns[1:])))
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
            curvas(ax[0], ax[1], res["historial"], "auc", "AUC")
            RocCurveDisplay.from_predictions(res["y_ev"], res["p_ev"], ax=ax[2], name="Red neuronal")
            ax[2].set_title("Curva ROC")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.markdown(f"La red detectó **{mc[1, 1]} de {mc[1].sum()} fallas reales** con **{mc[0, 1]} falsas "
                        f"alarmas** entre {mc[0].sum()} máquinas sanas. Las curvas de entrenamiento y validación "
                        "avanzan juntas, por lo que no hay sobreajuste.")
    with t3:
        if not res:
            sin_entrenar()
            return
        st.markdown("Ajusta las lecturas de los sensores (o carga un ejemplo) y pulsa **Diagnosticar**.")
        rg = res["rangos"]
        for c, v in zip(CAMPOS1, EJ1["Operación normal"]):
            st.session_state.setdefault(c, v)
        st.selectbox("Cargar un ejemplo", ["Elegir…"] + list(EJ1), key="ej1", on_change=cargar_ej1)
        col = st.columns(3, gap="medium")
        tipo = col[0].selectbox("Calidad del producto", list(redes.TIPOS), format_func=redes.TIPOS.get, key="p1_tipo")
        ta = col[1].slider("Temperatura del aire (K)", float(np.floor(rg["Air temperature"]["min"])),
                           float(np.ceil(rg["Air temperature"]["max"])), step=0.1, key="p1_ta")
        tp = col[2].slider("Temperatura del proceso (K)", float(np.floor(rg["Process temperature"]["min"])),
                           float(np.ceil(rg["Process temperature"]["max"])), step=0.1, key="p1_tp")
        rpm = col[0].slider("Velocidad de rotación (rpm)", int(rg["Rotational speed"]["min"]),
                            int(rg["Rotational speed"]["max"]), step=1, key="p1_rpm")
        tq = col[1].slider("Torque (Nm)", float(np.floor(rg["Torque"]["min"])), float(np.ceil(rg["Torque"]["max"])),
                           step=0.1, key="p1_tq")
        dw = col[2].slider("Desgaste de la herramienta (min)", 0, int(rg["Tool wear"]["max"]), step=1, key="p1_dw")
        if st.button("🔍 Diagnosticar máquina", type="primary"):
            p = redes.diagnosticar(res, tipo, ta, tp, rpm, tq, dw)
            st.session_state.setdefault("hist1", []).append(
                {"Máquina": f"Prueba {len(st.session_state.get('hist1', [])) + 1}", "Tipo": tipo,
                 "Aire (K)": ta, "Proceso (K)": tp, "rpm": rpm, "Torque (Nm)": tq, "Desgaste (min)": dw,
                 "Prob. de falla": p, "Diagnóstico": "FALLA" if p >= res["umbral"] else "SIN FALLA"})
        if st.session_state.get("hist1"):
            u = st.session_state["hist1"][-1]
            falla = u["Diagnóstico"] == "FALLA"
            st.markdown(es.veredicto("Diagnóstico de la red", u["Diagnóstico"],
                                     f"Probabilidad de falla: <b>{u['Prob. de falla']:.1%}</b> · la red indica falla "
                                     f"a partir de {res['umbral']:.0%} (marca negra)", "rojo" if falla else "verde")
                        + es.barra(u["Prob. de falla"], res["umbral"], "#B42318" if falla else "#067647"),
                        unsafe_allow_html=True)
        historial("hist1", {"Prob. de falla": "{:.1%}", "Aire (K)": "{:.1f}", "Proceso (K)": "{:.1f}",
                            "Torque (Nm)": "{:.1f}"}, "Máquinas diagnosticadas",
                  ("Diagnóstico", lambda v: es.ROJO if v == "FALLA" else es.VERDE))


# ------------------------------------------------------------------ 2. Regresión
EJ2 = {"Mañana de invierno": (8.0, 40.0, 1018.0, 85.0), "Tarde templada": (20.0, 55.0, 1013.0, 70.0),
       "Mediodía de verano": (32.0, 70.0, 1008.0, 45.0)}
CAMPOS2 = ["p2_t", "p2_v", "p2_p", "p2_h"]


def cargar_ej2():
    if st.session_state.ej2 in EJ2:
        for c, v in zip(CAMPOS2, EJ2[st.session_state.ej2]):
            st.session_state[c] = v


def red_2():
    encabezado(2)
    t1, t2, t3 = st.tabs(["① Entrenar", "② Resultados", "③ Probar la red"])
    with t1:
        pestana_entrenar(2)
    res = modelo(2)
    with t2:
        if not res:
            sin_entrenar()
        else:
            c = st.columns(4)
            c[0].metric("R² (evaluación)", f"{res['r2']:.3f}")
            c[1].metric("Correlación R", f"{res['r']:.3f}")
            c[2].metric("Error medio", f"{res['mae']:.2f} MW")
            c[3].metric("Épocas", res["epocas"])
            st.caption(REDES[2]["referencia"])
            y_ev, p_ev = res["y_ev"], res["p_ev"]
            comp = pd.DataFrame({"Registro": [f"Registro {i + 1}" for i in range(8)], "Real (MW)": y_ev[:8],
                                 "Predicho (MW)": p_ev[:8]})
            comp["Error (MW)"] = comp["Predicho (MW)"] - comp["Real (MW)"]
            a, b = st.columns([1.3, 1], gap="large")
            with a:
                mostrar(es.tabla(res["metricas"], "Métricas por conjunto")
                        .format({"R²": "{:.3f}", "MAE (MW)": "{:.2f}", "RMSE (MW)": "{:.2f}", "MAPE": "{:.2%}",
                                 "Muestras": "{:.0f}"}))
            with b:
                mostrar(es.tabla(comp, "Real vs. predicho (evaluación)")
                        .format("{:.2f}", subset=["Real (MW)", "Predicho (MW)", "Error (MW)"])
                        .map(lambda e: es.VERDE if abs(e) < 3 else es.AMARILLO if abs(e) < 6 else es.ROJO,
                             subset=["Error (MW)"]))
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
            curvas(ax[0], None, res["historial"], None, None); ax[0].set_yscale("log")
            lim = [min(y_ev.min(), p_ev.min()) - 5, max(y_ev.max(), p_ev.max()) + 5]
            ax[1].scatter(y_ev, p_ev, s=8, alpha=0.45, color=es.COLOR[2])
            ax[1].plot(lim, lim, "k--", lw=1, label="Ajuste perfecto")
            ax[1].set(title=f"Real vs. predicho (R² = {res['r2']:.3f})", xlabel="Real (MW)", ylabel="Predicho (MW)",
                      xlim=lim, ylim=lim); ax[1].legend()
            ax[2].hist(p_ev - y_ev, bins=30, color=es.COLOR[2], edgecolor="white")
            ax[2].axvline(0, color="k", ls="--", lw=1)
            ax[2].set(title="Distribución de errores", xlabel="Error (MW)", ylabel="Frecuencia")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.markdown(f"El error medio equivale a **{res['mae'] / y_ev.mean():.2%}** de la potencia de la planta. "
                        "Entrenamiento, validación y evaluación tienen métricas muy similares, por lo que no hay "
                        "sobreajuste, y los errores se reparten de forma simétrica alrededor de cero.")
    with t3:
        if not res:
            sin_entrenar()
            return
        st.markdown("Ajusta las condiciones ambientales (o carga un escenario) y pulsa **Estimar potencia**.")
        rg = res["rangos"]
        for c, v in zip(CAMPOS2, EJ2["Tarde templada"]):
            st.session_state.setdefault(c, v)
        st.selectbox("Cargar un escenario", ["Elegir…"] + list(EJ2), key="ej2", on_change=cargar_ej2)
        col = st.columns(2, gap="large")
        lim = {v: (float(np.floor(rg[v]["min"])), float(np.ceil(rg[v]["max"]))) for v in redes.VARIABLES_P2}
        t = col[0].slider(redes.VARIABLES_P2[0], *lim[redes.VARIABLES_P2[0]], step=0.1, key="p2_t")
        v = col[1].slider(redes.VARIABLES_P2[1], *lim[redes.VARIABLES_P2[1]], step=0.1, key="p2_v")
        p = col[0].slider(redes.VARIABLES_P2[2], *lim[redes.VARIABLES_P2[2]], step=0.1, key="p2_p")
        h = col[1].slider(redes.VARIABLES_P2[3], *lim[redes.VARIABLES_P2[3]], step=0.1, key="p2_h")
        if st.button("⚡ Estimar potencia", type="primary"):
            mw, nivel = redes.estimar(res, t, v, p, h)
            st.session_state.setdefault("hist2", []).append(
                {"Escenario": f"Prueba {len(st.session_state.get('hist2', [])) + 1}", "Temp. (°C)": t,
                 "Vacío (cm Hg)": v, "Presión (mbar)": p, "Humedad (%)": h, "Potencia (MW)": mw, "Nivel": nivel})
        if st.session_state.get("hist2"):
            u = st.session_state["hist2"][-1]
            c1, c2 = res["cortes"]
            tono = {"Alto": "verde", "Medio": "amarillo", "Bajo": "rojo"}[u["Nivel"]]
            izq, der = st.columns([1, 1.2], gap="large")
            with izq:
                st.markdown(es.veredicto("Potencia estimada", f"{u['Potencia (MW)']:.1f} MW",
                                         f"Nivel de generación: <b>{u['Nivel']}</b><br><span class='nota'>Bajo &lt; "
                                         f"{c1:.1f} MW · Alto &gt; {c2:.1f} MW (terciles de los datos históricos)</span>",
                                         tono), unsafe_allow_html=True)
            with der:
                fig, ax = plt.subplots(figsize=(6, 2.6))
                ax.hist(res["y_hist"], bins=40, color="#D0D5DD", edgecolor="white")
                ax.axvspan(res["y_hist"].min(), c1, color="#FEE4E2", alpha=0.6, zorder=0)
                ax.axvspan(c2, res["y_hist"].max(), color="#DCFAE6", alpha=0.6, zorder=0)
                ax.axvline(u["Potencia (MW)"], color=es.COLOR[2], lw=2.5, label="Tu estimación")
                ax.set(title="Tu estimación frente a la producción histórica", xlabel="Potencia (MW)", yticks=[])
                ax.legend(loc="upper left"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        historial("hist2", {"Potencia (MW)": "{:.1f}", "Temp. (°C)": "{:.1f}", "Vacío (cm Hg)": "{:.1f}",
                            "Presión (mbar)": "{:.1f}", "Humedad (%)": "{:.1f}"}, "Escenarios estimados",
                  ("Nivel", lambda n: es.VERDE if n == "Alto" else es.AMARILLO if n == "Medio" else es.ROJO))


# ------------------------------------------------------------------ 3. Series de tiempo
def ciclo_al_azar():
    res = modelo(3)
    st.session_state.p3_ciclo = int(np.random.choice(res["c_ev"]) + 1)


def red_3():
    encabezado(3)
    t1, t2, t3 = st.tabs(["① Entrenar", "② Resultados", "③ Probar la red"])
    with t1:
        pestana_entrenar(3)
    res = modelo(3)
    nombres = redes.ESTADOS
    cortos = ["100 %", "90 %", "80 %", "73 %"]
    tiempo = np.arange(600) / 10
    with t2:
        if not res:
            sin_entrenar()
        else:
            rep = res["reporte"]
            c = st.columns(4)
            c[0].metric("Exactitud", f"{rep['accuracy']:.3f}")
            c[1].metric("F1 promedio", f"{rep['macro avg']['f1-score']:.3f}")
            c[2].metric("Errores", f"{int((res['y_pred'] != res['y_ev']).sum())} de {len(res['y_ev'])}")
            c[3].metric("Épocas", res["epocas"])
            st.caption(REDES[3]["referencia"])
            clases = (pd.DataFrame(rep).T.drop("accuracy")
                      .rename(index={"macro avg": "Promedio macro", "weighted avg": "Promedio ponderado"})
                      .rename_axis("Estado de la válvula").reset_index()
                      .rename(columns={"precision": "Precisión", "recall": "Recall", "f1-score": "F1-score",
                                       "support": "Ciclos"}))
            matriz = pd.DataFrame(res["mc"], columns=[f"Pred. {c}" for c in cortos])
            matriz.insert(0, "Estado real", nombres)
            a, b = st.columns([1.15, 1], gap="large")
            with a:
                mostrar(es.tabla(clases, "Métricas por estado (evaluación)")
                        .format({"Precisión": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "Ciclos": "{:.0f}"}))
            with b:
                mostrar(es.tabla(matriz, "Matriz de confusión")
                        .apply(es.colores_matriz, axis=None, subset=list(matriz.columns[1:])))
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
            curvas(ax[0], ax[1], res["historial"], "accuracy", "Exactitud")
            for k in range(len(nombres)):
                ax[2].plot(tiempo, res["ejemplos"][k], lw=1, label=cortos[k])
            ax[2].set(title="Serie de tiempo: presión PS1 en un ciclo", xlabel="Tiempo (s)", ylabel="Presión (bar)")
            ax[2].legend(title="Válvula", fontsize=7)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.markdown("La exactitud se mantiene baja durante las primeras épocas y luego sube de golpe cuando la "
                        "red descubre los patrones temporales, un comportamiento típico de las redes LSTM. Entrenamiento "
                        "y validación terminan casi iguales, por lo que no hay sobreajuste.")
    with t3:
        if not res:
            sin_entrenar()
            return
        st.markdown("Elige un ciclo de operación del conjunto de evaluación (la red nunca lo vio al entrenar) y "
                    "pulsa **Diagnosticar ciclo**. La red analizará sus 60 segundos de señales.")
        ciclos = sorted(int(c) + 1 for c in res["c_ev"])
        st.session_state.setdefault("p3_ciclo", ciclos[0])
        a, b = st.columns([3, 1], gap="medium")
        ciclo = a.selectbox("Número de ciclo", ciclos, key="p3_ciclo")
        b.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        b.button("🎲 Ciclo al azar", on_click=ciclo_al_azar)
        i = int(np.where(res["c_ev"] == ciclo - 1)[0][0])
        crudo = res["X_ev_crudo"][i]
        fig, ax = plt.subplots(1, 2, figsize=(13, 2.8))
        ax[0].plot(tiempo, crudo[:, 0], color=es.COLOR[3], lw=1)
        ax[0].set(title=f"Ciclo {ciclo}: presión PS1", xlabel="Tiempo (s)", ylabel="bar")
        ax[1].plot(tiempo, crudo[:, 7], color="#7A5AF8", lw=1)
        ax[1].set(title=f"Ciclo {ciclo}: flujo FS1", xlabel="Tiempo (s)", ylabel="l/min")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        if st.button("🔍 Diagnosticar ciclo", type="primary"):
            prob = redes.diagnosticar_ciclo(res, i)
            pred, real = int(prob.argmax()), int(res["y_ev"][i])
            st.session_state.setdefault("hist3", []).append(
                {"Ciclo": ciclo, "Estado real": nombres[real], "Diagnóstico de la red": nombres[pred],
                 "Confianza": float(prob.max()), "Resultado": "CORRECTO" if pred == real else "INCORRECTO",
                 "_prob": prob})
        if st.session_state.get("hist3"):
            u = st.session_state["hist3"][-1]
            ok = u["Resultado"] == "CORRECTO"
            izq, der = st.columns([1, 1.2], gap="large")
            with izq:
                st.markdown(es.veredicto(f"Ciclo {u['Ciclo']} · diagnóstico de la red", u["Diagnóstico de la red"],
                                         f"Confianza: <b>{u['Confianza']:.1%}</b><br>Estado real: {u['Estado real']} "
                                         f"→ <b>{u['Resultado']}</b>", "verde" if ok else "rojo"),
                            unsafe_allow_html=True)
            with der:
                st.markdown("**Probabilidad de cada estado**")
                for nombre, pr in zip(nombres, u["_prob"]):
                    st.markdown(f"<span class='nota'>{nombre} — {pr:.1%}</span>" + es.barra(float(pr), None, es.COLOR[3]),
                                unsafe_allow_html=True)
        filas = [{k: v for k, v in f.items() if k != "_prob"} for f in st.session_state.get("hist3", [])]
        if filas:
            mostrar(es.tabla(pd.DataFrame(filas), "Ciclos diagnosticados").format({"Confianza": "{:.1%}"})
                    .map(lambda v: es.VERDE if v == "CORRECTO" else es.ROJO, subset=["Resultado"]))
            if st.button("Limpiar historial", key="limpiar_hist3"):
                st.session_state["hist3"] = []
                st.rerun()


# ------------------------------------------------------------------ Navegación
{"Inicio": pagina_inicio, REDES[1]["menu"]: red_1, REDES[2]["menu"]: red_2, REDES[3]["menu"]: red_3}[pagina]()
