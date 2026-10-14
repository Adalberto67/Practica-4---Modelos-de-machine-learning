"""
Práctica 4 · Modelos de machine learning — Página interactiva (Streamlit)
Alumno: Adalberto Enai Pech Us · Sistemas Inteligentes

Tres redes neuronales en una sola página: entrenarlas, ver sus resultados y probarlas con valores
elegidos por el usuario. La lógica de las redes está en redes.py y el diseño en estilo.py.
Ejecutar localmente:  streamlit run app.py
"""
import threading
import time

import numpy as np
import pandas as pd
import streamlit as st
import estilo as es
import graficas as gr
import redes

st.set_page_config(page_title="Práctica 4 · Redes neuronales", layout="wide")
st.markdown(es.CSS, unsafe_allow_html=True)


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
            referencia="En Colab: ROC-AUC = 0.975 · exactitud = 0.972",
            colab="https://colab.research.google.com/drive/1R-M4y-RtuAnrMwkMBOvXB1d4wmoI7CUd?usp=sharing"),
    2: dict(menu="2 · Regresión", tipo="Ajuste de funciones · Regresión",
            titulo="¿Cuánta energía generará la planta?", funcion=redes.entrenar_regresion, tiempo="1 a 3 minutos",
            problema="Planta eléctrica de ciclo combinado. A partir de las condiciones ambientales, la red estima "
                     "la potencia eléctrica (MW) que generará la planta.",
            dataset="Combined Cycle Power Plant · UCI (9,568 registros horarios reales)",
            url="https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant",
            red="Perceptrón multicapa 4 → 64 → 64 → 1 (salida lineal)",
            referencia="En Colab: R² = 0.945 · error medio = 2.90 MW",
            colab="https://colab.research.google.com/drive/1zwuJx0JELanmcuH73oSe00h5DZCK-xvk?usp=sharing"),
    3: dict(menu="3 · Series de tiempo", tipo="Series de tiempo · Clasificación de secuencias",
            titulo="¿En qué estado está la válvula?", funcion=redes.entrenar_series, tiempo="3 a 8 minutos",
            problema="Monitoreo de condición de un sistema hidráulico. La red analiza cómo evolucionan 9 sensores "
                     "durante un ciclo de trabajo de 60 segundos e identifica el estado de la válvula de control.",
            dataset="Condition Monitoring of Hydraulic Systems · UCI (2,205 ciclos de un banco de pruebas real)",
            url="https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems",
            red="Convolucional 1D (32-64-64) + LSTM (64) → 4 estados (softmax)",
            referencia="En Colab: exactitud = 0.988 · F1 promedio = 0.985",
            colab="https://colab.research.google.com/drive/14Bjxd9KFZI_FMcAOhLguCYrVanAJawsy?usp=sharing"),
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
    st.markdown("### Práctica 4")
    st.caption("Modelos de Machine Learning")
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
                <p class="nota">Datos: {r['dataset']}</p>
                <p><a href="{r['colab']}" target="_blank">Abrir cuaderno en Colab ↗</a></p></div>""",
                        unsafe_allow_html=True)

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
    izq, der = st.columns([4, 1.3], gap="large", vertical_alignment="center")
    with izq:
        st.markdown(f'<span class="chip" style="background:{es.COLOR[k]}1A; color:{es.COLOR[k]}">{r["tipo"]}</span>',
                    unsafe_allow_html=True)
        st.markdown(f"## {r['titulo']}")
        st.markdown(r["problema"])
    with der:
        st.link_button("Abrir cuaderno en Colab ↗", r["colab"])


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
    st.warning("Primero entrena la red en la pestaña **Entrenar**.")


def graficas(*charts):
    """Muestra gráficas interactivas lado a lado."""
    for col, ch in zip(st.columns(len(charts), gap="medium"), charts):
        with col:
            st.altair_chart(ch, use_container_width=True)


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


ALERTA = 0.5      # A partir de aquí se recomienda revisar la máquina aunque no llegue al umbral de falla


def nivel_riesgo(p, umbral):
    """Nivel de riesgo: (diagnóstico, tono de la tarjeta, acción recomendada, color de la barra)."""
    if p >= umbral:
        return "FALLA", "rojo", "detener la máquina y darle mantenimiento", "#B42318"
    if p >= min(ALERTA, umbral):
        return "REVISAR", "amarillo", "riesgo moderado: programar una inspección preventiva", "#DC6803"
    return "SIN FALLA", "verde", "la máquina puede operar con normalidad", "#067647"


def cargar_ej1():
    if st.session_state.ej1 in EJ1:
        for c, v in zip(CAMPOS1, EJ1[st.session_state.ej1]):
            st.session_state[c] = v


def red_1():
    encabezado(1)
    t1, t2, t3 = st.tabs(["Entrenar", "Resultados", "Probar la red"])
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
            st.markdown("##### Gráficas")
            st.caption("Pasa el cursor sobre las gráficas para ver los valores; usa la rueda del mouse para hacer zoom "
                       "y haz clic en la leyenda para resaltar una curva.")
            graficas(gr.curvas(res["historial"], "loss", "Pérdida", es.COLOR[1]),
                     gr.curvas(res["historial"], "auc", "AUC", es.COLOR[1]))
            graficas(gr.roc(res["y_ev"], res["p_ev"], res["auc"], es.COLOR[1]),
                     gr.histograma(res["p_ev"][res["y_ev"] == 1], "Probabilidad asignada", "#B42318",
                                   "Probabilidad que dio la red a las fallas reales"))
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
                 "Prob. de falla": p, "Diagnóstico": nivel_riesgo(p, res["umbral"])[0]})
        if st.session_state.get("hist1"):
            u = st.session_state["hist1"][-1]
            nivel, tono, accion, color = nivel_riesgo(u["Prob. de falla"], res["umbral"])
            st.markdown(es.veredicto("Diagnóstico de la red", nivel,
                                     f"Probabilidad de falla: <b>{u['Prob. de falla']:.1%}</b> · {accion}", tono)
                        + es.barra(u["Prob. de falla"], [ALERTA, res["umbral"]], color), unsafe_allow_html=True)
        with st.expander("¿Cómo se interpreta la probabilidad de falla?"):
            st.markdown(
                f"- **SIN FALLA** (menos de {ALERTA:.0%}): la máquina puede seguir operando con normalidad.\n"
                f"- **REVISAR** ({ALERTA:.0%} a {res['umbral']:.0%}): hay señales de riesgo; conviene programar una "
                "inspección preventiva, aunque la red todavía no la clasifica como falla.\n"
                f"- **FALLA** ({res['umbral']:.0%} o más): la red predice una falla; se recomienda detener la máquina y "
                "darle mantenimiento.\n\n"
                f"El umbral de falla ({res['umbral']:.0%}) no se eligió a mano: el algoritmo probó varios valores con los "
                "datos de validación y se quedó con el que da el mejor equilibrio entre detectar fallas y evitar falsas "
                "alarmas (máximo F1). Es alto porque durante el entrenamiento se dio más peso a las fallas (son solo el "
                "3 % de los datos), lo que hace que la red asigne probabilidades altas con facilidad; por eso un 80 % "
                "no equivale a un 80 % de probabilidad real de falla, sino a una señal de riesgo que merece revisión.")
        historial("hist1", {"Prob. de falla": "{:.1%}", "Aire (K)": "{:.1f}", "Proceso (K)": "{:.1f}",
                            "Torque (Nm)": "{:.1f}"}, "Máquinas diagnosticadas",
                  ("Diagnóstico", lambda v: es.ROJO if v == "FALLA" else es.AMARILLO if v == "REVISAR" else es.VERDE))


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
    t1, t2, t3 = st.tabs(["Entrenar", "Resultados", "Probar la red"])
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
            st.markdown("##### Gráficas")
            st.caption("Pasa el cursor sobre las gráficas para ver los valores; usa la rueda del mouse para hacer zoom "
                       "y haz clic en la leyenda para resaltar una curva.")
            graficas(gr.curvas(res["historial"], "loss", "Pérdida (MSE)", es.COLOR[2]),
                     gr.dispersion(y_ev, p_ev, es.COLOR[2]))
            graficas(gr.histograma(p_ev - y_ev, "Error (MW)", es.COLOR[2], "Distribución de errores"),
                     gr.curvas(res["historial"], "mae", "Error absoluto (normalizado)", es.COLOR[2]))
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
                st.altair_chart(gr.estimacion(res["y_hist"], u["Potencia (MW)"], res["cortes"], es.COLOR[2]),
                                use_container_width=True)
        historial("hist2", {"Potencia (MW)": "{:.1f}", "Temp. (°C)": "{:.1f}", "Vacío (cm Hg)": "{:.1f}",
                            "Presión (mbar)": "{:.1f}", "Humedad (%)": "{:.1f}"}, "Escenarios estimados",
                  ("Nivel", lambda n: es.VERDE if n == "Alto" else es.AMARILLO if n == "Medio" else es.ROJO))


# ------------------------------------------------------------------ 3. Series de tiempo
def ciclo_al_azar():
    res = modelo(3)
    st.session_state.p3_ciclo = int(np.random.choice(res["c_ev"]) + 1)


def red_3():
    encabezado(3)
    t1, t2, t3 = st.tabs(["Entrenar", "Resultados", "Probar la red"])
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
            st.markdown("##### Gráficas")
            st.caption("Pasa el cursor sobre las gráficas para ver los valores; usa la rueda del mouse para hacer zoom "
                       "y haz clic en la leyenda para resaltar una curva.")
            graficas(gr.curvas(res["historial"], "loss", "Pérdida", es.COLOR[3]),
                     gr.curvas(res["historial"], "accuracy", "Exactitud", es.COLOR[3]))
            st.altair_chart(gr.senales(tiempo, {f"Válvula {c}": res["ejemplos"][k] for k, c in enumerate(cortos)},
                                       "Presión (bar)", "Serie de tiempo: presión PS1 durante un ciclo de cada estado",
                                       ["#12B76A", "#2E90FA", "#F79009", "#F04438"]), use_container_width=True)
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
        graficas(gr.senales(tiempo, {f"PS{j + 1}": crudo[:, j] for j in range(6)}, "Presión (bar)",
                            f"Ciclo {ciclo}: presión (6 sensores)"),
                 gr.senales(tiempo, {"FS1": crudo[:, 7], "FS2": crudo[:, 8]}, "Flujo (l/min)",
                            f"Ciclo {ciclo}: flujo (2 sensores)", ["#7A5AF8", "#EE46BC"]))
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
