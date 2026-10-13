"""Gráficas interactivas (Altair): se puede pasar el cursor para ver valores, hacer zoom y desplazarse."""
import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

GRIS = "#98A2B3"
ALTO = 260


def _base(ch, titulo):
    return (ch.properties(title=titulo, height=ALTO)
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#EAECF0", domainColor="#D0D5DD", labelColor="#475467", titleColor="#344054")
            .configure_title(fontSize=14, anchor="start", color="#101828")
            .configure_legend(orient="bottom", title=None, labelColor="#344054"))


def curvas(historial, clave, nombre, color):
    """Evolución de una métrica por época para entrenamiento y validación."""
    df = pd.DataFrame({"Época": np.arange(1, len(historial[clave]) + 1),
                       "Entrenamiento": historial[clave], "Validación": historial[f"val_{clave}"]})
    df = df.melt("Época", var_name="Conjunto", value_name=nombre)
    sel = alt.selection_point(fields=["Conjunto"], bind="legend")
    ch = (alt.Chart(df).mark_line(strokeWidth=2.2)
          .encode(x="Época:Q", y=alt.Y(f"{nombre}:Q", scale=alt.Scale(zero=False)),
                  color=alt.Color("Conjunto:N", scale=alt.Scale(range=[color, "#F79009"])),
                  opacity=alt.condition(sel, alt.value(1), alt.value(0.15)),
                  tooltip=["Época", "Conjunto", alt.Tooltip(f"{nombre}:Q", format=".4f")])
          .add_params(sel).interactive())
    return _base(ch, f"{nombre} por época")


def roc(y, p, auc, color):
    fpr, tpr, _ = roc_curve(y, p)
    df = pd.DataFrame({"Falsos positivos": fpr, "Verdaderos positivos": tpr})
    area = (alt.Chart(df).mark_area(line={"color": color}, color=color, opacity=0.15, interpolate="step-after")
            .encode(x="Falsos positivos:Q", y="Verdaderos positivos:Q",
                    tooltip=[alt.Tooltip("Falsos positivos:Q", format=".3f"),
                             alt.Tooltip("Verdaderos positivos:Q", format=".3f")]))
    azar = alt.Chart(pd.DataFrame({"Falsos positivos": [0, 1], "Verdaderos positivos": [0, 1]})) \
        .mark_line(strokeDash=[4, 4], color=GRIS).encode(x="Falsos positivos:Q", y="Verdaderos positivos:Q")
    return _base(area + azar, f"Curva ROC (AUC = {auc:.3f})")


def dispersion(real, pred, color, unidad="MW"):
    df = pd.DataFrame({f"Real ({unidad})": real, f"Predicho ({unidad})": pred, f"Error ({unidad})": pred - real})
    lim = [float(min(real.min(), pred.min())) - 3, float(max(real.max(), pred.max())) + 3]
    escala = alt.Scale(domain=lim)
    puntos = (alt.Chart(df).mark_circle(size=22, opacity=0.45, color=color)
              .encode(x=alt.X(f"Real ({unidad}):Q", scale=escala), y=alt.Y(f"Predicho ({unidad}):Q", scale=escala),
                      tooltip=[alt.Tooltip(c, format=".2f") for c in df.columns]).interactive())
    diag = alt.Chart(pd.DataFrame({f"Real ({unidad})": lim, f"Predicho ({unidad})": lim})) \
        .mark_line(strokeDash=[5, 4], color="#101828") \
        .encode(x=alt.X(f"Real ({unidad}):Q", scale=escala), y=alt.Y(f"Predicho ({unidad}):Q", scale=escala))
    return _base(puntos + diag, "Real vs. predicho (línea = ajuste perfecto)")


def histograma(valores, nombre, color, titulo):
    df = pd.DataFrame({nombre: valores})
    ch = (alt.Chart(df).mark_bar(color=color, opacity=0.85)
          .encode(x=alt.X(f"{nombre}:Q", bin=alt.Bin(maxbins=35)), y=alt.Y("count()", title="Frecuencia"),
                  tooltip=[alt.Tooltip("count()", title="Frecuencia")]))
    cero = alt.Chart(pd.DataFrame({nombre: [0]})).mark_rule(strokeDash=[4, 4], color="#101828").encode(x=f"{nombre}:Q")
    return _base(ch + cero, titulo)


def senales(tiempo, series, unidad, titulo, colores=None):
    """Series de tiempo: una línea por cada elemento de `series` (dict nombre -> arreglo)."""
    df = pd.concat([pd.DataFrame({"Tiempo (s)": tiempo, unidad: v, "Serie": k}) for k, v in series.items()])
    sel = alt.selection_point(fields=["Serie"], bind="legend")
    color = alt.Color("Serie:N", scale=alt.Scale(range=colores) if colores else alt.Undefined,
                      sort=list(series))
    ch = (alt.Chart(df).mark_line(strokeWidth=1.4)
          .encode(x="Tiempo (s):Q", y=alt.Y(f"{unidad}:Q", scale=alt.Scale(zero=False)), color=color,
                  opacity=alt.condition(sel, alt.value(1), alt.value(0.12)),
                  tooltip=["Serie", alt.Tooltip("Tiempo (s):Q", format=".1f"), alt.Tooltip(f"{unidad}:Q", format=".2f")])
          .add_params(sel).interactive(bind_y=False))
    return _base(ch, titulo)


def estimacion(historico, mw, cortes, color):
    """Distribución histórica de la potencia con la estimación del usuario marcada."""
    df = pd.DataFrame({"Potencia (MW)": historico})
    lo, hi = float(historico.min()), float(historico.max())
    zonas = pd.DataFrame({"ini": [lo, cortes[0], cortes[1]], "fin": [cortes[0], cortes[1], hi],
                          "Nivel": ["Bajo", "Medio", "Alto"]})
    fondo = (alt.Chart(zonas).mark_rect(opacity=0.18)
             .encode(x=alt.X("ini:Q", title="Potencia (MW)"), x2="fin:Q",
                     color=alt.Color("Nivel:N", scale=alt.Scale(domain=["Bajo", "Medio", "Alto"],
                                                                range=["#F04438", "#F79009", "#12B76A"])),
                     tooltip=["Nivel"]))
    barras = (alt.Chart(df).mark_bar(color="#98A2B3")
              .encode(x=alt.X("Potencia (MW):Q", bin=alt.Bin(maxbins=40), title="Potencia (MW)"),
                      y=alt.Y("count()", title="Horas"),
                      tooltip=[alt.Tooltip("count()", title="Horas")]))
    linea = alt.Chart(pd.DataFrame({"x": [mw], "t": [f"Tu estimación: {mw:.1f} MW"]})) \
        .mark_rule(color=color, strokeWidth=3).encode(x=alt.X("x:Q", title="Potencia (MW)"), tooltip=["t"])
    return _base(fondo + barras + linea, "Tu estimación frente a la producción histórica")
