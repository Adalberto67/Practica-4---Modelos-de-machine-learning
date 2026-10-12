"""Estilos visuales de la página: CSS, tablas con formato y tarjetas de resultado."""
import matplotlib.pyplot as plt
import pandas as pd

AZUL, AZUL_OSC = "#2E5AA8", "#1F3B73"
COLOR = {1: "#2E5AA8", 2: "#C2570C", 3: "#0F766E"}          # Un color por red neuronal
VERDE = "background-color:#D1E7DD !important; color:#0F5132 !important; font-weight:700"
AMARILLO = "background-color:#FFF3CD !important; color:#664D03 !important; font-weight:700"
ROJO = "background-color:#F8D7DA !important; color:#842029 !important; font-weight:700"

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.portada {background: linear-gradient(120deg, #1F3B73 0%, #2E5AA8 60%, #3E7BD1 100%);
  color: #fff; border-radius: 18px; padding: 34px 38px; margin-bottom: 26px;}
.portada .etq {font-size: 0.8rem; letter-spacing: .14em; text-transform: uppercase; opacity: .8;}
.portada h1 {color: #fff; font-size: 2.3rem; margin: 6px 0 10px 0; line-height: 1.15;}
.portada p {font-size: 1.05rem; opacity: .92; margin: 0; max-width: 760px;}
.tarjeta {background: #fff; border: 1px solid #DCE3EF; border-radius: 14px; padding: 20px 22px;
  height: 100%; box-shadow: 0 1px 2px rgba(16,24,40,.04);}
.tarjeta .num {display: inline-block; width: 30px; height: 30px; border-radius: 50%; color: #fff;
  text-align: center; line-height: 30px; font-weight: 700; margin-right: 8px;}
.tarjeta h4 {display: inline; font-size: 1.05rem;}
.tarjeta p {color: #475467; font-size: .93rem; margin: 10px 0 0 0;}
.chip {display: inline-block; font-size: .75rem; font-weight: 700; letter-spacing: .05em;
  padding: 3px 10px; border-radius: 999px; text-transform: uppercase; margin-bottom: 6px;}
.paso {display: flex; gap: 14px; align-items: flex-start; margin: 8px 0;}
.paso b.n {flex: 0 0 34px; height: 34px; border-radius: 10px; background: #E9EEF8; color: #1F3B73;
  display: flex; align-items: center; justify-content: center; font-size: 1rem;}
.paso div {color: #344054;}
.veredicto {border-radius: 14px; padding: 18px 22px; margin: 6px 0 12px 0; border: 1px solid;}
.veredicto .t {font-size: .8rem; text-transform: uppercase; letter-spacing: .1em; opacity: .8;}
.veredicto .v {font-size: 1.9rem; font-weight: 800; margin: 2px 0;}
.barra {height: 12px; background: #E4E7EC; border-radius: 999px; position: relative; margin: 10px 0 4px 0;}
.barra .lleno {height: 12px; border-radius: 999px;}
.barra .marca {position: absolute; top: -5px; width: 3px; height: 22px; background: #101828;}
.nota {font-size: .82rem; color: #667085;}
</style>
"""


def tabla(df, titulo):
    """Tabla con título, encabezados y bordes integrados (mismo diseño que en Colab)."""
    return (df.style.hide(axis="index").set_caption(titulo).set_table_styles([
        {"selector": "", "props": "border-collapse:collapse; border:2px solid #1F3B73; font-family:sans-serif; "
                                  "font-size:14px; margin:6px 0 18px 0; width:100%"},
        {"selector": "caption", "props": "caption-side:top; background:#1F3B73; color:white; font-size:15px; "
                                         "font-weight:bold; padding:8px; text-align:center"},
        {"selector": "th", "props": "background:#2E5AA8; color:white; font-weight:bold; padding:7px 12px; "
                                    "border:1px solid #1F3B73; text-align:center"},
        {"selector": "td", "props": "padding:7px 12px; border:1px solid #9BB3D9; text-align:center; color:#101828"},
        {"selector": "tbody tr:nth-child(odd) td", "props": "background:white"},
        {"selector": "tbody tr:nth-child(even) td", "props": "background:#EEF3FB"},
        {"selector": "td:first-child", "props": "font-weight:bold; text-align:left"},
    ]))


def html(styler):
    return styler.to_html()


def colores_matriz(df):
    """Verde = aciertos (diagonal); rojo = errores."""
    return pd.DataFrame([[VERDE if i == j else ROJO if df.iat[i, j] > 0 else ""
                          for j in range(df.shape[1])] for i in range(df.shape[0])],
                        index=df.index, columns=df.columns)


def veredicto(titulo, valor, detalle, tono):
    """Tarjeta grande con el resultado de la red (tono: verde, amarillo o rojo)."""
    paleta = {"verde": ("#ECFDF3", "#067647", "#ABEFC6"), "amarillo": ("#FFFAEB", "#B54708", "#FEDF89"),
              "rojo": ("#FEF3F2", "#B42318", "#FECDCA"), "azul": ("#EFF4FF", "#1F3B73", "#B2CCFF")}
    fondo, texto, borde = paleta[tono]
    return (f'<div class="veredicto" style="background:{fondo}; color:{texto}; border-color:{borde}">'
            f'<div class="t">{titulo}</div><div class="v">{valor}</div><div>{detalle}</div></div>')


def barra(valor, marca=None, color="#2E5AA8"):
    """Barra de 0 a 1 con una marca opcional (por ejemplo, el umbral de decisión)."""
    m = f'<div class="marca" style="left:calc({marca * 100:.1f}% - 1px)"></div>' if marca is not None else ""
    return (f'<div class="barra"><div class="lleno" style="width:{valor * 100:.1f}%; background:{color}"></div>'
            f'{m}</div>')


def estilo_graficas():
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
                         "axes.titlesize": 11, "font.size": 9, "legend.frameon": False})
