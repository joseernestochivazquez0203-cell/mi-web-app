import datetime
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf

# Configuración de la página en Streamlit
st.set_page_config(
    page_title="Simulador de Portafolios Internacionales & Riesgo Estocástico",
    page_icon="💼",
    layout="wide",
)

fecha_hoy_str = datetime.datetime.now(ZoneInfo("America/Mexico_City")).strftime(
    "%d/%m/%Y"
)

# ==============================================================================
# DISEÑO: PALETA CON SIGNIFICADO, TEMPLATE DE PLOTLY Y CSS
#   Azul = principal | Verde = ganancia | Rojo = riesgo | Gris = referencia
#   Ámbar = destacado (moda / mínima varianza)
# ==============================================================================
AZUL = "#1F4E79"
VERDE = "#2E9E6B"
ROJO = "#D64545"
GRIS = "#8A94A6"
GRIS_OSC = "#5B6B82"
AMBAR = "#F2A900"
FONDO_APP = "#ffffff"
COLORSCALE_AZUL = [[0.0, "#C9DAEA"], [1.0, AZUL]]  # Sharpe: azul claro -> oscuro

_tpl = go.layout.Template(pio.templates["plotly_white"])
_tpl.layout.update(
    colorway=[AZUL, VERDE, AMBAR, ROJO, GRIS, "#4C8FC7", "#8E6BBF", "#2BA3A3"],
    font=dict(family="Source Sans Pro, Segoe UI, Arial, sans-serif", size=13, color="#1F2937"),
    title=dict(font=dict(size=15, color="#1F2937"), x=0.0, xanchor="left"),
    xaxis=dict(gridcolor="#E8ECF2", linecolor="#CBD2DC", zeroline=False),
    yaxis=dict(gridcolor="#E8ECF2", linecolor="#CBD2DC", zeroline=False),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
pio.templates["fin_clean"] = _tpl
pio.templates.default = "fin_clean"

st.markdown(
    f"""
    <style>
    /* ---- Fondo general de la aplicación ---- */
    [data-testid="stAppViewContainer"] {{
        background-color: {FONDO_APP};
    }}
    [data-testid="stHeader"] {{
        background-color: transparent;
    }}

    /* ---- Tipografía y jerarquía ---- */
    h1 {{ font-size: 1.9rem !important; font-weight: 700 !important; color: {AZUL}; letter-spacing: -0.01em; }}
    h2 {{ font-size: 1.45rem !important; font-weight: 650 !important; color: {AZUL};
          padding-bottom: 0.35rem; border-bottom: 2px solid {AZUL}; }}
    h3 {{ font-size: 1.15rem !important; font-weight: 600 !important; color: {AZUL}; }}
    hr {{ margin: 2.2rem 0 !important; border-color: rgba(31, 78, 121, 0.22) !important; }}
    [data-testid="stCaptionContainer"] {{ color: {GRIS_OSC}; margin-top: -0.4rem; }}

    /* ---- Tarjetas ---- */
    .stock-card {{
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid rgba(31, 78, 121, 0.20);
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        margin-bottom: 5px;
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid #DCE3EC;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(31, 78, 121, 0.05);
        background-color: white;
    }}
    [data-testid="stMetric"] {{
        background-color: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(31, 78, 121, 0.18);
        border-left: 4px solid {AZUL};
        border-radius: 8px;
        padding: 10px 14px;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 12px;
        background-color: rgba(255, 255, 255, 0.55);
    }}

    /* ---- Etiquetas de sección ---- */
    .pais-label {{
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: {AZUL}; margin-bottom: 0.4rem;
    }}
    .sb-label {{
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em;
        text-transform: uppercase; color: {GRIS_OSC};
        margin: 1.1rem 0 0.2rem 0; padding-bottom: 0.2rem;
        border-bottom: 1px solid #DCE3EC;
    }}

    /* ---- Botón discreto (restablecer) ---- */
    .stButton > button {{
        font-size: 0.78rem; padding: 0.15rem 0.8rem; min-height: 0;
        color: {AZUL}; border: 1px solid rgba(31, 78, 121, 0.35);
        background: rgba(255, 255, 255, 0.65);
    }}
    .stButton > button:hover {{ color: {AZUL}; border-color: {AZUL}; background: white; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _sb_label(texto):
  """Título pequeño en mayúsculas para agrupar controles en la barra lateral."""
  st.sidebar.markdown(f'<div class="sb-label">{texto}</div>', unsafe_allow_html=True)


def estilizar_tabla(styler):
  """Estilo uniforme: filas alternadas suaves + tipografía compacta.
  (No modifica datos ni formatos numéricos.)"""

  def _zebra(df):
    # Matriz con la MISMA forma que el DataFrame (filas x columnas).
    estilos = np.full(df.shape, "", dtype=object)
    estilos[1::2, :] = "background-color: rgba(31, 78, 121, 0.05)"
    return pd.DataFrame(estilos, index=df.index, columns=df.columns)

  # Styler.apply es perezoso (falla al renderizar), así que se valida antes:
  # solo se aplica zebra si índice y columnas son únicos.
  datos = styler.data
  if datos.index.is_unique and datos.columns.is_unique:
    styler = styler.apply(_zebra, axis=None)
  return styler.set_properties(**{"font-size": "13px"})


st.title("Simulador Integral de Portafolios, Optimización y Riesgo Estocástico")
st.caption(
    "Modelo cuantitativo multianual: Markowitz, Frontera Eficiente, Simulación"
    " Monte Carlo, Bootstrapping (10,000 corridas), Stop-Loss individual y"
    " métricas de riesgo."
)

# ==============================================================================
# 1. BARRA LATERAL (SIDEBAR): PARÁMETROS GENERALES Y CAPITAL DINÁMICO
# ==============================================================================
st.sidebar.header("Configuración")
_sb_label("Capital y renta fija")

capital_total_input = st.sidebar.number_input(
    "Capital Total a Invertir (MXN):",
    min_value=10000.0,
    max_value=100000000.0,
    value=1000000.0,
    step=50000.0,
)

pct_renta_fija = (
    st.sidebar.slider(
        "% Asignado a Renta Fija (CETES / T-Bills):", 0, 80, 30, 5
    )
    / 100.0
)

_sb_label("Optimización")
n_simulaciones = st.sidebar.slider(
    "Número de Portafolios (Simulación Markowitz):", 1000, 10000, 5000, 500
)
n_corridas = st.sidebar.slider(
    "Corridas de Optimización (Sharpe):", 100, 1000, 500, 100
)

_sb_label("Riesgo · VaR & Cutoff")
alpha_nivel = (
    st.sidebar.slider(
        "Nivel de Significancia (Alpha %):", 1.0, 15.0, 5.0, 0.5
    )
    / 100.0
)

_sb_label("Proyección estocástica")
_hoy = datetime.date.today()

# Fecha de inicio editable (por defecto: hoy). Se inicializa en session_state
# una sola vez para no mezclar `value=` con el estado y evitar conflictos.
if "fecha_inicio_sim" not in st.session_state:
    st.session_state["fecha_inicio_sim"] = _hoy

fecha_inicio_sim = st.sidebar.date_input(
    "Fecha de Inicio de Simulación:",
    key="fecha_inicio_sim",
    min_value=datetime.date(2000, 1, 1),
    max_value=_hoy,
    help="Por defecto es hoy. Los históricos de 1, 2 y 3 años se calculan hacia atrás desde esta fecha.",
)

# Fecha final: se mantiene consistente con la fecha de inicio.
_min_fin = fecha_inicio_sim + datetime.timedelta(days=1)
_max_fin = fecha_inicio_sim + datetime.timedelta(days=365 * 10)

if "fecha_fin_sim" not in st.session_state:
    st.session_state["fecha_fin_sim"] = fecha_inicio_sim + datetime.timedelta(days=365)
else:
    # Si el usuario movió la fecha de inicio, desplazamos la fecha final el mismo
    # número de días para conservar el horizonte que ya había elegido.
    _inicio_prev = st.session_state.get("_fecha_inicio_prev", fecha_inicio_sim)
    if _inicio_prev != fecha_inicio_sim:
        st.session_state["fecha_fin_sim"] += fecha_inicio_sim - _inicio_prev
    # Blindaje final: el valor siempre debe caer dentro de [min_value, max_value].
    st.session_state["fecha_fin_sim"] = min(
        max(st.session_state["fecha_fin_sim"], _min_fin), _max_fin
    )
st.session_state["_fecha_inicio_prev"] = fecha_inicio_sim

fecha_fin_sim = st.sidebar.date_input(
    "Fecha Final de Simulación:",
    key="fecha_fin_sim",
    min_value=_min_fin,
    max_value=_max_fin,
)

dias_totales = (fecha_fin_sim - fecha_inicio_sim).days
dias_habiles = max(1, int(dias_totales * (252 / 365)))
st.sidebar.info(
    f"📆 Días calendario: **{dias_totales}** | Días hábiles estimados:"
    f" **{dias_habiles}**"
)

# ==============================================================================
# 2. LISTAS DE EMPRESAS DISPONIBLES PARA CADA PAÍS (PARA LAS CAJITAS)
# ==============================================================================
opciones_mx = [
    "WALMEX.MX",
    "AMXB.MX",
    "FEMSAUBD.MX",
    "GFNORTEO.MX",
    "GMEXICOB.MX",
    "CEMEXCPO.MX",
    "BIMBOA.MX",
    "GAPB.MX",
    "ASURB.MX",
    "OMAB.MX",
    "AC.MX",
    "KOFUBL.MX",
    "KIMBERA.MX",
    "GFINBURO.MX",
    "GENTERA.MX",
    "PINFRA.MX",
    "ALFAA.MX",
    "CUERVO.MX",
    "LABB.MX",
    "CHDRAUIB.MX",
    "SORIANAB.MX",
    "LIVEPOLC-1.MX",
    "GCARSOA1.MX",
    "BBAJIOO.MX",
    "RA.MX",
    "FUNO11.MX",
    "FMTY14.MX",
    "TERRA13.MX",
    "DANHOS13.MX",
    "MEGACPO.MX",
    "TLEVISACPO.MX",
    "VOLARA.MX",
    "ALSEA.MX",
    "GRUMAB.MX",
    "BOLSAA.MX",
    "VESTA.MX",
    "ORBIA.MX",
    "PE&OLES.MX",
    "GCC.MX",
    "SITESB-1.MX",
]

opciones_usa = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "BRK-B",
    "LLY",
    "JPM",
    "V",
    "MA",
    "XOM",
    "UNH",
    "WMT",
    "COST",
    "HD",
    "PG",
    "JNJ",
    "NFLX",
    "CRM",
    "AMD",
    "INTC",
    "ORCL",
    "ADBE",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "CVX",
    "MRK",
    "ABBV",
    "PEP",
    "KO",
    "TMO",
    "ACN",
    "MCD",
    "DIS",
    "ABT",
    "CSCO",
    "LIN",
    "PM",
    "TXN",
    "NEE",
    "DHR",
    "AMGN",
    "RTX",
    "CAT",
    "IBM",
    "QCOM",
    "SPGI",
    "LOW",
    "BA",
    "UBER",
    "PLTR",
    "COIN",
    "PYPL",
    "SBUX",
    "BKNG",
]

# ==============================================================================
# 3. SELECTORES TIPO "CAJITA" EN LA INTERFAZ PRINCIPAL (5 POR PAÍS)
#    Persistencia: st.session_state <-> archivo JSON local (sobrevive a recargas
#    y reinicios de la app).
# ==============================================================================
ARCHIVO_SELECCIONES = Path(__file__).with_name("selecciones_activos.json")
KEYS_MX = ["m1", "m2", "m3", "m4", "m5"]
KEYS_USA = ["u1", "u2", "u3", "u4", "u5"]
DEFAULTS_SEL = {
    **{k: opciones_mx[i] for i, k in enumerate(KEYS_MX)},
    **{k: opciones_usa[i] for i, k in enumerate(KEYS_USA)},
}
OPCIONES_POR_KEY = {
    **{k: opciones_mx for k in KEYS_MX},
    **{k: opciones_usa for k in KEYS_USA},
}


def _cargar_selecciones():
    """Lee el JSON y devuelve solo valores válidos (si falla, usa defaults)."""
    sel = dict(DEFAULTS_SEL)
    try:
        with open(ARCHIVO_SELECCIONES, "r", encoding="utf-8") as f:
            guardado = json.load(f)
        for k, v in guardado.items():
            if k in sel and v in OPCIONES_POR_KEY[k]:
                sel[k] = v
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        pass
    return sel


def _guardar_selecciones(sel):
    """Escritura atómica para no corromper el JSON si la app se cierra a medias."""
    try:
        tmp = ARCHIVO_SELECCIONES.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(sel, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ARCHIVO_SELECCIONES)
    except OSError:
        pass  # Si el disco es de solo lectura, la app sigue funcionando sin persistir.


def _restablecer_selecciones():
    """Callback del botón: corre ANTES de instanciar los widgets, por eso es
    seguro modificar st.session_state aquí sin conflictos."""
    for k, v in DEFAULTS_SEL.items():
        st.session_state[k] = v
    st.session_state["_sel_guardadas"] = dict(DEFAULTS_SEL)
    try:
        ARCHIVO_SELECCIONES.unlink(missing_ok=True)
    except OSError:
        pass


# Primera carga de la sesión: volcar lo guardado en session_state (una sola vez).
if "_sel_guardadas" not in st.session_state:
    _inicial = _cargar_selecciones()
    for k, v in _inicial.items():
        st.session_state[k] = v
    st.session_state["_sel_guardadas"] = dict(_inicial)

st.markdown("---")
st.subheader("Selección de Activos")
st.caption("Elige 5 activos por país. Tu selección se guarda automáticamente.")
col_sel_mx, col_sel_usa = st.columns(2)

# Nota: sin `index=` porque el valor lo controla st.session_state[key].
# Distribución 3 + 2 para que los tickers se lean completos.
with col_sel_mx:
  with st.container(border=True):
    st.markdown('<div class="pais-label">México · BMV</div>', unsafe_allow_html=True)
    _f1 = st.columns(3)
    _f2 = st.columns(3)
    with _f1[0]:
      t1 = st.selectbox("M1", opciones_mx, key="m1")
    with _f1[1]:
      t2 = st.selectbox("M2", opciones_mx, key="m2")
    with _f1[2]:
      t3 = st.selectbox("M3", opciones_mx, key="m3")
    with _f2[0]:
      t4 = st.selectbox("M4", opciones_mx, key="m4")
    with _f2[1]:
      t5 = st.selectbox("M5", opciones_mx, key="m5")
  tickers_mx_list = [t1, t2, t3, t4, t5]

with col_sel_usa:
  with st.container(border=True):
    st.markdown('<div class="pais-label">EE. UU. · NYSE / NASDAQ</div>', unsafe_allow_html=True)
    _g1 = st.columns(3)
    _g2 = st.columns(3)
    with _g1[0]:
      u1 = st.selectbox("U1", opciones_usa, key="u1")
    with _g1[1]:
      u2 = st.selectbox("U2", opciones_usa, key="u2")
    with _g1[2]:
      u3 = st.selectbox("U3", opciones_usa, key="u3")
    with _g2[0]:
      u4 = st.selectbox("U4", opciones_usa, key="u4")
    with _g2[1]:
      u5 = st.selectbox("U5", opciones_usa, key="u5")
  tickers_usa_list = [u1, u2, u3, u4, u5]

# Guardar solo si cambió algo respecto a lo último persistido.
_seleccion_actual = {k: st.session_state[k] for k in DEFAULTS_SEL}
if _seleccion_actual != st.session_state["_sel_guardadas"]:
    _guardar_selecciones(_seleccion_actual)
    st.session_state["_sel_guardadas"] = dict(_seleccion_actual)

_c_espacio, _c_reset = st.columns([5, 2])
with _c_reset:
  st.button(
      "↺ Restablecer selecciones por defecto",
      on_click=_restablecer_selecciones,
      help="Vuelve a las primeras 5 empresas de cada país y borra lo guardado.",
      use_container_width=True,
  )

tickers_mx = {t.split(".")[0]: t for t in tickers_mx_list}
empresas_mx = list(tickers_mx.keys())

tickers_usa = {t: t for t in tickers_usa_list}
empresas_usa = list(tickers_usa.keys())
st.markdown("---")

fechas_horizontes = {
    "1 año": {
        "inicio": (fecha_inicio_sim - datetime.timedelta(days=365)).strftime(
            "%Y-%m-%d"
        ),
        "fin": fecha_inicio_sim.strftime("%Y-%m-%d"),
    },
    "2 años": {
        "inicio": (fecha_inicio_sim - datetime.timedelta(days=730)).strftime(
            "%Y-%m-%d"
        ),
        "fin": fecha_inicio_sim.strftime("%Y-%m-%d"),
    },
    "3 años": {
        "inicio": (fecha_inicio_sim - datetime.timedelta(days=1095)).strftime(
            "%Y-%m-%d"
        ),
        "fin": fecha_inicio_sim.strftime("%Y-%m-%d"),
    },
}


# ==============================================================================
# 4. FUNCIÓN SEGURA PARA OBTENER PRECIOS ACTUALES
# ==============================================================================
def obtener_precio_actual(ticker, default_val=100.0):
  try:
    df_p = yf.download(ticker, period="5d", progress=False)["Close"]
    if isinstance(df_p, pd.DataFrame):
      df_p = df_p.iloc[:, 0]
    val = float(df_p.dropna().iloc[-1])
    return val if not np.isnan(val) else default_val
  except Exception:
    return default_val


# ==============================================================================
# 5. FUNCIÓN DE SIMULACIÓN Y OPTIMIZACIÓN (MARKOWITZ)
# ==============================================================================
@st.cache_data
def ejecutar_simulacion_mercado(
    tickers_dict,
    lista_empresas,
    start,
    end,
    n_sim,
    n_corr,
    tipo_tasa="cetes",
    pct_rf=0.30,
):
  raw_data = yf.download(
      list(tickers_dict.values()),
      start=start,
      end=end,
      progress=False,
      auto_adjust=True,
  )

  if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
    if isinstance(raw_data.columns, pd.MultiIndex):
      if "Close" in raw_data.columns.get_level_values(0):
        datos = raw_data["Close"].copy()
      else:
        datos = raw_data.xs("Close", axis=1, level=-1, drop_level=True).copy()
    else:
      datos = (
          raw_data["Close"].to_frame()
          if "Close" in raw_data.columns
          else raw_data.copy()
      )
  else:
    datos = pd.DataFrame()

  if not datos.empty:
    mapa_inverso = {v: k for k, v in tickers_dict.items()}
    datos.rename(columns=mapa_inverso, inplace=True)

    cols_presentes = [c for c in lista_empresas if c in datos.columns]
    datos = datos[cols_presentes].sort_index()

    # Rellenar huecos puntuales (días inhábiles asimétricos en BMV) antes de limpiar
    datos = datos.ffill().bfill().dropna(how="all")
    # Eliminar únicamente columnas que vengan completamente vacías
    datos = datos.dropna(axis=1, how="all")

  if datos.empty or datos.shape[1] == 0:
    datos = pd.DataFrame(
        np.random.normal(100, 2, size=(100, len(lista_empresas))),
        columns=lista_empresas,
    )

  columnas_activas = list(datos.columns)
  rendimientos = np.log(datos / datos.shift(1)).dropna()
  promedio = rendimientos.mean()
  dias_ano = 252

  cov_df = rendimientos[columnas_activas].cov(ddof=0) * dias_ano
  Sigma = cov_df.values

  if tipo_tasa == "cetes":
    a, b, c = 0.04, 0.08, 0.10
  else:
    a, b, c = 0.0360, 0.0370, 0.0400

  k = (b - a) / (c - a)
  mejores_sharpe = []
  datos_animacion = []
  n_animacion_pasos = min(50, n_corr)
  # Un color distinto por iteración (paleta multicolor)
  colores_anim = px.colors.sample_colorscale(
      "Turbo", np.linspace(0.05, 0.95, n_animacion_pasos)
  )

  w_rf = float(pct_rf)
  w_portafolio = 1.0 - w_rf

  min_port = None
  min_desp_val = float("inf")

  for i in range(1, n_corr + 1):
    p_tasa = np.random.uniform(0, 1)
    if p_tasa < k:
      TL = a + np.sqrt(p_tasa * (b - a) * (c - a))
    else:
      TL = c - np.sqrt((1 - p_tasa) * (c - a) * (c - b))

    diario = (1 + TL) ** (1 / 360) - 1
    tasa_anual = TL

    aleatorios_sim = np.random.uniform(
        0, 1, size=(n_sim, len(columnas_activas))
    )
    wi_sim = aleatorios_sim / aleatorios_sim.sum(axis=1, keepdims=True)
    df_sim = pd.DataFrame(wi_sim, columns=columnas_activas)

    df_sim["Rp"] = (
        df_sim[columnas_activas].dot(promedio[columnas_activas]) * dias_ano
    )
    W_sim = df_sim[columnas_activas].values
    vp_sim = np.einsum("ij,jk,ik->i", W_sim, Sigma, W_sim)
    df_sim["vp"] = vp_sim
    df_sim["des_p"] = np.sqrt(vp_sim)
    df_sim["sharpe_ratio"] = (df_sim["Rp"] - tasa_anual) / df_sim["des_p"]

    idx_min_local = df_sim["des_p"].idxmin()
    val_min_local = df_sim.loc[idx_min_local, "des_p"]
    if val_min_local < min_desp_val:
      min_desp_val = val_min_local
      min_port = df_sim.loc[[idx_min_local]].copy()

    idx_max_sharpe = df_sim["sharpe_ratio"].idxmax()
    mejor_fila = df_sim.loc[[idx_max_sharpe]].copy()
    mejor_fila.index = [f"Corrida {i}"]
    mejor_fila["Tasa_diaria"] = diario
    mejor_fila["Tasa_anual"] = tasa_anual
    mejores_sharpe.append(mejor_fila)

    if i <= n_animacion_pasos:
      idx_opt = idx_max_sharpe
      riesgo_opt_anual = df_sim.loc[idx_opt, "des_p"]
      ret_opt_anual = df_sim.loc[idx_opt, "Rp"]
      max_sh_anual = df_sim.loc[idx_opt, "sharpe_ratio"]

      x_cal = np.linspace(0, riesgo_opt_anual * 1.25, 30)
      y_cal = tasa_anual + max_sh_anual * x_cal

      r_comb = w_portafolio * riesgo_opt_anual
      ret_comb = (w_rf * tasa_anual) + (w_portafolio * ret_opt_anual)

      df_vis = df_sim.sample(n=min(2500, len(df_sim))).copy()

      datos_animacion.append({
          "tasa_anual": tasa_anual,
          "nube_x": df_vis["des_p"].values,
          "nube_y": df_vis["Rp"].values,
          "x_cal": x_cal,
          "y_cal": y_cal,
          "star_x": riesgo_opt_anual,
          "star_y": ret_opt_anual,
          "star_sharpe": max_sh_anual,
          "mix_x": r_comb,
          "mix_y": ret_comb,
          "color": colores_anim[i - 1],
      })

  tabla_sharpe_max = pd.concat(mejores_sharpe)
  idx_max_global = tabla_sharpe_max["sharpe_ratio"].idxmax()
  max_sharpe_port = tabla_sharpe_max.loc[[idx_max_global]].copy()

  return (
      min_port,
      max_sharpe_port,
      datos_animacion,
      tabla_sharpe_max,
      datos,
      Sigma,
      columnas_activas,
  )


with st.spinner("Ejecutando simulaciones y optimizaciones de portafolio..."):
  res_mx_1 = ejecutar_simulacion_mercado(
      tickers_mx,
      empresas_mx,
      fechas_horizontes["1 año"]["inicio"],
      fechas_horizontes["1 año"]["fin"],
      n_simulaciones,
      n_corridas,
      "cetes",
      pct_renta_fija,
  )
  res_mx_2 = ejecutar_simulacion_mercado(
      tickers_mx,
      empresas_mx,
      fechas_horizontes["2 años"]["inicio"],
      fechas_horizontes["2 años"]["fin"],
      n_simulaciones,
      n_corridas,
      "cetes",
      pct_renta_fija,
  )
  res_mx_3 = ejecutar_simulacion_mercado(
      tickers_mx,
      empresas_mx,
      fechas_horizontes["3 años"]["inicio"],
      fechas_horizontes["3 años"]["fin"],
      n_simulaciones,
      n_corridas,
      "cetes",
      pct_renta_fija,
  )

  res_usa_1 = ejecutar_simulacion_mercado(
      tickers_usa,
      empresas_usa,
      fechas_horizontes["1 año"]["inicio"],
      fechas_horizontes["1 año"]["fin"],
      n_simulaciones,
      n_corridas,
      "tbills",
      pct_renta_fija,
  )
  res_usa_2 = ejecutar_simulacion_mercado(
      tickers_usa,
      empresas_usa,
      fechas_horizontes["2 años"]["inicio"],
      fechas_horizontes["2 años"]["fin"],
      n_simulaciones,
      n_corridas,
      "tbills",
      pct_renta_fija,
  )
  res_usa_3 = ejecutar_simulacion_mercado(
      tickers_usa,
      empresas_usa,
      fechas_horizontes["3 años"]["inicio"],
      fechas_horizontes["3 años"]["fin"],
      n_simulaciones,
      n_corridas,
      "tbills",
      pct_renta_fija,
  )

  (
      min_mx_1,
      max_mx_1,
      anim_mx_1,
      t1000_mx_1,
      datos_mx_1,
      cov_mx_1,
      cols_activas_mx_1,
  ) = res_mx_1
  (
      min_mx_2,
      max_mx_2,
      anim_mx_2,
      t1000_mx_2,
      datos_mx_2,
      cov_mx_2,
      cols_activas_mx_2,
  ) = res_mx_2
  (
      min_mx_3,
      max_mx_3,
      anim_mx_3,
      t1000_mx_3,
      datos_mx_3,
      cov_mx_3,
      cols_activas_mx_3,
  ) = res_mx_3

  (
      min_usa_1,
      max_usa_1,
      anim_usa_1,
      t1000_usa_1,
      datos_usa_1,
      cov_usa_1,
      cols_activas_usa_1,
  ) = res_usa_1
  (
      min_usa_2,
      max_usa_2,
      anim_usa_2,
      t1000_usa_2,
      datos_usa_2,
      cov_usa_2,
      cols_activas_usa_2,
  ) = res_usa_2
  (
      min_usa_3,
      max_usa_3,
      anim_usa_3,
      t1000_usa_3,
      datos_usa_3,
      cov_usa_3,
      cols_activas_usa_3,
  ) = res_usa_3

min_mx_1.index, min_mx_2.index, min_mx_3.index = (
    ["1 año"],
    ["2 años"],
    ["3 años"],
)
max_mx_1.index, max_mx_2.index, max_mx_3.index = (
    ["1 año"],
    ["2 años"],
    ["3 años"],
)
min_usa_1.index, min_usa_2.index, min_usa_3.index = (
    ["1 año"],
    ["2 años"],
    ["3 años"],
)
max_usa_1.index, max_usa_2.index, max_usa_3.index = (
    ["1 año"],
    ["2 años"],
    ["3 años"],
)


# ==============================================================================
# FUNCIÓN DINÁMICA DE ASIGNACIÓN DE CAPITAL
# ==============================================================================
def calcular_capitales_dinamicos(
    max_mx, max_usa, d_mx, d_usa, cap_total, pct_rf
):
  rend_h_mx = np.log(d_mx / d_mx.shift(1)).dropna()
  act_mx = [
      e for e in empresas_mx if e in max_mx.columns and e in rend_h_mx.columns
  ]
  s_mx_hist = (
      rend_h_mx[act_mx].dot(max_mx[act_mx].values[0])
      if len(act_mx) > 0
      else pd.Series(np.zeros(100))
  )

  rend_h_usa = np.log(d_usa / d_usa.shift(1)).dropna()
  act_usa = [
      e for e in empresas_usa if e in max_usa.columns and e in rend_h_usa.columns
  ]
  s_usa_hist = (
      rend_h_usa[act_usa].dot(max_usa[act_usa].values[0])
      if len(act_usa) > 0
      else pd.Series(np.zeros(100))
  )

  df_comb = pd.concat([s_mx_hist, s_usa_hist], axis=1).dropna()
  if len(df_comb) > 5:
    cov_mat = df_comb.cov().values * 252
    var_mx = cov_mat[0, 0]
    var_usa = cov_mat[1, 1]
    cov_mx_usa = cov_mat[0, 1]
    denom = var_mx + var_usa - 2 * cov_mx_usa
    if denom > 0:
      w_mx_din = np.clip((var_usa - cov_mx_usa) / denom, 0.15, 0.85)
    else:
      w_mx_din = 0.50
  else:
    w_mx_din = 0.50

  w_usa_din = 1.0 - w_mx_din

  cap_renta_variable = cap_total * (1.0 - pct_rf)
  cap_renta_fija = cap_total * pct_rf

  monto_mx_acc = cap_renta_variable * w_mx_din
  monto_usa_acc_mxn = cap_renta_variable * w_usa_din
  monto_cetes = cap_renta_fija * w_mx_din
  monto_tbills_mxn = cap_renta_fija * w_usa_din

  return (
      monto_mx_acc,
      monto_cetes,
      monto_usa_acc_mxn,
      monto_tbills_mxn,
      w_mx_din,
      w_usa_din,
  )


# ==============================================================================
# 6. CONSTRUCTOR DE GRÁFICA ANIMADA
# ==============================================================================
def construir_figura_animada_con_minima(
    datos_anim,
    min_port_df,
    lista_empresas,
    titulo_horizonte,
    lbl_tasa="CETES",
    is_mx=False,
    pct_rf=0.30,
):
  fig = go.Figure()
  n_pasos = len(datos_anim)
  pct_rf_num = pct_rf * 100 if pct_rf <= 1 else pct_rf
  pct_rv_num = 100 - pct_rf_num

  fig.add_trace(
      go.Scatter(
          x=[],
          y=[],
          mode="markers",
          marker=dict(
              size=5,
              color=datos_anim[0]["color"],
              opacity=0.65,
          ),
          name="Portafolios Simulados (color por iteración)",
      )
  )
  for j in range(n_pasos):
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color=datos_anim[j]["color"], width=1.4),
            opacity=0.75,
            showlegend=False,
        )
    )
  fig.add_trace(
      go.Scatter(
          x=[],
          y=[],
          mode="markers",
          marker=dict(
              size=10,
              color=AMBAR,
              symbol="star",
              line=dict(width=1, color="black"),
          ),
          name="Máximo Sharpe (Iteraciones)",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=[],
          y=[],
          mode="markers",
          marker=dict(
              size=15,
              color=ROJO,
              symbol="star",
              line=dict(width=1.5, color="black"),
          ),
          name="★ Máximo Sharpe Absoluto (Global)",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=[],
          y=[],
          mode="markers",
          marker=dict(
              size=8,
              color=VERDE,
              symbol="diamond",
              line=dict(width=1, color="black"),
          ),
          name=f"Mix {pct_rf_num:.0f}% {lbl_tasa} / {pct_rv_num:.0f}% Portafolio",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=[],
          y=[],
          mode="markers",
          marker=dict(size=8, color="darkred", symbol="circle"),
          name=lbl_tasa,
      )
  )

  min_x = min_port_df["des_p"].values[0]
  min_y = min_port_df["Rp"].values[0]
  fig.add_trace(
      go.Scatter(
          x=[min_x],
          y=[min_y],
          mode="markers",
          marker=dict(
              size=13,
              color="#111827",
              symbol="circle",
              line=dict(width=2, color="white"),
          ),
          name="Mínima Varianza",
      )
  )

  frames = []
  best_global_idx = max(
      range(len(datos_anim)), key=lambda idx: datos_anim[idx]["star_sharpe"]
  )

  for f_idx in range(n_pasos):
    f_data = []
    curr = datos_anim[f_idx]
    f_data.append(
        go.Scatter(
            x=curr["nube_x"],
            y=curr["nube_y"],
            mode="markers",
            marker=dict(size=5, color=curr["color"], opacity=0.65),
        )
    )
    for j in range(n_pasos):
      if j <= f_idx:
        f_data.append(
            go.Scatter(x=datos_anim[j]["x_cal"], y=datos_anim[j]["y_cal"])
        )
      else:
        f_data.append(go.Scatter(x=[], y=[]))

    stars_x, stars_y, stars_c = [], [], []
    for j in range(f_idx + 1):
      if j != best_global_idx:
        stars_x.append(datos_anim[j]["star_x"])
        stars_y.append(datos_anim[j]["star_y"])
        stars_c.append(datos_anim[j]["color"])
    f_data.append(go.Scatter(x=stars_x, y=stars_y, marker=dict(color=stars_c)))

    if f_idx >= best_global_idx:
      f_data.append(
          go.Scatter(
              x=[datos_anim[best_global_idx]["star_x"]],
              y=[datos_anim[best_global_idx]["star_y"]],
          )
      )
    else:
      f_data.append(go.Scatter(x=[], y=[]))

    f_data.append(
        go.Scatter(
            x=[datos_anim[j]["mix_x"] for j in range(f_idx + 1)],
            y=[datos_anim[j]["mix_y"] for j in range(f_idx + 1)],
            marker=dict(color=[datos_anim[j]["color"] for j in range(f_idx + 1)]),
        )
    )
    f_data.append(go.Scatter(x=[0], y=[curr["tasa_anual"]]))
    f_data.append(go.Scatter(x=[min_x], y=[min_y]))
    frames.append(go.Frame(data=f_data, name=f"Paso {f_idx + 1}"))

  for idx, t_data in enumerate(frames[0].data):
    fig.data[idx].x = t_data.x
    fig.data[idx].y = t_data.y
    _c0 = t_data.marker.color if t_data.marker is not None else None
    if _c0 is not None and len(_c0) > 0:  # no pisar el color base si el frame 0 viene vacío
      fig.data[idx].marker.color = _c0

  # Escala de ejes ajustada a los datos reales (antes: X fijo 0 a 1.2)
  _xs = np.concatenate(
      [np.asarray(d["nube_x"]) for d in datos_anim]
      + [np.array([d["star_x"], d["mix_x"], 0.0]) for d in datos_anim]
  )
  _ys = np.concatenate(
      [np.asarray(d["nube_y"]) for d in datos_anim]
      + [np.array([d["star_y"], d["mix_y"], d["tasa_anual"]]) for d in datos_anim]
  )
  _x_hi = float(np.nanmax(_xs)) * 1.08
  _x_lo = -0.02 * _x_hi  # margen para que CETES (x = 0) no quede cortado
  _y_pad = (float(np.nanmax(_ys)) - float(np.nanmin(_ys))) * 0.08
  _y_lo = float(np.nanmin(_ys)) - _y_pad
  _y_hi = float(np.nanmax(_ys)) + _y_pad

  fig.update_layout(
      title=f"Simulación Monte Carlo Paso a Paso ({titulo_horizonte})",
      template="fin_clean",
      height=500,
      xaxis=dict(
          title="Volatilidad (Riesgo Anual)",
          range=[_x_lo, _x_hi],
          tickformat=".0%",
      ),
      yaxis=dict(
          title="Rendimiento Anual",
          range=[_y_lo, _y_hi],
          tickformat=".0%",
      ),
      updatemenus=[{
          "type": "buttons",
          "buttons": [
              {
                  "label": "▶ Reproducir",
                  "method": "animate",
                  "args": [
                      None,
                      {
                          "frame": {"duration": 1000, "redraw": False},
                          "fromcurrent": True,
                      },
                  ],
              },
              {
                  "label": "⏸ Pausa",
                  "method": "animate",
                  "args": [
                      [None],
                      {
                          "frame": {"duration": 0, "redraw": False},
                          "mode": "immediate",
                      },
                  ],
              },
          ],
          "direction": "left",
          "pad": {"r": 10, "t": 87},
          "x": 0.1,
          "xanchor": "right",
          "y": 0,
          "yanchor": "top",
      }],
      sliders=[{
          "active": 0,
          "yanchor": "top",
          "xanchor": "left",
          "currentvalue": {
              "font": {"size": 13},
              "prefix": "Iteración: ",
              "visible": True,
              "xanchor": "right",
          },
          "pad": {"b": 10, "t": 50},
          "len": 0.9,
          "x": 0.1,
          "y": 0,
          "steps": [
              {
                  "args": [
                      [f"Paso {k}"],
                      {
                          "frame": {"duration": 300, "redraw": False},
                          "mode": "immediate",
                      },
                  ],
                  "label": f"#{k}",
                  "method": "animate",
              }
              for k in range(1, n_pasos + 1)
          ],
      }],
  )
  fig.frames = frames
  return fig


# ==============================================================================
# 7. SIMULADOR ESTOCÁSTICO CON BOOTSTRAPPING & STOP-LOSS (-10%)
# ==============================================================================
def generar_simulacion_y_estadisticos(
    max_mx,
    max_usa,
    n_dias,
    alpha,
    m_mx_acc,
    m_cetes,
    m_usa_mxn,
    m_tbills_mxn,
    num_simulaciones_futuras=10000,
    enfoque="global",
    datos_mx=None,
    datos_usa=None,
    stop_loss=-0.10,
):
  tc_inicial = 18.44
  shape_k = 28.52977
  scale_lambda = 18.32123

  monto_mx_acc = m_mx_acc if enfoque in ["global", "mexico"] else 0.0
  monto_cetes = m_cetes if enfoque in ["global", "mexico"] else 0.0
  monto_usa_acc_mxn = m_usa_mxn if enfoque in ["global", "usa"] else 0.0
  monto_tbills_mxn = m_tbills_mxn if enfoque in ["global", "usa"] else 0.0

  capital_total = (
      monto_mx_acc + monto_cetes + monto_usa_acc_mxn + monto_tbills_mxn
  )
  if capital_total == 0:
    capital_total = 1000000.0

  # CETES
  a_c, b_c, c_c = 0.07, 0.08, 0.09
  k_c = (b_c - a_c) / (c_c - a_c)
  u_c = np.random.uniform(0, 1, size=(num_simulaciones_futuras, n_dias))
  tasas_cetes = np.where(
      u_c < k_c,
      a_c + np.sqrt(u_c * (b_c - a_c) * (c_c - a_c)),
      c_c - np.sqrt((1 - u_c) * (c_c - a_c) * (c_c - b_c)),
  )
  factor_cetes_full = np.prod(1 + tasas_cetes / 360, axis=1)
  val_final_cetes_base = monto_cetes * factor_cetes_full

  # T-BILLS
  a_t, b_t, c_t = 0.0365, 0.0370, 0.0375
  k_t = (b_t - a_t) / (c_t - a_t)
  u_t = np.random.uniform(0, 1, size=(num_simulaciones_futuras, n_dias))
  tasas_tbills = np.where(
      u_t < k_t,
      a_t + np.sqrt(u_t * (b_t - a_t) * (c_t - a_t)),
      c_t - np.sqrt((1 - u_t) * (c_t - a_t) * (c_t - b_t)),
  )
  monto_tbills_usd = monto_tbills_mxn / tc_inicial
  val_final_tbills_usd = monto_tbills_usd * np.prod(
      1 + tasas_tbills / 360, axis=1
  )

  # MÉXICO
  val_final_mx_acc = np.zeros(num_simulaciones_futuras)
  val_extra_cetes_mx = np.zeros(num_simulaciones_futuras)
  conteo_salidas_mx = {}

  if monto_mx_acc > 0:
    rend_hist_mx = np.log(datos_mx / datos_mx.shift(1)).dropna()
    activos_mx = [
        e
        for e in empresas_mx
        if e in max_mx.columns and e in rend_hist_mx.columns
    ]
    pesos_mx = max_mx[activos_mx].values[0]
    conteo_salidas_mx = {e: 0 for e in activos_mx}

    mat_mx = rend_hist_mx[activos_mx].values
    idx_mx = np.random.choice(
        len(mat_mx), size=(num_simulaciones_futuras, n_dias), replace=True
    )
    sim_mx = mat_mx[idx_mx]

    tasa_prom_cetes = np.mean(tasas_cetes, axis=1)
    for j, emp in enumerate(activos_mx):
      cap_emp = monto_mx_acc * pesos_mx[j]
      if cap_emp <= 0:
        continue
      r_acum = np.cumsum(sim_mx[:, :, j], axis=1)
      toco_sl = r_acum <= stop_loss
      salio = np.any(toco_sl, axis=1)
      conteo_salidas_mx[emp] = int(np.sum(salio))

      dia_salida = np.argmax(toco_sl, axis=1)
      val_no_salio = cap_emp * np.exp(r_acum[:, -1])

      dias_restantes = np.maximum(0, n_dias - dia_salida)
      cap_refugiado = (cap_emp * (1 + stop_loss)) * (
          (1 + tasa_prom_cetes / 360) ** dias_restantes
      )

      val_final_mx_acc += np.where(salio, 0.0, val_no_salio)
      val_extra_cetes_mx += np.where(salio, cap_refugiado, 0.0)

  # EE. UU.
  monto_usa_acc_usd = monto_usa_acc_mxn / tc_inicial
  val_final_usa_acc_usd = np.zeros(num_simulaciones_futuras)
  val_extra_tbills_usa_usd = np.zeros(num_simulaciones_futuras)
  conteo_salidas_usa = {}

  if monto_usa_acc_usd > 0:
    rend_hist_usa = np.log(datos_usa / datos_usa.shift(1)).dropna()
    activos_usa = [
        e
        for e in empresas_usa
        if e in max_usa.columns and e in rend_hist_usa.columns
    ]
    pesos_usa = max_usa[activos_usa].values[0]
    conteo_salidas_usa = {e: 0 for e in activos_usa}

    mat_usa = rend_hist_usa[activos_usa].values
    idx_usa = np.random.choice(
        len(mat_usa), size=(num_simulaciones_futuras, n_dias), replace=True
    )
    sim_usa = mat_usa[idx_usa]

    tasa_prom_tbills = np.mean(tasas_tbills, axis=1)
    for j, emp in enumerate(activos_usa):
      cap_emp_usd = monto_usa_acc_usd * pesos_usa[j]
      if cap_emp_usd <= 0:
        continue
      r_acum = np.cumsum(sim_usa[:, :, j], axis=1)
      toco_sl = r_acum <= stop_loss
      salio = np.any(toco_sl, axis=1)
      conteo_salidas_usa[emp] = int(np.sum(salio))

      dia_salida = np.argmax(toco_sl, axis=1)
      val_no_salio = cap_emp_usd * np.exp(r_acum[:, -1])

      dias_restantes = np.maximum(0, n_dias - dia_salida)
      cap_refugiado_usd = (cap_emp_usd * (1 + stop_loss)) * (
          (1 + tasa_prom_tbills / 360) ** dias_restantes
      )

      val_final_usa_acc_usd += np.where(salio, 0.0, val_no_salio)
      val_extra_tbills_usa_usd += np.where(salio, cap_refugiado_usd, 0.0)

  # TIPO DE CAMBIO
  u_tc = np.random.uniform(0.001, 0.999, size=num_simulaciones_futuras)
  tc_final = scale_lambda * (u_tc / (1 - u_tc)) ** (1 / shape_k)

  # VALORES EN USD PUROS (SIN EFECTO CAMBIARIO)
  val_usa_rv_total_usd = val_final_usa_acc_usd + val_extra_tbills_usa_usd
  val_usa_total_usd = val_usa_rv_total_usd + val_final_tbills_usd

  # VALORES EN MXN
  val_final_usa_total_mxn = val_usa_total_usd * tc_final
  val_final_cetes_total = val_final_cetes_base + val_extra_cetes_mx
  val_final_global_mxn = (
      val_final_mx_acc + val_final_cetes_total + val_final_usa_total_mxn
  )

  # RENDIMIENTOS PORCENTUALES
  rend_global = (val_final_global_mxn - capital_total) / capital_total
  rend_mx_acc = (
      ((val_final_mx_acc + val_extra_cetes_mx) - monto_mx_acc) / monto_mx_acc
      if monto_mx_acc > 0
      else np.zeros_like(rend_global)
  )
  rend_usa_usd = (
      (val_usa_rv_total_usd - monto_usa_acc_usd) / monto_usa_acc_usd
      if monto_usa_acc_usd > 0
      else np.zeros_like(rend_global)
  )
  rend_cetes = (
      (val_final_cetes_base - monto_cetes) / monto_cetes
      if monto_cetes > 0
      else np.zeros_like(rend_global)
  )
  rend_tbills = (
      (val_final_tbills_usd - monto_tbills_usd) / monto_tbills_usd
      if monto_tbills_usd > 0
      else np.zeros_like(rend_global)
  )
  rend_cambiario = (tc_final - tc_inicial) / tc_inicial

  # VALORES NOMINALES FINALES (MXN y USD)
  val_mx_total_mxn = val_final_mx_acc + val_extra_cetes_mx
  val_usa_rv_mxn_val = val_usa_rv_total_usd * tc_final
  val_cetes_total_mxn = val_final_cetes_base
  val_tbills_mxn_val = val_final_tbills_usd * tc_final

  # DIFERENCIA CAMBIARIA PURA EN MXN
  val_usa_sin_cambio_mxn = val_usa_total_usd * tc_inicial
  diff_cambiaria_mxn = val_final_usa_total_mxn - val_usa_sin_cambio_mxn

  def calcular_fila_stats(arr, val_arr):
    rend = np.mean(arr)
    riesgo = np.std(arr)
    cutoff = np.percentile(val_arr, alpha * 100)
    var_alpha = np.mean(val_arr) - cutoff
    q1 = np.percentile(arr, 25)
    q2 = np.median(arr)
    q3 = np.percentile(arr, 75)
    return [
        rend,
        riesgo,
        cutoff,
        var_alpha,
        q1,
        q2,
        q3,
        np.max(arr),
        np.min(arr),
    ]

  def calcular_fila_stats_nominal(val_arr, alpha):
    rend_nom = np.mean(val_arr)
    riesgo_nom = np.std(val_arr)
    cutoff = np.percentile(val_arr, alpha * 100)
    var_alpha = rend_nom - cutoff
    q1_nom = np.percentile(val_arr, 25)
    q2_nom = np.median(val_arr)
    q3_nom = np.percentile(val_arr, 75)
    max_nom = np.max(val_arr)
    min_nom = np.min(val_arr)
    return [
        rend_nom,
        riesgo_nom,
        cutoff,
        var_alpha,
        q1_nom,
        q2_nom,
        q3_nom,
        max_nom,
        min_nom,
    ]

  filas_stats = [
      calcular_fila_stats(rend_global, val_final_global_mxn),
      calcular_fila_stats(rend_mx_acc, val_mx_total_mxn),
      calcular_fila_stats(rend_usa_usd, val_usa_rv_mxn_val),
      calcular_fila_stats(rend_cetes, val_cetes_total_mxn),
      calcular_fila_stats(rend_tbills, val_tbills_mxn_val),
      calcular_fila_stats(rend_cambiario, diff_cambiaria_mxn),
  ]

  inv_global = capital_total
  inv_mx_acc = monto_mx_acc
  inv_usa_rv_mxn = monto_usa_acc_mxn
  inv_usa_rv_usd = monto_usa_acc_usd
  inv_cetes = monto_cetes
  inv_tbills_mxn = monto_tbills_mxn
  inv_tbills_usd = monto_tbills_usd
  inv_diff_cambiaria = 0.0

  filas_stats_nominal = [
      [inv_global] + calcular_fila_stats_nominal(val_final_global_mxn, alpha),
      [inv_mx_acc] + calcular_fila_stats_nominal(val_mx_total_mxn, alpha),
      [inv_usa_rv_mxn] + calcular_fila_stats_nominal(val_usa_rv_mxn_val, alpha),
      [inv_usa_rv_usd]
      + calcular_fila_stats_nominal(val_usa_rv_total_usd, alpha),
      [inv_cetes] + calcular_fila_stats_nominal(val_cetes_total_mxn, alpha),
      [inv_tbills_mxn] + calcular_fila_stats_nominal(val_tbills_mxn_val, alpha),
      [inv_tbills_usd]
      + calcular_fila_stats_nominal(val_final_tbills_usd, alpha),
      [inv_diff_cambiaria]
      + calcular_fila_stats_nominal(diff_cambiaria_mxn, alpha),
  ]

  df_estadisticos = pd.DataFrame(
      filas_stats,
      index=[
          "Portafolio Global Final (MXN)",
          "Portafolio Renta Variable México (MXN)",
          "Portafolio Renta Variable EE. UU. (MXN)",
          "Portafolio Renta Fija CETES (MXN)",
          "Portafolio Renta Fija T-Bills (MXN)",
          "Efecto Rendimiento Cambiario",
      ],
      columns=[
          "Rendimiento",
          "Riesgo",
          f"{alpha*100:.1f}% CUTOFF",
          f"{alpha*100:.1f}% VAR",
          "Q1",
          "Q2",
          "Q3",
          "Máximo",
          "Mínimo",
      ],
  )

  df_estadisticos_nominal = pd.DataFrame(
      filas_stats_nominal,
      index=[
          "Portafolio Global Final (MXN)",
          "Portafolio Renta Variable México (MXN)",
          "Portafolio Renta Variable EE. UU. (MXN)",
          "Portafolio Renta Variable EE. UU. (USD)",
          "Portafolio Renta Fija CETES (MXN)",
          "Portafolio Renta Fija T-Bills (MXN)",
          "Portafolio Renta Fija T-Bills (USD)",
          "Diferencia Cambiaria Neta (RV USA + T-Bills en MXN)",
      ],
      columns=[
          "Inversión Inicial",
          "Valor esperado",
          "Riesgo",
          f"{alpha*100:.1f}% CUTOFF",
          f"{alpha*100:.1f}% VAR",
          "Q1",
          "Q2",
          "Q3",
          "Máximo",
          "Mínimo",
      ],
  )

  df_excel_10k = pd.DataFrame({
      "Modelado": range(1, num_simulaciones_futuras + 1),
      "Rendimiento en Pesos": rend_mx_acc,
      "Rendimiento en Dólares": rend_usa_usd,
      "Inicio del Dólar": tc_inicial,
      "Final del Dólar": tc_final,
      "Ganancia Cambiaria": rend_cambiario,
      "Diferencia Cambiaria (MXN)": diff_cambiaria_mxn,
      "Rendimiento Global": rend_global,
  })

  return (
      df_estadisticos,
      df_estadisticos_nominal,
      rend_global,
      val_final_global_mxn,
      conteo_salidas_mx,
      conteo_salidas_usa,
      df_excel_10k,
  )


# ==============================================================================
# 8. DEFINICIÓN DE LAS PESTAÑAS
# ==============================================================================
tab_mx, tab_usa, tab_port_enf, tab_excel, tab_anexo = st.tabs([
    "🇲🇽 Mercado Mexicano",
    "🇺🇸 Mercado EE. UU.",
    "🌐 Simulación por Enfoque (México, USA & Global)",
    "📑 Modelo Excel Estocástico & Títulos",
    "📐 Anexo de Fórmulas",
])

formatos_mx = {c: "{:.2%}" for c in empresas_mx}
formatos_mx.update({
    "Rp": "{:.2%}",
    "des_p": "{:.2%}",
    "Tasa_diaria": "{:.2%}",
    "Tasa_anual": "{:.2%}",
    "sharpe_ratio": "{:.2f}",
})
cols_min_mx = [c for c in empresas_mx + ["Rp", "vp", "des_p"]]
cols_sh_mx = [
    c
    for c in empresas_mx
    + ["Rp", "vp", "des_p", "Tasa_diaria", "Tasa_anual", "sharpe_ratio"]
]

formatos_usa = {c: "{:.2%}" for c in empresas_usa}
formatos_usa.update({
    "Rp": "{:.2%}",
    "des_p": "{:.2%}",
    "Tasa_diaria": "{:.2%}",
    "Tasa_anual": "{:.2%}",
    "sharpe_ratio": "{:.2f}",
})
cols_min_usa = [c for c in empresas_usa + ["Rp", "vp", "des_p"]]
cols_sh_usa = [
    c
    for c in empresas_usa
    + ["Rp", "vp", "des_p", "Tasa_diaria", "Tasa_anual", "sharpe_ratio"]
]


# ==============================================================================
# PESTAÑA 1: MERCADO MEXICANO
# ==============================================================================
with tab_mx:
  st.header("🌶️📂 Análisis de Portafolios - Activos Mexicanos y CETES")
  sub_mx_1, sub_mx_2, sub_mx_3, sub_mx_comp = st.tabs([
      "📅 1 Año",
      "📅 2 Años",
      "📅 3 Años",
      "📋 Tablas Comparativas (Mínima Varianza y Sharpe)",
  ])

  def renderizar_bloque_pais(
      datos_hist,
      cov_mat,
      min_port,
      max_port,
      anim_data,
      t1000_data,
      lista_empresas,
      cols_activas,
      titulo_h,
      lbl_tasa,
      is_mx,
  ):
    st.subheader(
        f"1. Evolución de las Acciones - Precios Normalizados ({titulo_h})"
    )
    st.plotly_chart(
        px.line(
            (datos_hist / datos_hist.iloc[0]) * 100,
            title=f"Evolución Comparativa de Precios Escalados ({titulo_h})",
        ),
        use_container_width=True,
    )

    st.subheader(f"2. Tabla de Evolución Histórica de Precios ({titulo_h})")
    st.dataframe(
        estilizar_tabla(datos_hist.tail(10).style.format("${:,.2f}")),
        use_container_width=True,
    )

    st.subheader(f"3. Estadísticos del Rendimiento Histórico ({titulo_h})")
    df_desc_rend = datos_hist.pct_change().dropna().describe()
    st.dataframe(
        estilizar_tabla(
            df_desc_rend.style.format(
                "{:.2%}", subset=pd.IndexSlice[df_desc_rend.index != "count", :]
            ).format("{:.0f}", subset=pd.IndexSlice[["count"], :])
        ),
        use_container_width=True,
    )

    st.subheader(f"4. Matriz de Covarianza Anualizada ({titulo_h})")
    matriz_limpia = np.array(cov_mat, dtype=float)
    if (
        matriz_limpia.ndim == 2
        and matriz_limpia.shape[0] > 1
        and matriz_limpia.shape[1] > 1
    ):
      # Escala: si no hay covarianzas negativas se usa azul de 0 a máx. (más
      # contraste); si las hay, escala divergente centrada en 0.
      _kw_color = (
          dict(color_continuous_scale=COLORSCALE_AZUL, zmin=0)
          if np.nanmin(matriz_limpia) >= 0
          else dict(color_continuous_scale="RdBu", color_continuous_midpoint=0)
      )
      fig_cov = px.imshow(
          matriz_limpia,
          x=cols_activas,
          y=cols_activas,
          aspect="auto",
          title="Matriz de Covarianza (Sigma) en %",
          **_kw_color,
      )
      fig_cov.update_traces(
          texttemplate="%{z:.2%}",
          hovertemplate="%{y} × %{x}: %{z:.3%}<extra></extra>",
      )
      fig_cov.update_layout(
          height=420,
          coloraxis_colorbar=dict(title="%", tickformat=".0%", thickness=14),
      )
      st.plotly_chart(fig_cov, use_container_width=True)
    else:
      st.warning("⚠️ No hay suficientes activos válidos.")

    st.subheader(f"5. Portafolio de Mínima Varianza ({titulo_h})")
    cols_min = [c for c in cols_activas + ["Rp", "vp", "des_p"]]
    st.dataframe(
        estilizar_tabla(
            min_port[[c for c in cols_min if c in min_port.columns]].style.format(
                formatos_mx if is_mx else formatos_usa
            )
        ),
        use_container_width=True,
    )

    st.subheader(f"6. Optimización por Ratio de Sharpe ({titulo_h})")
    st.plotly_chart(
        construir_figura_animada_con_minima(
            anim_data,
            min_port,
            cols_activas,
            titulo_h,
            lbl_tasa,
            is_mx=is_mx,
            pct_rf=pct_renta_fija,
        ),
        use_container_width=True,
    )

    st.markdown("#### Portafolio Óptimo Global (Máximo Sharpe Absoluto)")
    cols_sh = [
        c
        for c in cols_activas
        + ["Rp", "vp", "des_p", "Tasa_diaria", "Tasa_anual", "sharpe_ratio"]
    ]
    st.dataframe(
        estilizar_tabla(
            max_port[[c for c in cols_sh if c in max_port.columns]].style.format(
                formatos_mx if is_mx else formatos_usa
            )
        ),
        use_container_width=True,
    )

    st.subheader(
        f"7. Estadísticos Importantes & Intervalo de Confianza ({titulo_h})"
    )
    vals_sharpe = t1000_data["sharpe_ratio"].values
    df_stats_sharpe = pd.DataFrame({
        "Métrica": [
            "Media Sharpe",
            "Desviación Estándar Sharpe",
            "Mediana (Q2)",
            f"IC Inf ({ (1-alpha_nivel)*100:.1f}%)",
            f"IC Sup ({alpha_nivel*100:.1f}%)",
            "Máximo Sharpe",
            "Mínimo Sharpe",
        ],
        "Valor": [
            np.mean(vals_sharpe),
            np.std(vals_sharpe),
            np.median(vals_sharpe),
            np.percentile(vals_sharpe, alpha_nivel * 50),
            np.percentile(vals_sharpe, (1 - alpha_nivel / 2) * 100),
            np.max(vals_sharpe),
            np.min(vals_sharpe),
        ],
    })
    st.dataframe(
        estilizar_tabla(df_stats_sharpe.style.format({"Valor": "{:.2f}"})),
        use_container_width=True,
        hide_index=True,
    )

  with sub_mx_1:
    renderizar_bloque_pais(
        datos_mx_1,
        cov_mx_1,
        min_mx_1,
        max_mx_1,
        anim_mx_1,
        t1000_mx_1,
        empresas_mx,
        cols_activas_mx_1,
        "1 Año (México)",
        "CETES",
        True,
    )
  with sub_mx_2:
    renderizar_bloque_pais(
        datos_mx_2,
        cov_mx_2,
        min_mx_2,
        max_mx_2,
        anim_mx_2,
        t1000_mx_2,
        empresas_mx,
        cols_activas_mx_2,
        "2 Años (México)",
        "CETES",
        True,
    )
  with sub_mx_3:
    renderizar_bloque_pais(
        datos_mx_3,
        cov_mx_3,
        min_mx_3,
        max_mx_3,
        anim_mx_3,
        t1000_mx_3,
        empresas_mx,
        cols_activas_mx_3,
        "3 Años (México)",
        "CETES",
        True,
    )
  with sub_mx_comp:
    st.subheader("📋 Tablas Comparativas de los 3 Años (México)")
    st.markdown(
        "### A. Tabla Comparativa - Criterio de Mínima Varianza (1, 2 y 3 Años)"
    )
    st.dataframe(
        estilizar_tabla(
            pd.concat([min_mx_1, min_mx_2, min_mx_3])[
                [c for c in cols_min_mx if c in min_mx_1.columns]
            ].style.format(formatos_mx)
        ),
        use_container_width=True,
    )
    st.markdown(
        "### B. Tabla Comparativa - Criterio de Máximo Sharpe Ratio (1, 2 y 3"
        " Años)"
    )
    st.dataframe(
        estilizar_tabla(
            pd.concat([max_mx_1, max_mx_2, max_mx_3])[
                [c for c in cols_sh_mx if c in max_mx_1.columns]
            ].style.format(formatos_mx)
        ),
        use_container_width=True,
    )


# ==============================================================================
# PESTAÑA 2: ESTADOS UNIDOS
# ==============================================================================
with tab_usa:
  st.header("📂🗽 Análisis de Portafolios - Activos de EE. UU. y T-Bills")
  sub_usa_1, sub_usa_2, sub_usa_3, sub_usa_comp = st.tabs([
      "📅 1 Año",
      "📅 2 Años",
      "📅 3 Años",
      "📋 Tablas Comparativas (Mínima Varianza y Sharpe)",
  ])

  with sub_usa_1:
    renderizar_bloque_pais(
        datos_usa_1,
        cov_usa_1,
        min_usa_1,
        max_usa_1,
        anim_usa_1,
        t1000_usa_1,
        empresas_usa,
        cols_activas_usa_1,
        "1 Año (EE. UU.)",
        "T-Bills",
        False,
    )
  with sub_usa_2:
    renderizar_bloque_pais(
        datos_usa_2,
        cov_usa_2,
        min_usa_2,
        max_usa_2,
        anim_usa_2,
        t1000_usa_2,
        empresas_usa,
        cols_activas_usa_2,
        "2 Años (EE. UU.)",
        "T-Bills",
        False,
    )
  with sub_usa_3:
    renderizar_bloque_pais(
        datos_usa_3,
        cov_usa_3,
        min_usa_3,
        max_usa_3,
        anim_usa_3,
        t1000_usa_3,
        empresas_usa,
        cols_activas_usa_3,
        "3 Años (EE. UU.)",
        "T-Bills",
        False,
    )

  with sub_usa_comp:
    st.subheader("📋 Tablas Comparativas de los 3 Años (EE. UU.)")
    st.markdown(
        "### A. Tabla Comparativa - Criterio de Mínima Varianza (1, 2 y 3 Años)"
    )
    st.dataframe(
        estilizar_tabla(
            pd.concat([min_usa_1, min_usa_2, min_usa_3])[
                [c for c in cols_min_usa if c in min_usa_1.columns]
            ].style.format(formatos_usa)
        ),
        use_container_width=True,
    )
    st.markdown(
        "### B. Tabla Comparativa - Criterio de Máximo Sharpe Ratio (1, 2 y 3"
        " Años)"
    )
    st.dataframe(
        estilizar_tabla(
            pd.concat([max_usa_1, max_usa_2, max_usa_3])[
                [c for c in cols_sh_usa if c in max_usa_1.columns]
            ].style.format(formatos_usa)
        ),
        use_container_width=True,
    )

# ==============================================================================
# PESTAÑA 3: SIMULACIÓN POR ENFOQUE
# ==============================================================================
with tab_port_enf:
  st.header(
      "🌐 Simulación Estocástica por Enfoque: Full México, Full EE. UU. y Unión"
      " Global"
  )

  col_sel1, col_sel2 = st.columns(2)
  with col_sel1:
    enfoque_sel = st.selectbox(
        "Selecciona el enfoque de simulación:",
        [
            "Full México 🏛️ (Solo Acciones MX + CETES)",
            "Full EE. UU. 🗽 (Solo Acciones USA + T-Bills + TC)",
            "Global Unificado 🗺️⚖️(México + EE. UU. Combinados)",
        ],
        key="sel_enf_dinamico",
    )
  with col_sel2:
    horiz_sel = st.selectbox(
        "Selecciona el intervalo de tiempo para el que deseas evaluar tus Wi:",
        ["1 año", "2 años", "3 años"],
        key="sel_h_enfoque",
    )

  if horiz_sel == "1 año":
    s_mx, s_usa, d_mx, d_usa = max_mx_1, max_usa_1, datos_mx_1, datos_usa_1
  elif horiz_sel == "2 años":
    s_mx, s_usa, d_mx, d_usa = max_mx_2, max_usa_2, datos_mx_2, datos_usa_2
  else:
    s_mx, s_usa, d_mx, d_usa = max_mx_3, max_usa_3, datos_mx_3, datos_usa_3

  m_mx_acc, m_cetes, m_usa_mxn, m_tbills_mxn, w_mx_din, w_usa_din = (
      calcular_capitales_dinamicos(
          s_mx, s_usa, d_mx, d_usa, capital_total_input, pct_renta_fija
      )
  )

  if "Full México" in enfoque_sel:
    enf_key, mostrar_mx, mostrar_usa = "mexico", True, False
    titulo_enf = f"Full México (${m_mx_acc + m_cetes:,.2f} MXN Asignados)"
  elif "Full EE. UU." in enfoque_sel:
    enf_key, mostrar_mx, mostrar_usa = "usa", False, True
    titulo_enf = (
        f"Full EE. UU. (${m_usa_mxn + m_tbills_mxn:,.2f} MXN Equiv. en USD)"
    )
  else:
    enf_key, mostrar_mx, mostrar_usa = "global", True, True
    titulo_enf = (
        f"Portafolio Global Unificado (${capital_total_input:,.2f} MXN"
        " Dinámicos)"
    )

  st.markdown(
      "### 💰 Distribución Dinámica del Capital Total"
      f" (${capital_total_input:,.2f} MXN)"
  )
  df_dist_dinamica = pd.DataFrame({
      "Componente / Mercado": [
          "Acciones México (Optimizado W)",
          "Deuda México (CETES)",
          "Acciones EE. UU. (Optimizado W en USD)",
          "Deuda EE. UU. (T-Bills en USD)",
      ],
      "Peso Dinámico (Mín. Varianza)": [
          w_mx_din * (1 - pct_renta_fija),
          w_mx_din * pct_renta_fija,
          w_usa_din * (1 - pct_renta_fija),
          w_usa_din * pct_renta_fija,
      ],
      "Inversión Asignada (MXN)": [m_mx_acc, m_cetes, m_usa_mxn, m_tbills_mxn],
      "Tipo de Instrumento": [
          "Renta Variable MX",
          "Renta Fija MX",
          "Renta Variable USA",
          "Renta Fija USA",
      ],
  })
  st.dataframe(
      estilizar_tabla(
          df_dist_dinamica.style.format({
              "Peso Dinámico (Mín. Varianza)": "{:.2%}",
              "Inversión Asignada (MXN)": "${:,.2f}",
          })
      ),
      use_container_width=True,
  )

  precios_mx_actuales = {
      k: obtener_precio_actual(v, 100.0) for k, v in tickers_mx.items()
  }
  precios_usa_actuales = {
      k: obtener_precio_actual(v, 150.0) for k, v in tickers_usa.items()
  }

  col_precio_mx = f"Precio Actual (MXN - {fecha_hoy_str})"
  col_precio_usa = f"Precio Actual (USD - {fecha_hoy_str})"

  if mostrar_mx:
    st.markdown(f"#### 🇲🇽 Portafolio Accionario México (${m_mx_acc:,.2f} MXN)")
    activos_validos_mx = [e for e in empresas_mx if e in s_mx.columns]
    pesos_mx_opt = s_mx[activos_validos_mx].values[0]
    df_t_mx = pd.DataFrame({
        "Empresa": activos_validos_mx,
        "Peso Óptimo": pesos_mx_opt,
        "Capital Asignado (MXN)": pesos_mx_opt * m_mx_acc,
        col_precio_mx: [precios_mx_actuales[e] for e in activos_validos_mx],
    })
    df_t_mx["Títulos a Comprar"] = np.floor(
        df_t_mx["Capital Asignado (MXN)"] / df_t_mx[col_precio_mx]
    ).astype(int)
    st.dataframe(
        estilizar_tabla(
            df_t_mx.style.format({
                "Peso Óptimo": "{:.2%}",
                "Capital Asignado (MXN)": "${:,.2f}",
                col_precio_mx: "${:,.2f}",
                "Títulos a Comprar": "{:d}",
            })
        ),
        use_container_width=True,
    )

  if mostrar_usa:
    st.markdown(
        f"#### 🇺🇸 Portafolio Accionario EE. UU. (${m_usa_mxn:,.2f} MXN en USD)"
    )
    activos_validos_usa = [e for e in empresas_usa if e in s_usa.columns]
    pesos_usa_opt = s_usa[activos_validos_usa].values[0]
    tc_inicial = 18.44
    df_t_usa = pd.DataFrame({
        "Empresa": activos_validos_usa,
        "Peso Óptimo": pesos_usa_opt,
        "Capital Asignado (USD)": pesos_usa_opt * (m_usa_mxn / tc_inicial),
        col_precio_usa: [precios_usa_actuales[e] for e in activos_validos_usa],
    })
    df_t_usa["Títulos a Comprar"] = np.floor(
        df_t_usa["Capital Asignado (USD)"] / df_t_usa[col_precio_usa]
    ).astype(int)
    st.dataframe(
        estilizar_tabla(
            df_t_usa.style.format({
                "Peso Óptimo": "{:.2%}",
                "Capital Asignado (USD)": "${:,.2f}",
                col_precio_usa: "${:,.2f}",
                "Títulos a Comprar": "{:d}",
            })
        ),
        use_container_width=True,
    )

  st.markdown("---")

  # 1. EJECUCIÓN DE LAS 10,000 SIMULACIONES
  (
      df_est_enf,
      df_est_enf_nominal,
      rend_enf_sim,
      val_enf_sim,
      sl_mx_c,
      sl_usa_c,
      df_excel_10k,
  ) = generar_simulacion_y_estadisticos(
      s_mx,
      s_usa,
      dias_habiles,
      alpha_nivel,
      m_mx_acc,
      m_cetes,
      m_usa_mxn,
      m_tbills_mxn,
      num_simulaciones_futuras=10000,
      enfoque=enf_key,
      datos_mx=d_mx,
      datos_usa=d_usa,
      stop_loss=-0.10,
  )

  # TABLA 1: EN PORCENTAJES
  st.subheader(
      "📊 Resultados Finales en Porcentajes (10,000 Simulaciones"
      " Bootstrap + Stop-Loss Individual 10%)"
  )

  def formateador_tabla_riesgo(val, col_name):
    if isinstance(val, (int, float)):
      if "CUTOFF" in col_name or "VAR" in col_name:
        return f"${val:,.2f}"
      else:
        return f"{val:.2%}"
    return val

  st.dataframe(
      estilizar_tabla(
          df_est_enf.style.format({
              col: lambda v, c=col: formateador_tabla_riesgo(v, c)
              for col in df_est_enf.columns
          })
      ),
      use_container_width=True,
  )

  # TABLA 2: EN CANTIDADES NOMINALES (MXN y USD) CON INVERSIÓN INICIAL
  st.subheader(
      "💵 Resultados Finales en Cantidades Nominales (MXN, USD y"
      " Diferencia Cambiaria Neta)"
  )

  def formateador_tabla_nominal(val):
    if isinstance(val, (int, float)):
      return f"${val:,.2f}"
    return val

  st.dataframe(
      estilizar_tabla(
          df_est_enf_nominal.style.format({
              col: formateador_tabla_nominal
              for col in df_est_enf_nominal.columns
          })
      ),
      use_container_width=True,
  )

  col_k1, col_k2, col_k3, col_k4 = st.columns(4)
  col_k1.metric("Rendimiento Esperado", f"{np.mean(rend_enf_sim)*100:.2f}%")
  col_k2.metric("Volatilidad (Riesgo)", f"{np.std(rend_enf_sim)*100:.2f}%")
  col_k3.metric(
      f"VaR ({alpha_nivel*100:.1f}%)",
      f"${capital_total_input - np.percentile(val_enf_sim, alpha_nivel*100):,.2f} MXN",
  )
  col_k4.metric("Mediana Final (Q2)", f"${np.median(val_enf_sim):,.2f} MXN")

  st.markdown("---")

  st.subheader(
      "🛡️ Ranking de Salidas por Stop-Loss (-10% Individual → Refugio en CETES"
      " / T-Bills)"
  )
  filas_sl = []
  for emp, cnt in sl_mx_c.items():
    filas_sl.append({
        "Mercado": "🇲🇽 México",
        "Acción": emp,
        "Veces que Salió (-10%)": cnt,
        "% de Simulaciones": cnt / 10000.0,
    })
  for emp, cnt in sl_usa_c.items():
    filas_sl.append({
        "Mercado": "🇺🇸 EE. UU.",
        "Acción": emp,
        "Veces que Salió (-10%)": cnt,
        "% de Simulaciones": cnt / 10000.0,
    })

  if filas_sl:
    df_sl_ranking = (
        pd.DataFrame(filas_sl)
        .sort_values(by="Veces que Salió (-10%)", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        estilizar_tabla(
            df_sl_ranking.style.format({
                "Veces que Salió (-10%)": "{:,d}",
                "% de Simulaciones": "{:.2%}",
            })
        ),
        use_container_width=True,
    )

  st.markdown("---")
  st.subheader(
      "📋 Tabla Resumen de 10,000 Simulaciones Bootstrap -"
      " Primeras 15 Filas"
  )
  st.dataframe(
      estilizar_tabla(
          df_excel_10k.head(15).style.format({
              "Rendimiento en Pesos": "{:.2%}",
              "Rendimiento en Dólares": "{:.2%}",
              "Inicio del Dólar": "${:,.2f}",
              "Final del Dólar": "${:,.2f}",
              "Ganancia Cambiaria": "{:.2%}",
              "Diferencia Cambiaria (MXN)": "${:,.2f}",
              "Rendimiento Global": "{:.2%}",
          })
      ),
      use_container_width=True,
  )

  st.markdown("---")
  st.subheader(f"Gráficas de Distribución y Comportamiento Estocástico ({titulo_enf})")
  st.caption(
      "La zona roja marca el peor escenario según el nivel de significancia"
      " elegido (Cutoff)."
  )

  media_val = np.mean(val_enf_sim)
  mediana_val = np.median(val_enf_sim)
  q1_val = np.percentile(val_enf_sim, 25)
  q3_val = np.percentile(val_enf_sim, 75)
  cutoff_val = np.percentile(val_enf_sim, alpha_nivel * 100)

  conteos, bordes = np.histogram(val_enf_sim, bins=50)
  idx_moda = np.argmax(conteos)
  moda_val = (bordes[idx_moda] + bordes[idx_moda + 1]) / 2

  fig_hist = px.histogram(
      x=val_enf_sim,
      nbins=50,
      labels={"x": "Valor Final en MXN"},
      title=(
          "Distribución de Probabilidad del Valor Final (10,000 Corridas"
          " Bootstrap)"
      ),
      color_discrete_sequence=[AZUL],
      opacity=0.9,
  )
  fig_hist.update_traces(marker_line_width=0)
  fig_hist.data[0].showlegend = False

  # Zona de peor escenario (a la izquierda del Cutoff)
  fig_hist.add_vrect(
      x0=float(np.min(val_enf_sim)),
      x1=float(cutoff_val),
      fillcolor=ROJO,
      opacity=0.08,
      line_width=0,
      layer="below",
  )

  # Líneas de referencia: se identifican por color/trazo en la leyenda
  # (sin etiquetas de texto encimadas).
  _lineas_hist = [
      ("Media", media_val, VERDE, "solid", 2.5, True),
      ("Mediana (Q2)", mediana_val, GRIS_OSC, "dash", 2, True),
      ("Moda", moda_val, AMBAR, "dot", 2.5, True),
      ("Q1 / Q3", q1_val, GRIS, "dashdot", 1.5, True),
      ("Q1 / Q3", q3_val, GRIS, "dashdot", 1.5, False),
      (f"Cutoff ({alpha_nivel*100:.1f}%)", cutoff_val, ROJO, "dash", 2.5, True),
  ]
  for _n, _x, _c, _d, _w, _leg in _lineas_hist:
    fig_hist.add_vline(x=_x, line_dash=_d, line_color=_c, line_width=_w)
    if _leg:
      fig_hist.add_trace(
          go.Scatter(
              x=[None],
              y=[None],
              mode="lines",
              line=dict(color=_c, dash=_d, width=_w),
              name=_n,
          )
      )

  fig_hist.add_annotation(
      xref="paper",
      yref="paper",
      x=0.99,
      y=0.97,
      xanchor="right",
      yanchor="top",
      text=(
          "<b>Estadísticos (MXN)</b><br>"
          f"Media: ${media_val:,.2f}<br>"
          f"Mediana (Q2): ${mediana_val:,.2f}<br>"
          f"Moda: ${moda_val:,.2f}<br>"
          f"Q1 (25%): ${q1_val:,.2f}<br>"
          f"Q3 (75%): ${q3_val:,.2f}<br>"
          f"Cutoff ({alpha_nivel*100:.1f}%): ${cutoff_val:,.2f}"
      ),
      showarrow=False,
      align="left",
      bgcolor="rgba(255, 255, 255, 0.95)",
      bordercolor="#D5DBE5",
      borderwidth=1,
      borderpad=8,
      font=dict(size=12, color="#1F2937"),
  )

  fig_hist.update_layout(
      xaxis_title="Valor Final (MXN)",
      yaxis_title="Frecuencia (Corridas)",
      template="fin_clean",
      bargap=0.04,
      showlegend=True,
      legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
      margin=dict(t=70, b=90),
  )
  st.plotly_chart(fig_hist, use_container_width=True)

  col_g1, col_g2 = st.columns(2)
  with col_g1:
    fig_box = px.box(
        y=val_enf_sim / 1000,
        points="outliers",
        title="Boxplot - Dispersión del Valor Final (Miles MXN)",
        labels={"y": "Valor Final (Miles MXN)"},
        color_discrete_sequence=[AZUL],
    )
    fig_box.update_traces(
        boxmean="sd",
        fillcolor="rgba(31, 78, 121, 0.25)",
        line=dict(color=AZUL, width=1.8),
        marker=dict(size=4, opacity=0.4, color=AZUL),
    )
    fig_box.update_layout(template="fin_clean")
    st.plotly_chart(fig_box, use_container_width=True)

  with col_g2:
    n_sims = len(rend_enf_sim)
    pasos = np.arange(1, n_sims + 1)
    rend_pct = rend_enf_sim * 100

    media_acum = np.cumsum(rend_pct) / pasos
    media_cuad_acum = np.cumsum(rend_pct**2) / pasos
    std_acum = np.sqrt(np.maximum(0, media_cuad_acum - media_acum**2))

    media_r = np.mean(rend_pct)
    cutoff_r = np.percentile(rend_pct, alpha_nivel * 100)

    step_plot = max(1, n_sims // 2000)

    fig_evol_r = go.Figure()
    fig_evol_r.add_trace(
        go.Scatter(
            x=pasos[::step_plot],
            y=rend_pct[::step_plot],
            mode="markers",
            marker=dict(size=3, color=GRIS, opacity=0.25),
            name="Rendimiento Individual (%)",
        )
    )
    fig_evol_r.add_trace(
        go.Scatter(
            x=pasos,
            y=media_acum + std_acum,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_evol_r.add_trace(
        go.Scatter(
            x=pasos,
            y=media_acum - std_acum,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(46, 158, 107, 0.15)",
            name="Banda ±1 Desv. Estándar",
        )
    )
    fig_evol_r.add_trace(
        go.Scatter(
            x=pasos,
            y=media_acum,
            mode="lines",
            line=dict(color=VERDE, width=2.5),
            name=f"Media Acumulada ({media_r:.2f}%)",
        )
    )
    fig_evol_r.add_hline(y=cutoff_r, line_dash="dash", line_color=ROJO, line_width=1.8)
    # Etiqueta del Cutoff fuera del área de puntos (a la derecha)
    fig_evol_r.add_annotation(
        xref="paper",
        yref="y",
        x=1.01,
        y=cutoff_r,
        xanchor="left",
        showarrow=False,
        text=f"Cutoff ({alpha_nivel*100:.1f}%)<br>{cutoff_r:.2f}%",
        font=dict(size=11, color=ROJO),
    )
    fig_evol_r.update_layout(
        title="Evolución y Convergencia del Rendimiento en las Corridas",
        xaxis_title="Número de Corrida (#1 a #10,000)",
        yaxis_title="Rendimiento (%)",
        template="fin_clean",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=60, b=90, r=90),
    )
    st.plotly_chart(fig_evol_r, use_container_width=True)

  st.markdown("---")
  st.subheader("Gráfica de Optimización Ratio de Sharpe (Portafolio Combinado MX + USA)")
  st.caption(
      "Cada punto es una mezcla MX / EE. UU.; el color indica su Ratio de"
      " Sharpe. La línea gris es la Línea de Asignación de Capital (CAL)."
  )
  rp_mx_anual = s_mx["Rp"].values[0]
  desv_mx_anual = s_mx["des_p"].values[0]
  rp_usa_anual = s_usa["Rp"].values[0]
  desv_usa_anual = s_usa["des_p"].values[0]

  w_grid = np.linspace(0, 1, 60)
  ret_comb_grid = w_grid * rp_mx_anual + (1 - w_grid) * rp_usa_anual
  riesgo_comb_grid = np.sqrt(
      (w_grid * desv_mx_anual) ** 2 + ((1 - w_grid) * desv_usa_anual) ** 2
  )

  media_cetes = (0.04 + 0.08 + 0.10) / 3
  media_tbills = (0.0360 + 0.0370 + 0.0400) / 3
  tasa_rf_global = media_cetes * w_mx_din + media_tbills * w_usa_din

  sharpe_comb_grid = (ret_comb_grid - tasa_rf_global) / riesgo_comb_grid

  idx_best_sh = np.argmax(sharpe_comb_grid)
  idx_min_var_g = np.argmin(riesgo_comb_grid)

  # Rango de ejes ajustado a la frontera real (la curva se ve más abierta)
  _x_pad = max((riesgo_comb_grid.max() - riesgo_comb_grid.min()) * 0.15, 1e-4)
  _y_pad = max((ret_comb_grid.max() - ret_comb_grid.min()) * 0.15, 1e-4)
  x_lo = float(riesgo_comb_grid.min() - _x_pad)
  x_hi = float(riesgo_comb_grid.max() + _x_pad)
  y_lo = float(ret_comb_grid.min() - _y_pad)
  y_hi = float(ret_comb_grid.max() + _y_pad)

  fig_sh_global = go.Figure()
  fig_sh_global.add_trace(
      go.Scatter(
          x=riesgo_comb_grid,
          y=ret_comb_grid,
          mode="markers",
          marker=dict(
              size=8,
              color=sharpe_comb_grid,
              colorscale=COLORSCALE_AZUL,
              showscale=True,
              colorbar=dict(title="Sharpe", thickness=12, x=1.02),
              line=dict(width=0.5, color="white"),
          ),
          name="Portafolios Combinados",
          showlegend=False,
      )
  )
  fig_sh_global.add_trace(
      go.Scatter(
          x=[riesgo_comb_grid[idx_best_sh]],
          y=[ret_comb_grid[idx_best_sh]],
          mode="markers+text",
          text=[f"Sharpe {sharpe_comb_grid[idx_best_sh]:.2f}"],
          textposition="top left",
          textfont=dict(color=VERDE, size=12),
          marker=dict(
              size=16,
              color=VERDE,
              symbol="star",
              line=dict(width=1.5, color="white"),
          ),
          name="Máximo Sharpe Global",
      )
  )
  fig_sh_global.add_trace(
      go.Scatter(
          x=[riesgo_comb_grid[idx_min_var_g]],
          y=[ret_comb_grid[idx_min_var_g]],
          mode="markers+text",
          text=["Mín. varianza"],
          textposition="middle left",
          textfont=dict(color="#B07F00", size=12),
          marker=dict(
              size=12,
              color=AMBAR,
              symbol="circle",
              line=dict(width=1.5, color="white"),
          ),
          name="Mínima Varianza Global",
      )
  )

  x_cal_g = np.linspace(x_lo, x_hi, 30)
  y_cal_g = tasa_rf_global + sharpe_comb_grid[idx_best_sh] * x_cal_g
  fig_sh_global.add_trace(
      go.Scatter(
          x=x_cal_g,
          y=y_cal_g,
          mode="lines",
          line=dict(color=GRIS, width=1.5, dash="dash"),
          name="Línea de Asignación de Capital (CAL)",
      )
  )
  fig_sh_global.update_layout(
      title="Frontera Eficiente & Ratio de Sharpe (Combinación MX + EE. UU.)",
      xaxis_title="Volatilidad (Riesgo Anual)",
      yaxis_title="Rendimiento Anual Esperado",
      xaxis=dict(range=[x_lo, x_hi]),
      yaxis=dict(range=[y_lo, y_hi]),
      template="fin_clean",
      legend=dict(
          orientation="h",
          yanchor="top",
          y=-0.2,
          xanchor="center",
          x=0.5,
      ),
      margin=dict(t=70, b=90, r=80),
  )
  st.plotly_chart(fig_sh_global, use_container_width=True)

  st.markdown("---")
  st.subheader(f"Tabla de Simulación Diaria Estocástica ({titulo_enf})")
  st.caption("Muestra de los primeros 15 días de la proyección, a partir de la fecha de inicio elegida.")
  dias_muestra = 15
  fechas_m = [
      (fecha_inicio_sim + datetime.timedelta(days=i)).strftime("%d/%m/%Y")
      for i in range(dias_muestra)
  ]

  tc_base = 18.44
  val_mx = m_mx_acc
  val_cetes = m_cetes
  val_usa_usd = m_usa_mxn / tc_base
  val_tb_usd = m_tbills_mxn / tc_base

  rend_h_mx = np.log(d_mx / d_mx.shift(1)).dropna()
  act_mx_d = [
      e for e in empresas_mx if e in s_mx.columns and e in rend_h_mx.columns
  ]
  port_hist_mx_demo = (
      rend_h_mx[act_mx_d].dot(s_mx[act_mx_d].values[0]).values
      if len(act_mx_d) > 0
      else np.zeros(10)
  )

  rend_h_usa = np.log(d_usa / d_usa.shift(1)).dropna()
  act_usa_d = [
      e for e in empresas_usa if e in s_usa.columns and e in rend_h_usa.columns
  ]
  port_hist_usa_demo = (
      rend_h_usa[act_usa_d].dot(s_usa[act_usa_d].values[0]).values
      if len(act_usa_d) > 0
      else np.zeros(10)
  )

  filas_excel = []
  for idx, f in enumerate(fechas_m):
    p_cetes = np.random.uniform(0, 1)
    tasa_c = (
        0.078 + np.sqrt(p_cetes * 0.002 * 0.004)
        if p_cetes < 0.5
        else 0.082 - np.sqrt((1 - p_cetes) * 0.004 * 0.002)
    )
    p_tb = np.random.uniform(0, 1)
    tasa_t = (
        0.0368 + np.sqrt(p_tb * 0.0002 * 0.0004)
        if p_tb < 0.5
        else 0.0372 - np.sqrt((1 - p_tb) * 0.0004 * 0.0002)
    )

    if idx == 0:
      tc, rend_mx_n, rend_usa_n = tc_base, 0.0, 0.0
    else:
      rend_mx_n = float(np.random.choice(port_hist_mx_demo))
      rend_usa_n = float(np.random.choice(port_hist_usa_demo))
      val_mx *= 1 + rend_mx_n
      val_cetes *= 1 + tasa_c / 360
      val_usa_usd *= 1 + rend_usa_n
      val_tb_usd *= 1 + tasa_t / 360
      u_w = np.random.uniform(0.3, 0.7)
      tc = 18.32123 * (u_w / (1 - u_w)) ** (1 / 28.52977)

    total_usa_mxn = (val_usa_usd + val_tb_usd) * tc
    total_global = val_mx + val_cetes + total_usa_mxn

    fila_dict = {"Fecha": f}
    if mostrar_mx:
      fila_dict.update({
          "Rend. MX (Boot)": rend_mx_n,
          "Portafolio Acc. MX ($)": val_mx,
          "Prob CETES": p_cetes,
          "Tasa CETES (Triang)": tasa_c,
          "Valor CETES ($)": val_cetes,
      })
    if mostrar_usa:
      fila_dict.update({
          "Rend. USA (Boot)": rend_usa_n,
          "Portafolio Acc. USA (USD)": val_usa_usd,
          "Prob T-Bills": p_tb,
          "Tasa T-Bills (Triang)": tasa_t,
          "Valor T-Bills (USD)": val_tb_usd,
          "Tipo Cambio USD/MXN": tc,
          "Valor Total USA (MXN)": total_usa_mxn,
      })
    if enf_key == "global":
      fila_dict["VALOR TOTAL PORTAFOLIO GLOBAL (MXN)"] = total_global
    elif enf_key == "mexico":
      fila_dict["VALOR TOTAL MÉXICO (MXN)"] = val_mx + val_cetes
    else:
      fila_dict["VALOR TOTAL USA (MXN)"] = total_usa_mxn
    filas_excel.append(fila_dict)

  df_excel_demo = pd.DataFrame(filas_excel)
  formatos_tabla = {
      "Rend. MX (Boot)": "{:.2%}",
      "Portafolio Acc. MX ($)": "${:,.2f}",
      "Prob CETES": "{:.2f}",
      "Tasa CETES (Triang)": "{:.2%}",
      "Valor CETES ($)": "${:,.2f}",
      "Rend. USA (Boot)": "{:.2%}",
      "Portafolio Acc. USA (USD)": "${:,.2f}",
      "Prob T-Bills": "{:.2f}",
      "Tasa T-Bills (Triang)": "{:.2%}",
      "Valor T-Bills (USD)": "${:,.2f}",
      "Tipo Cambio USD/MXN": "${:,.4f}",
      "Valor Total USA (MXN)": "${:,.2f}",
      "VALOR TOTAL PORTAFOLIO GLOBAL (MXN)": "${:,.2f}",
      "VALOR TOTAL MÉXICO (MXN)": "${:,.2f}",
      "VALOR TOTAL USA (MXN)": "${:,.2f}",
  }
  st.dataframe(
      estilizar_tabla(
          df_excel_demo.style.format({
              c: v for c, v in formatos_tabla.items() if c in df_excel_demo.columns
          })
      ),
      use_container_width=True,
  )


# ==============================================================================
# PESTAÑA 4: MODELO EXCEL ESTOCÁSTICO & TÍTULOS
# ==============================================================================
with tab_excel:
  st.header(
      "📑 Simulación Diaria Estocástica & Distribución de Acciones (Títulos"
      " Enteros)"
  )
  horizonte_excel = st.selectbox(
      "Selecciona el horizonte para extraer los pesos óptimos:",
      ["1 año", "2 años", "3 años"],
      key="sel_excel",
  )
  if horizonte_excel == "1 año":
    mx_p, usa_p, d_mx_ex, d_usa_ex = (
        max_mx_1,
        max_usa_1,
        datos_mx_1,
        datos_usa_1,
    )
  elif horizonte_excel == "2 años":
    mx_p, usa_p, d_mx_ex, d_usa_ex = (
        max_mx_2,
        max_usa_2,
        datos_mx_2,
        datos_usa_2,
    )
  else:
    mx_p, usa_p, d_mx_ex, d_usa_ex = (
        max_mx_3,
        max_usa_3,
        datos_mx_3,
        datos_usa_3,
    )

  m_mx_ex, _, m_usa_ex, _, _, _ = calcular_capitales_dinamicos(
      mx_p, usa_p, d_mx_ex, d_usa_ex, capital_total_input, pct_renta_fija
  )
  precios_mx_actuales = {
      k: obtener_precio_actual(v, 100.0) for k, v in tickers_mx.items()
  }
  precios_usa_actuales = {
      k: obtener_precio_actual(v, 150.0) for k, v in tickers_usa.items()
  }

  col_precio_mx_ex = f"Precio Actual (MXN - {fecha_hoy_str})"
  col_precio_usa_ex = f"Precio Actual (USD - {fecha_hoy_str})"

  st.subheader(
      f"🇲🇽 Asignación de Títulos - Portafolio México (${m_mx_ex:,.2f} MXN)"
  )
  activos_validos_mx = [e for e in empresas_mx if e in mx_p.columns]
  pesos_mx_opt = mx_p[activos_validos_mx].values[0]
  df_titulos_mx = pd.DataFrame({
      "Empresa": activos_validos_mx,
      "Peso Óptimo": pesos_mx_opt,
      "Capital Asignado (MXN)": pesos_mx_opt * m_mx_ex,
      col_precio_mx_ex: [precios_mx_actuales[e] for e in activos_validos_mx],
  })
  df_titulos_mx["Títulos a Comprar"] = np.floor(
      df_titulos_mx["Capital Asignado (MXN)"]
      / df_titulos_mx[col_precio_mx_ex]
  ).astype(int)
  st.dataframe(
      estilizar_tabla(
          df_titulos_mx.style.format({
              "Peso Óptimo": "{:.2%}",
              "Capital Asignado (MXN)": "${:,.2f}",
              col_precio_mx_ex: "${:,.2f}",
              "Títulos a Comprar": "{:d}",
          })
      ),
      use_container_width=True,
  )

  st.subheader(
      f"🇺🇸 Asignación de Títulos - Portafolio EE. UU. (${m_usa_ex:,.2f} MXN /"
      " USD)"
  )
  activos_validos_usa = [e for e in empresas_usa if e in usa_p.columns]
  pesos_usa_opt = usa_p[activos_validos_usa].values[0]
  tc_inicial = 18.44
  df_titulos_usa = pd.DataFrame({
      "Empresa": activos_validos_usa,
      "Peso Óptimo": pesos_usa_opt,
      "Capital Asignado (USD)": pesos_usa_opt * (m_usa_ex / tc_inicial),
      col_precio_usa_ex: [precios_usa_actuales[e] for e in activos_validos_usa],
  })
  df_titulos_usa["Títulos a Comprar"] = np.floor(
      df_titulos_usa["Capital Asignado (USD)"]
      / df_titulos_usa[col_precio_usa_ex]
  ).astype(int)
  st.dataframe(
      estilizar_tabla(
          df_titulos_usa.style.format({
              "Peso Óptimo": "{:.2%}",
              "Capital Asignado (USD)": "${:,.2f}",
              col_precio_usa_ex: "${:,.2f}",
              "Títulos a Comprar": "{:d}",
          })
      ),
      use_container_width=True,
  )


# ==============================================================================
# PESTAÑA 5: ANEXO DE FÓRMULAS
# ==============================================================================
with tab_anexo:
  st.header("📐 Anexo Cuantitativo: Fórmulas y Modelos Matemáticos")
  col1, col2 = st.columns(2)
  with col1:
    st.markdown("### 1. Rendimiento y Volatilidad del Portafolio")
    st.latex(
        r"R_p = \sum_{i=1}^{n} w_i \cdot \mu_i = \mathbf{w}^T \boldsymbol{\mu}"
    )
    st.latex(
        r"\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w} ="
        r" \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij}"
    )
    st.latex(r"\sigma_p = \sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}")
    st.markdown("### 2. Ratio de Sharpe & Línea de Asignación de Capital (CAL)")
    st.latex(r"Sharpe = \frac{R_p - R_f}{\sigma_p}")
    st.latex(r"E(R_c) = R_f + Sharpe \cdot \sigma_c")
  with col2:
    st.markdown("### 3. Distribución Triangular (CETES / T-Bills)")
    st.latex(r"k = \frac{b - a}{c - a}")
    st.latex(
        r"T_L = \begin{cases} a + \sqrt{U \cdot (b - a)(c - a)} & \text{si } U"
        r" < k \\ c - \sqrt{(1 - U)(c - a)(c - b)} & \text{si } U \ge k"
        r" \end{cases}"
    )
    st.markdown("### 4. Métricas de Riesgo (Cutoff & VaR)")
    st.latex(r"Cutoff_{\alpha} = Q_{\alpha}(V_{final})")
    st.latex(r"VaR_{\alpha} = \mathbb{E}[V_{final}] - Cutoff_{\alpha}")

  st.markdown("---")
  st.markdown("### 5. Tipo de Cambio Estocástico (Distribución Weibull)")
  st.latex(r"TC = \lambda \cdot \left( \frac{U}{1 - U} \right)^{\frac{1}{k}}")

  st.markdown("---")
  st.markdown(
      "### 6. Bootstrapping No Paramétrico (10,000 Corridas) & Stop-Loss"
      " Individual (-10%)"
  )
  st.latex(
      r"r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right), \quad"
      r" \tilde{r}_{i,d} = r_{i, I_d}, \quad I_d \sim"
      r" \text{UniformeDiscreto}(1, T)"
  )
  st.latex(
      r"V_{i,\text{final}} = \begin{cases} C_{i,0} (1 - 0.10) \cdot"
      r" \left(1 + \frac{\bar{r}_{\text{CETES/T-Bills}}}{360}\right)^{D - t^*} &"
      r" \text{si } \sum_{d=1}^{t^*} \tilde{r}_{i,d} \le -0.10 \\ C_{i,0}"
      r" \cdot \exp\left(\sum_{d=1}^{D} \tilde{r}_{i,d}\right) & \text{en"
      r" otro caso} \end{cases}"
  )
