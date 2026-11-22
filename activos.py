import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Matriz de Correlación — México y EE. UU.",
    page_icon="🔗",
    layout="wide",
)

fecha_hoy = datetime.datetime.now(ZoneInfo("America/Mexico_City")).date()

st.title("🔗 Matriz de Correlación: Empresas de México y EE. UU.")
st.markdown(
    "Vista enfocada exclusivamente en correlaciones históricas entre activos"
    " cotizados en México (BMV) y EE. UU."
)

# ==============================================================================
# 1. CATÁLOGO AMPLIO DE EMISORAS (MÉXICO Y EE. UU.)
# ==============================================================================
DEFAULT_MX = [
    "WALMEX.MX", "AMXB.MX", "FEMSAUBD.MX", "GFNORTEO.MX", "GMEXICOB.MX",
    "CEMEXCPO.MX", "BIMBOA.MX", "GAPB.MX", "ASURB.MX", "OMAB.MX",
    "AC.MX", "KOFUBL.MX", "KIMBERA.MX", "GFINBURO.MX", "GENTERA.MX",
    "PINFRA.MX", "ALFAA.MX", "CUERVO.MX", "LABB.MX", "CHDRAUIB.MX",
    "SORIANAB.MX", "LIVEPOLC-1.MX", "GCARSOA1.MX", "BBAJIOO.MX", "RA.MX",
    "FUNO11.MX", "FMTY14.MX", "TERRA13.MX", "DANHOS13.MX", "MEGACPO.MX",
    "TLEVISACPO.MX", "VOLARA.MX", "ALSEA.MX", "GRUMAB.MX", "BOLSAA.MX",
    "VESTA.MX", "ORBIA.MX", "PE&OLES.MX", "GCC.MX", "SITESB-1.MX",
]

DEFAULT_USA = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "LLY",
    "JPM", "V", "MA", "XOM", "UNH", "WMT", "COST", "HD", "PG", "JNJ",
    "NFLX", "CRM", "AMD", "INTC", "ORCL", "ADBE", "BAC", "WFC", "GS", "MS",
    "CVX", "MRK", "ABBV", "PEP", "KO", "TMO", "ACN", "MCD", "DIS", "ABT",
    "CSCO", "LIN", "PM", "TXN", "NEE", "DHR", "AMGN", "RTX", "CAT", "IBM",
    "QCOM", "SPGI", "LOW", "BA", "UBER", "PLTR", "COIN", "PYPL", "SBUX", "BKNG",
]

# ==============================================================================
# 2. CONFIGURACIÓN EN BARRA LATERAL
# ==============================================================================
st.sidebar.header("⚙️ Parámetros")

horizonte_dias = st.sidebar.selectbox(
    "Ventana histórica:",
    options={
        "6 meses": 180,
        "1 año": 365,
        "2 años": 730,
        "3 años": 1095,
        "5 años": 1825,
    }.items(),
    format_func=lambda x: x[0],
    index=1,
)[1]

tipo_rend = st.sidebar.radio(
    "Tipo de rendimiento para correlación:",
    ["Logarítmicos ln(Pt / Pt-1)", "Porcentuales (Pct Change)"],
    index=0,
)

metodo_corr = st.sidebar.selectbox(
    "Método de correlación:",
    ["pearson", "spearman", "kendall"],
    index=0,
)

fecha_inicio = fecha_hoy - datetime.timedelta(days=horizonte_dias)
fecha_fin = fecha_hoy

# ==============================================================================
# 3. SELECCIÓN DE TICKERS + EXTRAS PERSONALIZADOS
# ==============================================================================
col_mx, col_usa = st.columns(2)

with col_mx:
  sel_mx = st.multiselect(
      "🇲🇽 Empresas México (BMV)",
      options=DEFAULT_MX,
      default=DEFAULT_MX[:15],
  )
  extra_mx = st.text_input(
      "Agregar tickers MX extra (separados por coma):",
      placeholder="Ej. BOLSA.MX, CULTIBA.MX",
  )

with col_usa:
  sel_usa = st.multiselect(
      "🇺🇸 Empresas EE. UU.",
      options=DEFAULT_USA,
      default=DEFAULT_USA[:15],
  )
  extra_usa = st.text_input(
      "Agregar tickers EE. UU. extra (separados por coma):",
      placeholder="Ej. SOFI, SNOW, CRWD",
  )

tickers_mx = sorted(
    set(sel_mx + [t.strip().upper() for t in extra_mx.split(",") if t.strip()])
)
tickers_usa = sorted(
    set(sel_usa + [t.strip().upper() for t in extra_usa.split(",") if t.strip()])
)
todos_tickers = list(dict.fromkeys(tickers_mx + tickers_usa))

if len(todos_tickers) < 2:
  st.warning("Selecciona al menos 2 empresas para calcular la matriz de correlación.")
  st.stop()


# ==============================================================================
# 4. DESCARGA DE PRECIOS Y CÁLCULO DE MATRIZ
# ==============================================================================
@st.cache_data(ttl=3600)
def descargar_precios(tickers_tuple, start_str, end_str):
  raw = yf.download(
      list(tickers_tuple),
      start=start_str,
      end=end_str,
      auto_adjust=True,
      progress=False,
  )
  if raw.empty:
    return pd.DataFrame()

  if isinstance(raw.columns, pd.MultiIndex):
    if "Close" in raw.columns.get_level_values(0):
      df = raw["Close"]
    else:
      df = raw.xs(raw.columns.get_level_values(0)[0], axis=1, level=0)
  else:
    df = raw["Close"] if "Close" in raw.columns else raw

  if isinstance(df, pd.Series):
    df = df.to_frame(name=tickers_tuple[0])

  df.columns = [str(c).replace(".MX", "") if str(c).endswith(".MX") else str(c) for c in df.columns]
  df = df.dropna(axis=1, how="all")
  return df


with st.spinner("Descargando precios históricos y calculando correlaciones..."):
  precios = descargar_precios(
      tuple(todos_tickers),
      fecha_inicio.strftime("%Y-%m-%d"),
      fecha_fin.strftime("%Y-%m-%d"),
  )

if precios.empty or precios.shape[1] < 2:
  st.error("No se pudieron obtener suficientes datos de precios para los tickers seleccionados.")
  st.stop()

if tipo_rend.startswith("Logarítmicos"):
  rendimientos = np.log(precios / precios.shift(1)).dropna(how="all")
else:
  rendimientos = precios.pct_change().dropna(how="all")

cols_mx_limpias = [t.replace(".MX", "") for t in tickers_mx]
cols_usa_limpias = [t.replace(".MX", "") for t in tickers_usa]

cols_mx_disp = [c for c in cols_mx_limpias if c in rendimientos.columns]
cols_usa_disp = [c for c in cols_usa_limpias if c in rendimientos.columns]


def mostrar_seccion_correlacion(df_r, cols_subset, titulo):
  if len(cols_subset) < 2:
    st.info(f"Se necesitan al menos 2 activos válidos para {titulo}.")
    return

  corr = df_r[cols_subset].corr(method=metodo_corr)
  mostrar_numeros = len(cols_subset) <= 20

  fig = px.imshow(
      corr,
      x=corr.columns,
      y=corr.index,
      color_continuous_scale="RdBu_r",
      zmin=-1,
      zmax=1,
      text_auto=".2f" if mostrar_numeros else False,
      aspect="auto",
      title=titulo,
  )
  fig.update_layout(
      height=max(450, min(900, 28 * len(cols_subset) + 120)),
      xaxis_title="",
      yaxis_title="",
  )
  st.plotly_chart(fig, use_container_width=True)

  with st.expander(f"Ver tabla numérica — {titulo}"):
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

  # Top pares con mayor y menor correlación
  mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
  pares = (
      corr.where(mask)
      .stack()
      .reset_index()
      .rename(columns={"level_0": "Activo A", "level_1": "Activo B", 0: "Correlación"})
  )
  if not pares.empty:
    c1, c2 = st.columns(2)
    with c1:
      st.markdown("**Mayor correlación**")
      st.dataframe(
          pares.sort_values("Correlación", ascending=False).head(5).style.format({"Correlación": "{:.2f}"}),
          use_container_width=True,
          hide_index=True,
      )
    with c2:
      st.markdown("**Menor correlación**")
      st.dataframe(
          pares.sort_values("Correlación", ascending=True).head(5).style.format({"Correlación": "{:.2f}"}),
          use_container_width=True,
          hide_index=True,
      )


# ==============================================================================
# 5. PESTAÑAS DE CORRELACIÓN
# ==============================================================================
t_comb, t_mx, t_usa = st.tabs([
    "🌐 Combinada (MX + EE. UU.)",
    "🇲🇽 Solo México",
    "🇺🇸 Solo EE. UU.",
])

with t_comb:
  cols_comb = list(dict.fromkeys(cols_mx_disp + cols_usa_disp))
  mostrar_seccion_correlacion(
      rendimientos, cols_comb, "Matriz de Correlación Combinada (MX + EE. UU.)"
  )

with t_mx:
  mostrar_seccion_correlacion(
      rendimientos, cols_mx_disp, "Matriz de Correlación — México"
  )

with t_usa:
  mostrar_seccion_correlacion(
      rendimientos, cols_usa_disp, "Matriz de Correlación — EE. UU."
  )
