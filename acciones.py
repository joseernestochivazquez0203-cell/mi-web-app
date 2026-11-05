import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuración de la página en Streamlit
st.set_page_config(
    page_title="Simulador de Portafolios & Monte Carlo",
    page_icon="📈",
    layout="wide",
)

st.title("📊 Simulador de Portafolios de Inversión y Monte Carlo")
st.markdown(
    "Modelo cuantitativo para optimización de portafolios con acciones de la BMV y simulación de CETES."
)

# ==============================================================================
# 1. SIDEBAR: PARÁMETROS Y HORIZONTE TEMPORAL
# ==============================================================================
st.sidebar.header("⚙️ Configuración del Análisis")

horizonte_opcion = st.sidebar.selectbox(
    "Seleccione el Horizonte Temporal:",
    options=["1 año", "2 años", "3 años"],
    index=0,
)

# Mapeo de fechas según el horizonte seleccionado
fechas_horizontes = {
    "1 año": {"inicio": "2025-09-18", "fin": "2026-09-16"},
    "2 años": {"inicio": "2024-09-18", "fin": "2026-09-16"},
    "3 años": {"inicio": "2023-09-18", "fin": "2026-09-16"},
}

fecha_inicio = fechas_horizontes[horizonte_opcion]["inicio"]
fecha_fin = fechas_horizontes[horizonte_opcion]["fin"]

st.sidebar.info(
    f"Período de análisis: **{fecha_inicio}** al **{fecha_fin}** ({horizonte_opcion})"
)

n_simulaciones = st.sidebar.slider(
    "Número de Portafolios (Simulación):", 1000, 10000, 5000, 500
)
n_corridas = st.sidebar.slider(
    "Corridas de Optimización (Sharpe):", 100, 1000, 500, 100
)

# ==============================================================================
# 2. DESCARGA Y PROCESAMIENTO DE DATOS (YFINANCE)
# ==============================================================================
empresas = {
    "GMEXICOB.MX": "Grupo México",
    "LAMOSA.MX": "Lamosa",
    "AMXB.MX": "América Movil",
    "BIMBOA.MX": "Bimbo",
    "KIMBERA.MX": "Kimbera",
}


@st.cache_data
def cargar_datos(start, end):
  datos = yf.download(list(empresas.keys()), start=start, end=end)["Close"]
  # Manejo robusto para índices multinivel de yfinance recientes
  if isinstance(datos.columns, pd.MultiIndex):
    datos.columns = datos.columns.droplevel(0)
  datos.rename(columns=empresas, inplace=True)
  return datos[list(empresas.values())].dropna()


with st.spinner("Descargando datos históricos desde Yahoo Finance..."):
  datos = cargar_datos(fecha_inicio, fecha_fin)

st.subheader("📈 Comportamiento Histórico de los Activos")
st.line_chart(datos)

# ==============================================================================
# 3. CÁLCULO DE RENDIMIENTOS Y ESTADÍSTICOS
# ==============================================================================
rendimientos = np.log(datos / datos.shift(1)).dropna()
promedio = rendimientos.mean()
varianza = rendimientos.var()
desviacion_estandar = rendimientos.std()

estadisticas = pd.DataFrame({
    "Promedio": promedio,
    "Varianza": varianza,
    "Desviación Estándar": desviacion_estandar,
}).T

with st.expander("Ver Estadísticos Descriptivos de los Activos"):
  st.dataframe(estadisticas.style.format("{:.6f}"))

# ==============================================================================
# 4. SIMULACIÓN MONTE CARLO DE CETES Y PORTAFOLIOS
# ==============================================================================
# Parámetros CETES
a, b, c = 0.04, 0.08, 0.10
k = (b - a) / (c - a)
dias_ano = 252

# Generación de pesos aleatorios base
np.random.seed(42)
numeros_aleatorios = pd.DataFrame(
    np.random.uniform(0, 1, size=(len(datos) - 1, datos.shape[1])),
    index=datos.index[:-1],
    columns=datos.columns,
)
wi = numeros_aleatorios.div(numeros_aleatorios.sum(axis=1), axis=0)
matriz_covarianza_wi = wi[list(empresas.values())].cov(ddof=0)
Sigma = matriz_covarianza_wi.values

mejores_sharpe = []
datos_iteraciones = []
colores_iteraciones = px.colors.sample_colorscale(
    "turbo", np.linspace(0.1, 0.9, min(n_corridas, 200))
)

for i in range(1, n_corridas + 1):
  # A) CETES aleatorio
  p_cetes = np.random.uniform(0, 1)
  if p_cetes < k:
    TL = a + np.sqrt(p_cetes * (b - a) * (c - a))
  else:
    TL = c - np.sqrt((1 - p_cetes) * (c - a) * (c - b))

  diario = (1 + TL) ** (1 / 360) - 1
  cetes_anual = TL

  # B) Portafolios simulados
  aleatorios_sim = np.random.uniform(
      0, 1, size=(n_simulaciones, len(empresas))
  )
  wi_sim = aleatorios_sim / aleatorios_sim.sum(axis=1, keepdims=True)
  df_sim = pd.DataFrame(wi_sim, columns=list(empresas.values()))

  df_sim["Rp"] = df_sim[list(empresas.values())].dot(promedio)
  W_sim = df_sim[list(empresas.values())].values
  vp_sim = np.einsum("ij,jk,ik->i", W_sim, Sigma, W_sim)
  df_sim["vp"] = vp_sim
  df_sim["des_p"] = np.sqrt(vp_sim)
  df_sim["sharpe_ratio"] = (df_sim["Rp"] - diario) / df_sim["des_p"]

  idx_max_sharpe = df_sim["sharpe_ratio"].idxmax()
  mejor_fila = df_sim.loc[[idx_max_sharpe]].copy()
  mejor_fila.index = [f"{i}"]
  mejor_fila["CETES_diario"] = diario
  mejores_sharpe.append(mejor_fila)

  if i <= len(colores_iteraciones):
    Rp_anual_sim = df_sim["Rp"] * dias_ano
    des_p_anual_sim = df_sim["des_p"] * np.sqrt(dias_ano)
    sharpe_anual_sim = (Rp_anual_sim - cetes_anual) / des_p_anual_sim
    idx_opt = sharpe_anual_sim.idxmax()

    datos_iteraciones.append({
        "cetes_anual": cetes_anual,
        "nube_x": (df_sim["des_p"] * np.sqrt(dias_ano)).values,
        "nube_y": (df_sim["Rp"] * dias_ano).values,
        "star_x": des_p_anual_sim.loc[idx_opt],
        "star_y": Rp_anual_sim.loc[idx_opt],
        "color": colores_iteraciones[i - 1],
    })

tabla_10_sharpe_max = pd.concat(mejores_sharpe)
columnas_finales = (
    list(empresas.values()) + ["Rp", "vp", "des_p", "CETES_diario", "sharpe_ratio"]
)
tabla_10_sharpe_max = tabla_10_sharpe_max[columnas_finales]

# ==============================================================================
# 5. RESULTADOS Y PORTAFOLIO ÓPTIMO (MÁXIMO SHARPE)
# ==============================================================================
st.markdown("---")
st.subheader(f"🏆 Portafolio de Máximo Sharpe ({horizonte_opcion})")

idx_max_global = tabla_10_sharpe_max["sharpe_ratio"].idxmax()
maxshare1 = tabla_10_sharpe_max.loc[[idx_max_global]].copy()
maxshare1.index = [horizonte_opcion]

st.dataframe(maxshare1.style.format("{:.6f}"))

# ==============================================================================
# 6. VISUALIZACIÓN INTERACTIVA CON PLOTLY
# ==============================================================================
st.markdown("---")
st.subheader("📉 Frontera Eficiente y Simulación")

fig = go.Figure()
# Mostrar la nube de la última corrida como ejemplo visual
if datos_iteraciones:
  ultima = datos_iteraciones[-1]
  fig.add_trace(
      go.Scatter(
          x=ultima["nube_x"],
          y=ultima["nube_y"],
          mode="markers",
          marker=dict(size=4, color="blue", opacity=0.4),
          name="Portafolios Simulados",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=[ultima["star_x"]],
          y=[ultima["star_y"]],
          mode="markers",
          marker=dict(
              size=14, color="gold", symbol="star", line=dict(color="black", width=1)
          ),
          name="Máximo Sharpe",
      )
  )

fig.update_layout(
    title=f"Frontera Eficiente - Horizonte: {horizonte_opcion}",
    xaxis_title="Riesgo Anual (Desviación Estándar)",
    yaxis_title="Rendimiento Anual Esperado",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)
