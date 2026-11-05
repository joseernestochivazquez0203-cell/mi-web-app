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

st.title("📊 Simulador de Portafolios de Inversión y Monte Carlo (1, 2 y 3 Años)")
st.markdown(
    "Modelo cuantitativo para la optimización y comparación multianual de portafolios con activos de la BMV."
)

# ==============================================================================
# 1. SIDEBAR: PARÁMETROS GENERALES
# ==============================================================================
st.sidebar.header("⚙️ Configuración del Análisis")

n_simulaciones = st.sidebar.slider(
    "Número de Portafolios (Simulación):", 1000, 10000, 5000, 500
)
n_corridas = st.sidebar.slider(
    "Corridas de Optimización (Sharpe):", 100, 1000, 500, 100
)

# Definición de las empresas de la BMV
empresas = [
    "Grupo México",
    "Lamosa",
    "América Movil",
    "Bimbo",
    "Kimbera",
]
tickers = {
    "Grupo México": "GMEXICOB.MX",
    "Lamosa": "LAMOSA.MX",
    "América Movil": "AMXB.MX",
    "Bimbo": "BIMBOA.MX",
    "Kimbera": "KIMBERA.MX",
}

# ==============================================================================
# 2. FUNCIÓN DE PROCESAMIENTO MULTIPERIODO (1, 2 y 3 AÑOS)
# ==============================================================================
fechas_horizontes = {
    "1 año": {"inicio": "2025-09-18", "fin": "2026-09-16"},
    "2 años": {"inicio": "2024-09-18", "fin": "2026-09-16"},
    "3 años": {"inicio": "2023-09-18", "fin": "2026-09-16"},
}


@st.cache_data
def ejecutar_simulacion_periodo(start, end, n_sim, n_corr):
  # Descarga de datos
  datos = yf.download(list(tickers.values()), start=start, end=end)["Close"]
  if isinstance(datos.columns, pd.MultiIndex):
    datos.columns = datos.columns.droplevel(0)
  datos.rename(
      columns={v: k for k, v in tickers.items()}, inplace=True
  )
  datos = datos[empresas].dropna()

  # Rendimientos y Estadísticos
  rendimientos = np.log(datos / datos.shift(1)).dropna()
  promedio = rendimientos.mean()
  matriz_covarianza = rendimientos[empresas].cov(ddof=0)
  Sigma = matriz_covarianza.values

  # Parámetros CETES
  a, b, c = 0.04, 0.08, 0.10
  k = (b - a) / (c - a)
  dias_ano = 252

  # Pesos base para covarianza de pesos (o simulación directa)
  np.random.seed(42)
  aleatorios_base = np.random.uniform(
      0, 1, size=(len(datos) - 1, len(empresas))
  )
  wi_base = aleatorios_base / aleatorios_base.sum(axis=1, keepdims=True)
  matriz_cov_wi = pd.DataFrame(wi_base, columns=empresas).cov(ddof=0)
  Sigma_wi = matriz_cov_wi.values

  mejores_sharpe = []
  df_sim_global = None

  for i in range(1, n_corr + 1):
    p_cetes = np.random.uniform(0, 1)
    if p_cetes < k:
      TL = a + np.sqrt(p_cetes * (b - a) * (c - a))
    else:
      TL = c - np.sqrt((1 - p_cetes) * (c - a) * (c - b))

    diario = (1 + TL) ** (1 / 360) - 1

    aleatorios_sim = np.random.uniform(0, 1, size=(n_sim, len(empresas)))
    wi_sim = aleatorios_sim / aleatorios_sim.sum(axis=1, keepdims=True)
    df_sim = pd.DataFrame(wi_sim, columns=empresas)

    df_sim["Rp"] = df_sim[empresas].dot(promedio)
    W_sim = df_sim[empresas].values
    vp_sim = np.einsum("ij,jk,ik->i", W_sim, Sigma_wi, W_sim)
    df_sim["vp"] = vp_sim
    df_sim["des_p"] = np.sqrt(vp_sim)
    df_sim["sharpe_ratio"] = (df_sim["Rp"] - diario) / df_sim["des_p"]

    if i == 1:
      df_sim_global = df_sim.copy()  # Guardar para la frontera eficiente

    idx_max_sharpe = df_sim["sharpe_ratio"].idxmax()
    mejor_fila = df_sim.loc[[idx_max_sharpe]].copy()
    mejor_fila.index = [f"{i}"]
    mejor_fila["CETES_diario"] = diario
    mejores_sharpe.append(mejor_fila)

  tabla_sharpe_max = pd.concat(mejores_sharpe)

  # 1. Portafolio de Mínima Desviación del periodo
  idx_min_desp = df_sim_global["des_p"].idxmin()
  min_port = df_sim_global.loc[[idx_min_desp]].copy()

  # 2. Portafolio de Máximo Sharpe global entre las corridas
  idx_max_global = tabla_sharpe_max["sharpe_ratio"].idxmax()
  max_sharpe_port = tabla_sharpe_max.loc[[idx_max_global]].copy()

  return min_port, max_sharpe_port, df_sim_global, promedio, Sigma


with st.spinner(
    "Calculando simulaciones para 1, 2 y 3 años (esto puede tomar unos"
    " segundos)..."
):
  # Ejecutar de forma automática para los 3 periodos
  res_1yr = ejecutar_simulacion_periodo(
      fechas_horizontes["1 año"]["inicio"],
      fechas_horizontes["1 año"]["fin"],
      n_simulaciones,
      n_corridas,
  )
  res_2yr = ejecutar_simulacion_periodo(
      fechas_horizontes["2 años"]["inicio"],
      fechas_horizontes["2 años"]["fin"],
      n_simulaciones,
      n_corridas,
  )
  res_3yr = ejecutar_simulacion_periodo(
      fechas_horizontes["3 años"]["inicio"],
      fechas_horizontes["3 años"]["fin"],
      n_simulaciones,
      n_corridas,
  )

  min_1yr, maxshare1, df_sim_1yr, prom_1, sig_1 = res_1yr
  min_2yr, maxshare2, df_sim_2yr, prom_2, sig_2 = res_2yr
  min_3yr, maxshare3, df_sim_3yr, prom_3, sig_3 = res_3yr

# Renombrar índices para las tablas comparativas
min_1yr.index, min_2yr.index, min_3yr.index = ["1 año"], ["2 años"], ["3 años"]
maxshare1.index, maxshare2.index, maxshare3.index = (
    ["1 año"],
    ["2 años"],
    ["3 años"],
)

# ==============================================================================
# 3. TABLAS COMPARATIVAS (MÍNIMA DESVIACIÓN Y MÁXIMO SHARPE)
# ==============================================================================
st.markdown("---")
st.subheader(
    "📉 Tabla Comparativa de Portafolios Óptimos (Mínima Desviación)"
)

tabla_comparativa_min = pd.concat([min_1yr, min_2yr, min_3yr])
columnas_ordenadas_min = empresas + ["Rp", "vp", "des_p"]
# Asegurar que existan solo las columnas requeridas
tabla_comparativa_min = tabla_comparativa_min[
    [c for c in columnas_ordenadas_min if c in tabla_comparativa_min.columns]
]
st.dataframe(tabla_comparativa_min.style.format("{:.6f}"))

st.markdown("---")
st.subheader("🏆 Tabla Comparativa de Máximos Sharpe Ratios (1, 2 y 3 Años)")

tabla_comparativa_sharpe = pd.concat([maxshare1, maxshare2, maxshare3])
columnas_orden_sharpe = empresas + [
    "Rp",
    "vp",
    "des_p",
    "CETES_diario",
    "sharpe_ratio",
]
tabla_comparativa_sharpe = tabla_comparativa_sharpe[
    [
        c
        for c in columnas_orden_sharpe
        if c in tabla_comparativa_sharpe.columns
    ]
]
st.dataframe(tabla_comparativa_sharpe.style.format("{:.6f}"))

# ==============================================================================
# 4. VISUALIZACIÓN INTERACTIVA DE FRONTERA EFICIENTE POR HORIZONTE
# ==============================================================================
st.markdown("---")
st.subheader("📊 Visualización de Fronteras Eficientes")

horizonte_seleccionado = st.selectbox(
    "Seleccione el horizonte temporal a graficar:",
    options=["1 año", "2 años", "3 años"],
    index=0,
)

# Mapeo de datos seleccionados
if horizonte_seleccionado == "1 año":
  df_actual = df_sim_1yr
  star_actual = maxshare1
elif horizonte_seleccionado == "2 años":
  df_actual = df_sim_2yr
  star_actual = maxshare2
else:
  df_actual = df_sim_3yr
  star_actual = maxshare3

dias_ano = 252
fig = go.Figure()

# Nube de puntos simulados anualizados
fig.add_trace(
    go.Scatter(
        x=df_actual["des_p"] * np.sqrt(dias_ano),
        y=df_actual["Rp"] * dias_ano,
        mode="markers",
        marker=dict(size=4, color="blue", opacity=0.4),
        name="Portafolios Simulados",
    )
)

# Estrella del Máximo Sharpe
fig.add_trace(
    go.Scatter(
        x=star_actual["des_p"] * np.sqrt(dias_ano),
        y=star_actual["Rp"] * dias_ano,
        mode="markers",
        marker=dict(
            size=14, color="gold", symbol="star", line=dict(color="black", width=1)
        ),
        name="Máximo Sharpe",
    )
)

fig.update_layout(
    title=f"Frontera Eficiente y Portafolio Óptimo - {horizonte_seleccionado}",
    xaxis_title="Riesgo Anual (Desviación Estándar)",
    yaxis_title="Rendimiento Anual Esperado",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)
