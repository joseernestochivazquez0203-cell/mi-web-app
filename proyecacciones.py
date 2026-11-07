import datetime
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Configuración de la página en Streamlit
st.set_page_config(
    page_title="Simulador de Portafolios Internacionales & Riesgo Estocástico",
    page_icon="📈",
    layout="wide",
)

# Estilo CSS personalizado para hacer las "cajitas / botones" más bonitos y profesionales
st.markdown("""
    <style>
    .stock-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .stock-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        border-color: #3498db;
    }
    .stock-title {
        font-size: 16px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    .stock-ticker {
        font-size: 13px;
        color: #7f8c8d;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

st.title(
    "📊 Simulador Integral de Portafolios, Optimización y Riesgo Estocástico"
)
st.markdown(
    "Modelo cuantitativo multianual con Markowitz, Frontera Eficiente, Simulación"
    " Monte Carlo y Métricas de Riesgo."
)

# ==============================================================================
# 1. BARRA LATERAL (SIDEBAR): PARÁMETROS GENERALES Y TICKERS PERSONALIZABLES
# ==============================================================================
st.sidebar.header("⚙️ Configuración del Análisis")

n_simulaciones = st.sidebar.slider(
    "Número de Portafolios (Simulación):", 1000, 10000, 5000, 500
)
n_corridas = st.sidebar.slider(
    "Corridas de Optimización (Sharpe):", 100, 1000, 500, 100
)

st.sidebar.subheader("🎯 Parámetros de Riesgo (VaR & Cutoff)")
alpha_nivel = (
    st.sidebar.slider(
        "Nivel de Significancia (Alpha %):", 1.0, 15.0, 5.0, 0.5
    )
    / 100.0
)

st.sidebar.subheader("📅 Proyección Estocástica Futura")
fecha_inicio_sim = datetime.date.today()
fecha_fin_sim = st.sidebar.date_input(
    "Fecha Final de Simulación:",
    value=fecha_inicio_sim + datetime.timedelta(days=365),
    min_value=fecha_inicio_sim + datetime.timedelta(days=1),
)

dias_totales = (fecha_fin_sim - fecha_inicio_sim).days
dias_habiles = int(dias_totales * (252 / 365))
st.sidebar.info(
    f"📆 Días calendario: **{dias_totales}** | Días hábiles estimados:"
    f" **{dias_habiles}**"
)

st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Personalización de Tickers (Exactamente 5 o más)")
st.sidebar.markdown(
    "*(Modifica o escribe los tickers separados por comas tal como en Yahoo"
    " Finance)*"
)

# EXACTAMENTE 5 POR DEFECTO PARA EVITAR ERRORES DE DATOS VACÍOS
default_tickers_mx_str = "LAMOSA.MX, WALMEX.MX, AMXB.MX, BIMBOA.MX, FEMSAUBD.MX"
default_tickers_usa_str = "MSFT, AAPL, GOOGL, AMZN, NVDA"

input_mx_str = st.sidebar.text_area(
    "Tickers México (BMV):", value=default_tickers_mx_str, height=70
)
input_usa_str = st.sidebar.text_area(
    "Tickers EE. UU. (Tech):", value=default_tickers_usa_str, height=70
)

tickers_mx_list = [t.strip() for t in input_mx_str.split(",") if t.strip()]
tickers_usa_list = [t.strip() for t in input_usa_str.split(",") if t.strip()]

tickers_mx = {t.split(".")[0]: t for t in tickers_mx_list}
empresas_mx = list(tickers_mx.keys())

tickers_usa = {t: t for t in tickers_usa_list}
empresas_usa = list(tickers_usa.keys())

fechas_horizontes = {
    "1 año": {
        "inicio": (
            fecha_inicio_sim - datetime.timedelta(days=365)
        ).strftime("%Y-%m-%d"),
        "fin": fecha_inicio_sim.strftime("%Y-%m-%d"),
    },
    "2 años": {
        "inicio": (
            fecha_inicio_sim - datetime.timedelta(days=730)
        ).strftime("%Y-%m-%d"),
        "fin": fecha_inicio_sim.strftime("%Y-%m-%d"),
    },
    "3 años": {
        "inicio": (
            fecha_inicio_sim - datetime.timedelta(days=1095)
        ).strftime("%Y-%m-%d"),
        "fin": fecha_inicio_sim.strftime("%Y-%m-%d"),
    },
}


# ==============================================================================
# 2. FUNCIÓN SEGURA PARA OBTENER PRECIOS ACTUALES
# ==============================================================================
def obtener_precio_actual(ticker, default_val=100.0):
  try:
    df_p = yf.download(ticker, period="5d", progress=False)["Close"]
    if isinstance(df_p, pd.DataFrame):
      df_p = df_p.iloc[:, 0]
    val = float(df_p.dropna().iloc[-1])
    return val if not np.isnan(val) else default_val
  except:
    return default_val


# ==============================================================================
# 3. FUNCIÓN DE SIMULACIÓN Y OPTIMIZACIÓN (MARKOWITZ) SEGURA ANTE NA
# ==============================================================================
@st.cache_data
def ejecutar_simulacion_mercado(
    tickers_dict, lista_empresas, start, end, n_sim, n_corr, tipo_tasa="cetes"
):
  raw_data = yf.download(list(tickers_dict.values()), start=start, end=end)
  if isinstance(raw_data, pd.DataFrame) and "Close" in raw_data.columns:
    datos = raw_data["Close"]
  else:
    datos = raw_data

  if isinstance(datos, pd.DataFrame):
    if isinstance(datos.columns, pd.MultiIndex):
      datos.columns = datos.columns.droplevel(0)
    datos.rename(columns={v: k for k, v in tickers_dict.items()}, inplace=True)
    # Limpiar estrictamente columnas con NAs o vacías
    datos = datos[[c for c in lista_empresas if c in datos.columns]].dropna(
        how="all"
    )
    datos = datos.dropna(axis=1, how="any")
  else:
    datos = pd.DataFrame(datos)
    datos.columns = [lista_empresas[0]]
    datos = datos.dropna()

  if datos.empty or datos.shape[1] == 0:
    # Fallback seguro si faltan datos
    datos = pd.DataFrame(
        np.random.normal(100, 2, size=(100, len(lista_empresas))),
        columns=lista_empresas,
    )

  rendimientos = np.log(datos / datos.shift(1)).dropna()
  promedio = rendimientos.mean()
  dias_ano = 252
  Sigma = rendimientos.cov(ddof=0).values * dias_ano

  if tipo_tasa == "cetes":
    a, b, c = 0.04, 0.08, 0.10
  else:
    a, b, c = 0.0360, 0.0370, 0.0400

  k = (b - a) / (c - a)
  mejores_sharpe = []
  datos_animacion = []
  n_animacion_pasos = min(30, n_corr)
  colores_anim = px.colors.sample_colorscale(
      "turbo", np.linspace(0.1, 0.9, n_animacion_pasos)
  )

  w_cetes = 0.30
  w_portafolio = 0.70
  df_sim_global = None
  columnas_activas = list(datos.columns)

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

    df_sim["Rp"] = df_sim[columnas_activas].dot(promedio[columnas_activas]) * dias_ano
    W_sim = df_sim[columnas_activas].values
    vp_sim = np.einsum("ij,jk,ik->i", W_sim, Sigma, W_sim)
    df_sim["vp"] = vp_sim
    df_sim["des_p"] = np.sqrt(vp_sim)
    df_sim["sharpe_ratio"] = (df_sim["Rp"] - tasa_anual) / df_sim["des_p"]

    if i == 1:
      df_sim_global = df_sim.copy()

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
      ret_comb = (w_cetes * tasa_anual) + (w_portafolio * ret_opt_anual)

      df_vis = df_sim.sample(n=min(1500, len(df_sim))).copy()

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
  idx_min_desp = df_sim_global["des_p"].idxmin()
  min_port = df_sim_global.loc[[idx_min_desp]].copy()

  idx_max_global = tabla_sharpe_max["sharpe_ratio"].idxmax()
  max_sharpe_port = tabla_sharpe_max.loc[[idx_max_global]].copy()

  return min_port, max_sharpe_port, datos_animacion, tabla_sharpe_max, datos, Sigma


with st.spinner("Ejecutando simulaciones con los tickers personalizados..."):
  res_mx_1 = ejecutar_simulacion_mercado(
      tickers_mx,
      empresas_mx,
      fechas_horizontes["1 año"]["inicio"],
      fechas_horizontes["1 año"]["fin"],
      n_simulaciones,
      n_corridas,
      "cetes",
  )
  res_mx_2 = ejecutar_simulacion_mercado(
      tickers_mx,
      empresas_mx,
      fechas_horizontes["2 años"]["inicio"],
      fechas_horizontes["2 años"]["fin"],
      n_simulaciones,
      n_corridas,
      "cetes",
  )
  res_mx_3 = ejecutar_simulacion_mercado(
      tickers_mx,
      empresas_mx,
      fechas_horizontes["3 años"]["inicio"],
      fechas_horizontes["3 años"]["fin"],
      n_simulaciones,
      n_corridas,
      "cetes",
  )

  res_usa_1 = ejecutar_simulacion_mercado(
      tickers_usa,
      empresas_usa,
      fechas_horizontes["1 año"]["inicio"],
      fechas_horizontes["1 año"]["fin"],
      n_simulaciones,
      n_corridas,
      "tbills",
  )
  res_usa_2 = ejecutar_simulacion_mercado(
      tickers_usa,
      empresas_usa,
      fechas_horizontes["2 años"]["inicio"],
      fechas_horizontes["2 años"]["fin"],
      n_simulaciones,
      n_corridas,
      "tbills",
  )
  res_usa_3 = ejecutar_simulacion_mercado(
      tickers_usa,
      empresas_usa,
      fechas_horizontes["3 años"]["inicio"],
      fechas_horizontes["3 años"]["fin"],
      n_simulaciones,
      n_corridas,
      "tbills",
  )

  min_mx_1, max_mx_1, anim_mx_1, t1000_mx_1, datos_mx_1, cov_mx_1 = res_mx_1
  min_mx_2, max_mx_2, anim_mx_2, t1000_mx_2, datos_mx_2, cov_mx_2 = res_mx_2
  min_mx_3, max_mx_3, anim_mx_3, t1000_mx_3, datos_mx_3, cov_mx_3 = res_mx_3

  min_usa_1, max_usa_1, anim_usa_1, t1000_usa_1, datos_usa_1, cov_usa_1 = (
      res_usa_1
  )
  min_usa_2, max_usa_2, anim_usa_2, t1000_usa_2, datos_usa_2, cov_usa_2 = (
      res_usa_2
  )
  min_usa_3, max_usa_3, anim_usa_3, t1000_usa_3, datos_usa_3, cov_usa_3 = (
      res_usa_3
  )

min_mx_1.index, min_mx_2.index, min_mx_3.index = ["1 año"], ["2 años"], ["3 años"]
max_mx_1.index, max_mx_2.index, max_mx_3.index = ["1 año"], ["2 años"], ["3 años"]
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
# 4. CONSTRUCTOR DE GRÁFICA ANIMADA (FRONTERA EFICIENTE)
# ==============================================================================
def construir_figura_animada_con_minima(
    datos_anim,
    min_port_df,
    lista_empresas,
    titulo_horizonte,
    lbl_tasa="CETES",
    is_mx=False,
):
  fig = go.Figure()
  n_pasos = len(datos_anim)

  fig.add_trace(
      go.Scatter(
          x=[],
          y=[],
          mode="markers",
          marker=dict(size=3.5, color="#3498db", opacity=0.4),
          name="Portafolios Simulados",
      )
  )
  for j in range(n_pasos):
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="rgba(231, 76, 60, 0.3)", width=1),
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
              color="gold",
              symbol="star",
              line=dict(width=1, color="black"),
          ),
          name="Máximo Sharpe",
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
              size=12,
              color="blue",
              symbol="circle",
              line=dict(width=1.5, color="black"),
          ),
          name="🔵 Mínima Varianza",
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
            marker=dict(size=3.5, color="#3498db"),
        )
    )
    for j in range(n_pasos):
      if j <= f_idx:
        f_data.append(
            go.Scatter(x=datos_anim[j]["x_cal"], y=datos_anim[j]["y_cal"])
        )
      else:
        f_data.append(go.Scatter(x=[], y=[]))

    stars_x = [datos_anim[j]["star_x"] for j in range(f_idx + 1)]
    stars_y = [datos_anim[j]["star_y"] for j in range(f_idx + 1)]
    f_data.append(go.Scatter(x=stars_x, y=stars_y))
    f_data.append(go.Scatter(x=[min_x], y=[min_y]))

    frames.append(go.Frame(data=f_data, name=f"Paso {f_idx + 1}"))

  for idx, t_data in enumerate(frames[0].data):
    fig.data[idx].x = t_data.x
    fig.data[idx].y = t_data.y

  fig.update_layout(
      title=f"Frontera Eficiente & Monte Carlo ({titulo_horizonte})",
      template="plotly_white",
      xaxis=dict(title="Volatilidad (Riesgo Anual)", range=[0, 1.2]),
      yaxis=dict(title="Rendimiento Anual", range=[-0.5, 0.8]),
      updatemenus=[{
          "type": "buttons",
          "buttons": [
              {
                  "label": "▶ Reproducir",
                  "method": "animate",
                  "args": [
                      None,
                      {
                          "frame": {"duration": 400, "redraw": False},
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
  )
  fig.frames = frames
  return fig


# ==============================================================================
# 5. SIMULADOR ESTOCÁSTICO DIARIO Y ESTADÍSTICOS
# ==============================================================================
def generar_simulacion_y_estadisticos(
    max_mx,
    max_usa,
    n_dias,
    alpha,
    num_simulaciones_futuras=1000,
    enfoque="global",
):
  tc_inicial = 18.44
  shape_k = 28.52977
  scale_lambda = 18.32123

  monto_mx_acc = 500000.0 if enfoque in ["global", "mexico"] else 0.0
  monto_cetes = 150000.0 if enfoque in ["global", "mexico"] else 0.0
  monto_usa_acc_mxn = 245000.0 if enfoque in ["global", "usa"] else 0.0
  monto_tbills_mxn = 105000.0 if enfoque in ["global", "usa"] else 0.0

  capital_total = monto_mx_acc + monto_cetes + monto_usa_acc_mxn + monto_tbills_mxn
  if capital_total == 0:
    capital_total = 1000000.0

  mu_mx = max_mx["Rp"].values[0] / 252
  sigma_mx = max_mx["des_p"].values[0] / np.sqrt(252)
  rend_sim_mx = np.random.normal(
      mu_mx, sigma_mx, size=(num_simulaciones_futuras, n_dias)
  )
  val_final_mx_acc = monto_mx_acc * np.exp(np.sum(rend_sim_mx, axis=1))

  a_c, b_c, c_c = 0.07, 0.08, 0.09
  k_c = (b_c - a_c) / (c_c - a_c)
  u_c = np.random.uniform(0, 1, size=(num_simulaciones_futuras, n_dias))
  tasas_cetes = np.where(
      u_c < k_c,
      a_c + np.sqrt(u_c * (b_c - a_c) * (c_c - a_c)),
      c_c - np.sqrt((1 - u_c) * (c_c - a_c) * (c_c - b_c)),
  )
  factor_cetes = np.prod(1 + tasas_cetes / 360, axis=1)
  val_final_cetes = monto_cetes * factor_cetes

  monto_usa_acc_usd = monto_usa_acc_mxn / tc_inicial
  mu_usa = max_usa["Rp"].values[0] / 252
  sigma_usa = max_usa["des_p"].values[0] / np.sqrt(252)
  rend_sim_usa = np.random.normal(
      mu_usa, sigma_usa, size=(num_simulaciones_futuras, n_dias)
  )
  val_final_usa_acc_usd = monto_usa_acc_usd * np.exp(
      np.sum(rend_sim_usa, axis=1)
  )

  monto_tbills_usd = monto_tbills_mxn / tc_inicial
  a_t, b_t, c_t = 0.0365, 0.0370, 0.0375
  k_t = (b_t - a_t) / (c_t - a_t)
  u_t = np.random.uniform(0, 1, size=(num_simulaciones_futuras, n_dias))
  tasas_tbills = np.where(
      u_t < k_t,
      a_t + np.sqrt(u_t * (b_t - a_t) * (c_t - a_t)),
      c_t - np.sqrt((1 - u_t) * (c_t - a_t) * (c_t - b_t)),
  )
  factor_tbills = np.prod(1 + tasas_tbills / 360, axis=1)
  val_final_tbills_usd = monto_tbills_usd * factor_tbills

  u_tc = np.random.uniform(0.001, 0.999, size=num_simulaciones_futuras)
  tc_final = scale_lambda * (u_tc / (1 - u_tc)) ** (1 / shape_k)

  val_final_usa_total_mxn = (
      val_final_usa_acc_usd + val_final_tbills_usd
  ) * tc_final
  val_final_global_mxn = (
      val_final_mx_acc + val_final_cetes + val_final_usa_total_mxn
  )

  rend_global = (val_final_global_mxn - capital_total) / capital_total
  rend_mx_acc = (
      (val_final_mx_acc - monto_mx_acc) / monto_mx_acc
      if monto_mx_acc > 0
      else np.zeros_like(rend_global)
  )
  rend_usa_usd = (
      (val_final_usa_acc_usd - monto_usa_acc_usd) / monto_usa_acc_usd
      if monto_usa_acc_usd > 0
      else np.zeros_like(rend_global)
  )
  rend_cetes = (
      (val_final_cetes - monto_cetes) / monto_cetes
      if monto_cetes > 0
      else np.zeros_like(rend_global)
  )
  rend_cambiario = (tc_final - tc_inicial) / tc_inicial

  def calcular_fila_stats(arr):
    rend = np.mean(arr)
    riesgo = np.std(arr)
    cutoff = np.percentile(arr, alpha * 100)
    var_alpha = rend - cutoff
    q1 = np.percentile(arr, 25)
    q2 = np.median(arr)
    q3 = np.percentile(arr, 75)
    v_max = np.max(arr)
    v_min = np.min(arr)
    return [rend, riesgo, cutoff, var_alpha, q1, q2, q3, v_max, v_min]

  filas_stats = [
      calcular_fila_stats(rend_global),
      calcular_fila_stats(rend_mx_acc),
      calcular_fila_stats(rend_usa_usd),
      calcular_fila_stats(rend_cetes),
      calcular_fila_stats(rend_cambiario),
  ]

  df_estadisticos = pd.DataFrame(
      filas_stats,
      index=[
          "portafolio final",
          "portafolio pesos",
          "portafolio dolares",
          "portafolio pesos (cetes)",
          "ganancia por rendimiento cambiario",
      ],
      columns=[
          "rendimiento",
          "riesgo",
          f"{alpha*100:.1f}% CUTOFF",
          f"{alpha*100:.1f}% VAR",
          "Q1",
          "Q2",
          "Q3",
          "max",
          "min",
      ],
  )
  return df_estadisticos, rend_global, val_final_global_mxn


# ==============================================================================
# 6. DEFINICIÓN DE PESTAÑAS (INCLUYENDO LAS 5 CAJITAS / BOTONES ESTÉTICOS)
# ==============================================================================
tab_mx, tab_usa, tab_port_enf, tab_excel, tab_anexo = st.tabs([
    "🇲🇽 Mercado Mexicano (BMV)",
    "🇺🇸 Mercado EE. UU. (Tech)",
    "🌐 Simulación por Enfoque (México, USA & Global)",
    "📑 Modelo Excel Estocástico & Títulos",
    "📐 Anexo de Fórmulas & Parámetros Quant",
])

# Formatos de despliegue
formatos_mx = {c: "{:.2%}" for c in empresas_mx}
formatos_mx.update({
    "Rp": "{:.4%}",
    "des_p": "{:.4%}",
    "Tasa_diaria": "{:.4%}",
    "Tasa_anual": "{:.2%}",
    "sharpe_ratio": "{:.4f}",
})
cols_min_mx = [c for c in empresas_mx + ["Rp", "vp", "des_p"]]
cols_sh_mx = [
    c
    for c in empresas_mx
    + ["Rp", "vp", "des_p", "Tasa_diaria", "Tasa_anual", "sharpe_ratio"]
]

formatos_usa = {c: "{:.2%}" for c in empresas_usa}
formatos_usa.update({
    "Rp": "{:.4%}",
    "des_p": "{:.4%}",
    "Tasa_diaria": "{:.4%}",
    "Tasa_anual": "{:.4%}",
    "sharpe_ratio": "{:.4f}",
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
  st.header("🇲🇽 Análisis de Portafolios - Activos Mexicanos y CETES")

  # MOSTRAR LAS 5 EMPRESAS EN CAJITAS / BOTONES ESTÉTICOS
  st.markdown("### 🏢 Acciones Activas en el Portafolio (México)")
  cols_cards_mx = st.columns(5)
  for idx, emp in enumerate(empresas_mx[:5]):
    with cols_cards_mx[idx]:
      ticker_val = tickers_mx.get(emp, emp)
      st.markdown(
          f"""
            <div class="stock-card">
                <div class="stock-title">{emp}</div>
                <div class="stock-ticker">{ticker_val}</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

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
      titulo_h,
      lbl_tasa,
      is_mx,
  ):
    st.subheader(f"1. Evolución de las Acciones - Precios Normalizados ({titulo_h})")
    precios_norm = (datos_hist / datos_hist.iloc[0]) * 100
    fig_norm = px.line(
        precios_norm,
        title=f"Evolución Comparativa de Precios Escalados ({titulo_h})",
    )
    st.plotly_chart(fig_norm, use_container_width=True)

    st.subheader(f"2. Portafolio de Mínima Varianza ({titulo_h})")
    cols_min = [c for c in lista_empresas + ["Rp", "vp", "des_p"]]
    st.dataframe(
        min_port[[c for c in cols_min if c in min_port.columns]].style.format(
            formatos_mx if is_mx else formatos_usa
        ),
        use_container_width=True,
    )

    st.subheader(f"3. Optimización por Ratio de Sharpe ({titulo_h})")
    st.plotly_chart(
        construir_figura_animada_con_minima(
            anim_data,
            min_port,
            lista_empresas,
            titulo_h,
            lbl_tasa,
            is_mx=is_mx,
        ),
        use_container_width=True,
    )

    st.markdown("#### Portafolio Óptimo Global (Máximo Sharpe Absoluto)")
    cols_sh = [
        c
        for c in lista_empresas
        + ["Rp", "vp", "des_p", "Tasa_diaria", "Tasa_anual", "sharpe_ratio"]
    ]
    st.dataframe(
        max_port[[c for c in cols_sh if c in max_port.columns]].style.format(
            formatos_mx if is_mx else formatos_usa
        ),
        use_container_width=True,
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
        "3 Años (México)",
        "CETES",
        True,
    )

  with sub_mx_comp:
    st.subheader("📋 Tablas Comparativas de los 3 Años (México)")
    st.dataframe(
        pd.concat([min_mx_1, min_mx_2, min_mx_3])[
            [c for c in cols_min_mx if c in min_mx_1.columns]
        ].style.format(formatos_mx),
        use_container_width=True,
    )


# ==============================================================================
# PESTAÑA 2: ESTADOS UNIDOS
# ==============================================================================
with tab_usa:
  st.header("🇺🇸 Análisis de Portafolios - Activos de EE. UU. y T-Bills")

  # MOSTRAR LAS 5 EMPRESAS DE USA EN CAJITAS / BOTONES ESTÉTICOS
  st.markdown("### 🏢 Acciones Activas en el Portafolio (EE. UU.)")
  cols_cards_usa = st.columns(5)
  for idx, emp in enumerate(empresas_usa[:5]):
    with cols_cards_usa[idx]:
      ticker_val = tickers_usa.get(emp, emp)
      st.markdown(
          f"""
            <div class="stock-card">
                <div class="stock-title">{emp}</div>
                <div class="stock-ticker">{ticker_val}</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

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
        "3 Años (EE. UU.)",
        "T-Bills",
        False,
    )

  with sub_usa_comp:
    st.subheader("📋 Tablas Comparativas de los 3 Años (EE. UU.)")
    st.dataframe(
        pd.concat([min_usa_1, min_usa_2, min_usa_3])[
            [c for c in cols_min_usa if c in min_usa_1.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )


# ==============================================================================
# PESTAÑA 3: SIMULACIÓN POR ENFOQUE (DINÁMICA)
# ==============================================================================
with tab_port_enf:
  st.header("🌐 Simulación Estocástica por Enfoque (México, EE. UU. y Global)")

  col_sel1, col_sel2 = st.columns(2)
  with col_sel1:
    enfoque_sel = st.selectbox(
        "Selecciona el enfoque de simulación:",
        [
            "Full México (Solo Acciones MX + CETES)",
            "Full EE. UU. (Solo Acciones USA + T-Bills + TC)",
            "Global Unificado (México + EE. UU. Combinados)",
        ],
        key="sel_enf_dinamico",
    )
  with col_sel2:
    horiz_sel = st.selectbox(
        "Selecciona el horizonte temporal:",
        ["1 año", "2 años", "3 años"],
        key="sel_h_enfoque",
    )

  if horiz_sel == "1 año":
    s_mx, s_usa = max_mx_1, max_usa_1
  elif horiz_sel == "2 años":
    s_mx, s_usa = max_mx_2, max_usa_2
  else:
    s_mx, s_usa = max_mx_3, max_usa_3

  if "Full México" in enfoque_sel:
    enf_key = "mexico"
    titulo_enf = "Full México ($650,000 MXN Iniciales)"
    mostrar_mx = True
    mostrar_usa = False
  elif "Full EE. UU." in enfoque_sel:
    enf_key = "usa"
    titulo_enf = "Full EE. UU. ($350,000 MXN Equiv. en USD)"
    mostrar_mx = False
    mostrar_usa = True
  else:
    enf_key = "global"
    titulo_enf = "Portafolio Global Unificado ($1,000,000 MXN Iniciales)"
    mostrar_mx = True
    mostrar_usa = True

  precios_mx_actuales = {
      k: obtener_precio_actual(v, 100.0) for k, v in tickers_mx.items()
  }
  precios_usa_actuales = {
      k: obtener_precio_actual(v, 150.0) for k, v in tickers_usa.items()
  }

  if mostrar_mx:
    st.markdown("#### 🇲🇽 Portafolio Accionario México ($500,000 MXN)")
    activos_validos_mx = [e for e in empresas_mx if e in s_mx.columns]
    pesos_mx_opt = s_mx[activos_validos_mx].values[0]
    df_t_mx = pd.DataFrame({
        "Empresa": activos_validos_mx,
        "Peso Óptimo": pesos_mx_opt,
        "Capital Asignado (MXN)": pesos_mx_opt * 500000.0,
        "Precio Actual (MXN)": [
            precios_mx_actuales.get(e, 100.0) for e in activos_validos_mx
        ],
    })
    df_t_mx["Títulos a Comprar"] = np.floor(
        df_t_mx["Capital Asignado (MXN)"] / df_t_mx["Precio Actual (MXN)"]
    ).astype(int)
    st.dataframe(
        df_t_mx.style.format({
            "Peso Óptimo": "{:.2%}",
            "Capital Asignado (MXN)": "${:,.2f}",
            "Precio Actual (MXN)": "${:,.2f}",
            "Títulos a Comprar": "{:d}",
        }),
        use_container_width=True,
    )

  if mostrar_usa:
    st.markdown("#### 🇺🇸 Portafolio Accionario EE. UU. ($245,000 MXN en USD)")
    activos_validos_usa = [e for e in empresas_usa if e in s_usa.columns]
    pesos_usa_opt = s_usa[activos_validos_usa].values[0]
    tc_inicial = 18.44
    df_t_usa = pd.DataFrame({
        "Empresa": activos_validos_usa,
        "Peso Óptimo": pesos_usa_opt,
        "Capital Asignado (USD)": pesos_usa_opt * (245000.0 / tc_inicial),
        "Precio Actual (USD)": [
            precios_usa_actuales.get(e, 150.0) for e in activos_validos_usa
        ],
    })
    df_t_usa["Títulos a Comprar"] = np.floor(
        df_t_usa["Capital Asignado (USD)"] / df_t_usa["Precio Actual (USD)"]
    ).astype(int)
    st.dataframe(
        df_t_usa.style.format({
            "Peso Óptimo": "{:.2%}",
            "Capital Asignado (USD)": "${:,.2f}",
            "Precio Actual (USD)": "${:,.2f}",
            "Títulos a Comprar": "{:d}",
        }),
        use_container_width=True,
    )

  st.markdown("---")
  df_est_enf, rend_enf_sim, val_enf_sim = generar_simulacion_y_estadisticos(
      s_mx, s_usa, dias_habiles, alpha_nivel, enfoque=enf_key
  )
  st.subheader(f"📊 Tabla de Estadísticos & Riesgo Estocástico ({titulo_enf})")
  st.dataframe(
      df_est_enf.style.format(
          {col: "{:.2%}" for col in df_est_enf.columns}
      ),
      use_container_width=True,
  )

  # Métricas
  col_k1, col_k2, col_k3, col_k4 = st.columns(4)
  col_k1.metric(
      "Rendimiento Esperado", f"{np.mean(rend_enf_sim)*100:.2f}%"
  )
  col_k2.metric("Volatilidad (Riesgo)", f"{np.std(rend_enf_sim)*100:.2f}%")
  col_k3.metric(
      f"VaR ({alpha_nivel*100:.1f}%)",
      f"{(np.mean(rend_enf_sim) - np.percentile(rend_enf_sim, alpha_nivel*100))*100:.2f}%",
  )
  col_k4.metric("Mediana Final (Q2)", f"${np.median(val_enf_sim):,.2f} MXN")

  st.markdown("---")
  st.subheader("📈 Gráfica de Distribución del Valor Final")
  fig_hist = px.histogram(
      x=val_enf_sim,
      nbins=50,
      labels={"x": "Valor Final en MXN"},
      title=f"Distribución de Probabilidad - {titulo_enf}",
      color_discrete_sequence=["#2980b9"],
  )
  st.plotly_chart(fig_hist, use_container_width=True)

  # Tabla de simulación diaria dinámica
  st.subheader("📈 Tabla de Simulación Diaria Estocástica Dinámica")
  dias_muestra = 15
  fechas_m = [
      (fecha_inicio_sim + datetime.timedelta(days=i)).strftime("%d/%m/%Y")
      for i in range(dias_muestra)
  ]
  tc_base = 18.44
  val_mx = 500000.0
  val_cetes = 150000.0
  val_usa_usd = 245000.0 / tc_base
  val_tb_usd = 105000.0 / tc_base

  filas_excel = []
  for idx, f in enumerate(fechas_m):
    p_cetes = np.random.uniform(0, 1)
    tasa_c = 0.08
    p_tb = np.random.uniform(0, 1)
    tasa_t = 0.037
    rend_mx_n = np.random.normal(0.0005, 0.012)
    rend_usa_n = np.random.normal(0.0006, 0.015)

    if idx > 0:
      val_mx *= 1 + rend_mx_n
      val_cetes *= 1 + tasa_c / 360
      val_usa_usd *= 1 + rend_usa_n
      val_tb_usd *= 1 + tasa_t / 360
      tc = tc_base * (1 + np.random.normal(0, 0.002))
    else:
      tc = tc_base
      rend_mx_n, rend_usa_n = 0.0, 0.0

    total_usa_mxn = (val_usa_usd + val_tb_usd) * tc
    total_global = val_mx + val_cetes + total_usa_mxn

    fila_dict = {"Fecha": f}
    if mostrar_mx:
      fila_dict.update({
          "Rend. MX (Norm)": rend_mx_n,
          "Portafolio Acc. MX ($)": val_mx,
          "Tasa CETES": tasa_c,
          "Valor CETES ($)": val_cetes,
      })
    if mostrar_usa:
      fila_dict.update({
          "Rend. USA (Norm)": rend_usa_n,
          "Portafolio Acc. USA (USD)": val_usa_usd,
          "Tasa T-Bills": tasa_t,
          "Tipo Cambio": tc,
          "Total USA (MXN)": total_usa_mxn,
      })
    if enf_key == "global":
      fila_dict["VALOR TOTAL GLOBAL (MXN)"] = total_global
    filas_excel.append(fila_dict)

  df_excel_demo = pd.DataFrame(filas_excel)
  st.dataframe(df_excel_demo.style.format(na_rep="-"), use_container_width=True)


# ==============================================================================
# PESTAÑA 4: MODELO EXCEL & TÍTULOS
# ==============================================================================
with tab_excel:
  st.header("📑 Resumen de Asignación de Títulos Enteros")
  horizonte_excel = st.selectbox(
      "Horizonte para títulos:", ["1 año", "2 años", "3 años"], key="sel_excel"
  )
  mx_p, usa_p = (
      (max_mx_1, max_usa_1)
      if horizonte_excel == "1 año"
      else (max_mx_2, max_usa_2)
      if horizonte_excel == "2 años"
      else (max_mx_3, max_usa_3)
  )

  st.markdown("### 🇲🇽 México")
  act_mx_val = [e for e in empresas_mx if e in mx_p.columns]
  st.dataframe(
      pd.DataFrame({
          "Empresa": act_mx_val,
          "Peso": mx_p[act_mx_val].values[0],
      }).style.format({"Peso": "{:.2%}"}),
      use_container_width=True,
  )

  st.markdown("### 🇺🇸 EE. UU.")
  act_usa_val = [e for e in empresas_usa if e in usa_p.columns]
  st.dataframe(
      pd.DataFrame({
          "Empresa": act_usa_val,
          "Peso": usa_p[act_usa_val].values[0],
      }).style.format({"Peso": "{:.2%}"}),
      use_container_width=True,
  )


# ==============================================================================
# PESTAÑA 5: ANEXO DE FÓRMULAS & PARÁMETROS QUANT
# ==============================================================================
with tab_anexo:
  st.header("📐 Anexo Cuantitativo: Fórmulas y Parámetros Utilizados")
  st.markdown(
      "Aquí tienes la demostración formal y algebraica exacta de los modelos"
      " matemáticos implementados en todo el simulador, ideal para tu"
      " exposición:"
  )

  col1, col2 = st.columns(2)

  with col1:
    st.markdown("### 1. Modelo de Markowitz (Teoría de Portafolios)")
    st.markdown(
        "**Rendimiento Esperado del Portafolio ($R_p$):** Promedio ponderado"
        " de los rendimientos logarítmicos anualizados de los activos individuales."
    )
    st.latex(r"R_p = \sum_{i=1}^{n} w_i \cdot \mu_i = \mathbf{w}^T \boldsymbol{\mu}")

    st.markdown(
        "**Varianza y Volatilidad del Portafolio ($\sigma_p^2$ y $\sigma_p$):**"
        " Mide el riesgo considerando la covarianza entre los activos."
    )
    st.latex(
        r"\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w} = \sum_{i=1}^n"
        r" \sum_{j=1}^n w_i w_j \sigma_{ij}"
    )
    st.latex(r"\sigma_p = \sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}")

    st.markdown("### 2. Ratio de Sharpe y Línea de Asignación (CAL)")
    st.markdown(
        "Mide el exceso de rendimiento por unidad de riesgo total asumido frente"
        " a la tasa libre de riesgo ($R_f$ como CETES o T-Bills)."
    )
    st.latex(r"Sharpe = \frac{R_p - R_f}{\sigma_p}")
    st.latex(r"E(R_c) = R_f + Sharpe \cdot \sigma_c")

  with col2:
    st.markdown(
        "### 3. Simulación Estocástica de Tasas (Distribución Triangular)"
    )
    st.markdown(
        "Modela tasas de CETES y T-Bills delimitadas por un mínimo ($a$), moda"
        " ($b$) y máximo ($c$). Con $k = \\frac{b-a}{c-a}$:"
    )
    st.latex(r"""
        T_L = \begin{cases} 
        a + \sqrt{U \cdot (b - a)(c - a)} & \text{si } U < k \\ 
        c - \sqrt{(1 - U)(c - a)(c - b)} & \text{si } U \ge k 
        \end{cases}
        """)

    st.markdown("### 4. Tipo de Cambio Estocástico (Distribución Weibull)")
    st.markdown(
        "Modela la volatilidad cambiaria USD/MXN mediante parámetros de escala"
        " ($\lambda = 18.32$) y forma ($k = 28.53$):"
    )
    st.latex(r"TC = \lambda \cdot \left( \frac{U}{1 - U} \right)^{\frac{1}{k}}")

    st.markdown("### 5. Métricas de Riesgo Estocástico (Cutoff & VaR)")
    st.markdown(
        "Value at Risk ($VaR$) al nivel de significancia $\alpha$ (ej. 5%):"
    )
    st.latex(r"Cutoff_{\alpha} = Q_{\alpha}(R_{global})")
    st.latex(r"VaR_{\alpha} = \mu_{global} - Cutoff_{\alpha}")
