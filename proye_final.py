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

# Estilo CSS personalizado para las cajitas de selección
st.markdown("""
    <style>
    .stock-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title(
    "📊 Simulador Integral de Portafolios, Optimización y Riesgo Estocástico"
)
st.markdown(
    "Modelo cuantitativo multianual con Markowitz, Frontera Eficiente, Simulación"
    " Monte Carlo, Bootstrapping (10,000 corridas), Stop-Loss Individual y Métricas de Riesgo."
)

# ==============================================================================
# 1. BARRA LATERAL (SIDEBAR): PARÁMETROS GENERALES Y CAPITAL DINÁMICO
# ==============================================================================
st.sidebar.header("⚙️ Configuración del Análisis")

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

n_simulaciones = st.sidebar.slider(
    "Número de Portafolios (Simulación Markowitz):", 1000, 10000, 5000, 500
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
dias_habiles = max(1, int(dias_totales * (252 / 365)))
st.sidebar.info(
    f"📆 Días calendario: **{dias_totales}** | Días hábiles estimados:"
    f" **{dias_habiles}**"
)

# ==============================================================================
# 2. LISTAS DE 15 EMPRESAS DISPONIBLES PARA CADA PAÍS (PARA LAS CAJITAS)
# ==============================================================================
opciones_mx = [
    "LAMOSA.MX", "WALMEX.MX", "AMXB.MX", "BIMBOA.MX", "FEMSAUBD.MX",
    "ALFAA.MX", "CEMEXCPO.MX", "GENTERA.MX", "GCARSOA.MX", "ELEKTRA.MX",
    "ASURB.MX", "GAPB.MX", "OMAB.MX", "KOFUBL.MX", "PINFRA.MX",
]

opciones_usa = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "NFLX", "AMD", "INTC",
    "QCOM", "IBM", "ORCL", "JPM", "V",
]

# ==============================================================================
# 3. SELECTORES TIPO "CAJITA" EN LA INTERFAZ PRINCIPAL (5 POR PAÍS)
# ==============================================================================
st.markdown("---")
st.subheader("🏢 Selección Interactiva de Activos (Cajitas de Exposición)")
col_sel_mx, col_sel_usa = st.columns(2)

with col_sel_mx:
  st.markdown("#### 🇲🇽 Selecciona 5 Activos para México (BMV)")
  c_mx1, c_mx2, c_mx3, c_mx4, c_mx5 = st.columns(5)
  with c_mx1:
    t1 = st.selectbox("M1", opciones_mx, index=0, key="m1")
  with c_mx2:
    t2 = st.selectbox("M2", opciones_mx, index=1, key="m2")
  with c_mx3:
    t3 = st.selectbox("M3", opciones_mx, index=2, key="m3")
  with c_mx4:
    t4 = st.selectbox("M4", opciones_mx, index=3, key="m4")
  with c_mx5:
    t5 = st.selectbox("M5", opciones_mx, index=4, key="m5")
  tickers_mx_list = [t1, t2, t3, t4, t5]

with col_sel_usa:
  st.markdown("#### 🇺🇸 Selecciona 5 Activos para EE. UU. (Tech/Global)")
  c_usa1, c_usa2, c_usa3, c_usa4, c_usa5 = st.columns(5)
  with c_usa1:
    u1 = st.selectbox("U1", opciones_usa, index=0, key="u1")
  with c_usa2:
    u2 = st.selectbox("U2", opciones_usa, index=1, key="u2")
  with c_usa3:
    u3 = st.selectbox("U3", opciones_usa, index=2, key="u3")
  with c_usa4:
    u4 = st.selectbox("U4", opciones_usa, index=3, key="u4")
  with c_usa5:
    u5 = st.selectbox("U5", opciones_usa, index=4, key="u5")
  tickers_usa_list = [u1, u2, u3, u4, u5]

tickers_mx = {t.split(".")[0]: t for t in tickers_mx_list}
empresas_mx = list(tickers_mx.keys())

tickers_usa = {t: t for t in tickers_usa_list}
empresas_usa = list(tickers_usa.keys())
st.markdown("---")

fechas_horizontes = {
    "1 año": {
        "inicio": (fecha_inicio_sim - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
        "fin": fecha_inicio_sim.strftime("%Y-%m-%d"),
    },
    "2 años": {
        "inicio": (fecha_inicio_sim - datetime.timedelta(days=730)).strftime("%Y-%m-%d"),
        "fin": fecha_inicio_sim.strftime("%Y-%m-%d"),
    },
    "3 años": {
        "inicio": (fecha_inicio_sim - datetime.timedelta(days=1095)).strftime("%Y-%m-%d"),
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
# 5. FUNCIÓN DE SIMULACIÓN Y OPTIMIZACIÓN (MARKOWITZ) - CON MÍNIMA VARIANZA GLOBAL REAL
# ==============================================================================
@st.cache_data
def ejecutar_simulacion_mercado(
    tickers_dict, lista_empresas, start, end, n_sim, n_corr, tipo_tasa="cetes"
):
  raw_data = yf.download(
      list(tickers_dict.values()), start=start, end=end, progress=False
  )
  if isinstance(raw_data, pd.DataFrame) and "Close" in raw_data.columns:
    datos = raw_data["Close"]
  else:
    datos = raw_data

  if isinstance(datos, pd.DataFrame):
    if isinstance(datos.columns, pd.MultiIndex):
      datos.columns = datos.columns.droplevel(0)
    datos.rename(columns={v: k for k, v in tickers_dict.items()}, inplace=True)
    datos = datos[[c for c in lista_empresas if c in datos.columns]].dropna(
        how="all"
    )
    datos = datos.dropna(axis=1, how="any")
  else:
    datos = pd.DataFrame(datos)
    datos.columns = [lista_empresas[0]]
    datos = datos.dropna()

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
  colores_anim = px.colors.sample_colorscale(
      "turbo", np.linspace(0.1, 0.9, n_animacion_pasos)
  )

  w_cetes = 0.30
  w_portafolio = 0.70

  min_port = None
  min_desp_val = float('inf')

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

    # RASTREAR LA MÍNIMA VARIANZA GLOBAL REAL EN TODAS LAS CORRIDAS
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
      ret_comb = (w_cetes * tasa_anual) + (w_portafolio * ret_opt_anual)

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

min_mx_1.index, min_mx_2.index, min_mx_3.index = ["1 año"], ["2 años"], ["3 años"]
max_mx_1.index, max_mx_2.index, max_mx_3.index = ["1 año"], ["2 años"], ["3 años"]
min_usa_1.index, min_usa_2.index, min_usa_3.index = ["1 año"], ["2 años"], ["3 años"]
max_usa_1.index, max_usa_2.index, max_usa_3.index = ["1 año"], ["2 años"], ["3 años"]


# ==============================================================================
# FUNCIÓN DINÁMICA DE ASIGNACIÓN DE CAPITAL (MÍNIMA VARIANZA ENTRE MX Y USA)
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
):
  fig = go.Figure()
  n_pasos = len(datos_anim)

  fig.add_trace(
      go.Scatter(
          x=[],
          y=[],
          mode="markers",
          marker=dict(
              size=4,
              color=[],
              colorscale="Viridis",
              showscale=False,
              opacity=0.6,
          ),
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
              color="#e74c3c",
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
              color="#27ae60",
              symbol="diamond",
              line=dict(width=1, color="black"),
          ),
          name=f"Mix 30% {lbl_tasa} / 70% Portafolio",
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
            mode="markers",
            marker=dict(
              size=4, color=curr["nube_y"], colorscale="Viridis", showscale=False
          ),
        )
    )
    for j in range(n_pasos):
      if j <= f_idx:
        f_data.append(
            go.Scatter(x=datos_anim[j]["x_cal"], y=datos_anim[j]["y_cal"])
        )
      else:
        f_data.append(go.Scatter(x=[], y=[]))

    stars_x, stars_y = [], []
    for j in range(f_idx + 1):
      if j != best_global_idx:
        stars_x.append(datos_anim[j]["star_x"])
        stars_y.append(datos_anim[j]["star_y"])
    f_data.append(go.Scatter(x=stars_x, y=stars_y))

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
        )
    )
    f_data.append(go.Scatter(x=[0], y=[curr["tasa_anual"]]))
    f_data.append(go.Scatter(x=[min_x], y=[min_y]))
    frames.append(go.Frame(data=f_data, name=f"Paso {f_idx + 1}"))

  for idx, t_data in enumerate(frames[0].data):
    fig.data[idx].x = t_data.x
    fig.data[idx].y = t_data.y

  fig.update_layout(
      title=f"Simulación Monte Carlo Paso a Paso ({titulo_horizonte})",
      template="plotly_white",
      height=500,
      xaxis=dict(title="Volatility (Riesgo Anual)", range=[0, 1.2]),
      yaxis=dict(title="Return (Rendimiento Anual)", range=[-0.1, 0.4]),
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
# 7. SIMULADOR ESTOCÁSTICO CON BOOTSTRAPPING (10,000) Y STOP-LOSS INDIVIDUAL (-10%)
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

  # --- TASAS DIARIAS CETES (Triangular) ---
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

  # --- TASAS DIARIAS T-BILLS (Triangular) ---
  a_t, b_t, c_t = 0.0365, 0.0370, 0.0375
  k_t = (b_t - a_t) / (c_t - a_t)
  u_t = np.random.uniform(0, 1, size=(num_simulaciones_futuras, n_dias))
  tasas_tbills = np.where(
      u_t < k_t,
      a_t + np.sqrt(u_t * (b_t - a_t) * (c_t - a_t)),
      c_t - np.sqrt((1 - u_t) * (c_t - a_t) * (c_t - b_t)),
  )
  val_final_tbills_usd = (monto_tbills_mxn / tc_inicial) * np.prod(
      1 + tasas_tbills / 360, axis=1
  )

  # --- BOOTSTRAPPING + STOP-LOSS INDIVIDUAL (-10%) PARA MÉXICO ---
  val_final_mx_acc = np.zeros(num_simulaciones_futuras)
  val_extra_cetes_mx = np.zeros(num_simulaciones_futuras)
  conteo_salidas_mx = {}

  if monto_mx_acc > 0:
    rend_hist_mx = np.log(datos_mx / datos_mx.shift(1)).dropna()
    activos_mx = [
        e for e in empresas_mx if e in max_mx.columns and e in rend_hist_mx.columns
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

  # --- BOOTSTRAPPING + STOP-LOSS INDIVIDUAL (-10%) PARA EE. UU. (REFUGIO EN T-BILLS) ---
  monto_usa_acc_usd = monto_usa_acc_mxn / tc_inicial
  val_final_usa_acc_usd = np.zeros(num_simulaciones_futuras)
  val_extra_tbills_usa_mxn = np.zeros(num_simulaciones_futuras)
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
      # Refugio en T-Bills expresado en MXN (con su respectiva tasa y tipo de cambio)
      cap_refugiado_mxn = ((cap_emp_usd * tc_inicial) * (1 + stop_loss)) * (
          (1 + tasa_prom_tbills / 360) ** dias_restantes
      )

      val_final_usa_acc_usd += np.where(salio, 0.0, val_no_salio)
      val_extra_tbills_usa_mxn += np.where(salio, cap_refugiado_mxn, 0.0)

  # --- Tipo de Cambio (Weibull) ---
  u_tc = np.random.uniform(0.001, 0.999, size=num_simulaciones_futuras)
  tc_final = scale_lambda * (u_tc / (1 - u_tc)) ** (1 / shape_k)

  val_final_usa_total_mxn = (
      val_final_usa_acc_usd * tc_final
  ) + val_final_tbills_usd * tc_final + val_extra_tbills_usa_mxn
  
  val_final_cetes_total = (
      val_final_cetes_base + val_extra_cetes_mx
  )
  
  val_final_global_mxn = (
      val_final_mx_acc + val_final_cetes_total + val_final_usa_total_mxn
  )

  rend_global = (val_final_global_mxn - capital_total) / capital_total
  rend_mx_acc = (
      ((val_final_mx_acc + val_extra_cetes_mx) - monto_mx_acc) / monto_mx_acc
      if monto_mx_acc > 0
      else np.zeros_like(rend_global)
  )
  
  # CORRECCIÓN DE RENDIMIENTO EN DÓLARES (INCLUYENDO T-BILLS REFUGIO)
  val_usa_total_neto = val_final_usa_acc_usd + (val_extra_tbills_usa_mxn / tc_inicial)
  rend_usa_usd = (
      (val_usa_total_neto - monto_usa_acc_usd) / monto_usa_acc_usd
      if monto_usa_acc_usd > 0
      else np.zeros_like(rend_global)
  )
  
  rend_cetes = (
      (val_final_cetes_base - monto_cetes) / monto_cetes
      if monto_cetes > 0
      else np.zeros_like(rend_global)
  )
  rend_cambiario = (tc_final - tc_inicial) / tc_inicial

  # --- VALORES FINALES EN PESOS (MXN) PARA CADA COMPONENTE ---
  val_mx_total_mxn = val_final_mx_acc + val_extra_cetes_mx
  val_usa_total_mxn_val = (
      val_final_usa_acc_usd * tc_final
  ) + val_extra_tbills_usa_mxn
  val_cetes_total_mxn = val_final_cetes_base
  val_cambiario_mxn = (monto_usa_acc_usd + (monto_tbills_mxn / tc_inicial)) * tc_final

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

  filas_stats = [
      calcular_fila_stats(rend_global, val_final_global_mxn),
      calcular_fila_stats(rend_mx_acc, val_mx_total_mxn),
      calcular_fila_stats(rend_usa_usd, val_usa_total_mxn_val),
      calcular_fila_stats(rend_cetes, val_cetes_total_mxn),
      calcular_fila_stats(rend_cambiario, val_cambiario_mxn),
  ]

  df_estadisticos = pd.DataFrame(
      filas_stats,
      index=[
          "Portafolio Global Final",
          "Portafolio Renta Variable México",
          "Portafolio Renta Variable EE. UU.",
          "Portafolio Renta Fija (CETES)",
          "Efecto Rendimiento Cambiario",
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

  df_excel_10k = pd.DataFrame({
      "Modelado": range(1, num_simulaciones_futuras + 1),
      "Rendimiento en Pesos": rend_mx_acc,
      "Rendimiento en Dólares": rend_usa_usd,
      "Inicio del Dólar": tc_inicial,
      "Final del Dólar": tc_final,
      "Ganancia Cambiaria": rend_cambiario,
      "Rendimiento Global": rend_global,
  })

  return (
      df_estadisticos,
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
        datos_hist.tail(10).style.format("${:,.2f}"), use_container_width=True
    )

    st.subheader(f"3. Estadísticos del Rendimiento Histórico ({titulo_h})")
    st.dataframe(
        datos_hist.pct_change().dropna().describe().style.format("{:.4f}"),
        use_container_width=True,
    )

    st.subheader(f"4. Matriz de Covarianza Anualizada ({titulo_h})")
    matriz_limpia = np.array(cov_mat, dtype=float)
    if (
        matriz_limpia.ndim == 2
        and matriz_limpia.shape[0] > 1
        and matriz_limpia.shape[1] > 1
    ):
      st.plotly_chart(
          px.imshow(
              matriz_limpia,
              x=cols_activas,
              y=cols_activas,
              color_continuous_scale="Viridis",
              text_auto=".4f",
              title="Matriz de Covarianza (Sigma)",
          ),
          use_container_width=True,
      )
    else:
      st.warning(
          "⚠️ No hay suficientes activos válidos para calcular la matriz de"
          " covarianza."
      )

    st.subheader(f"5. Portafolio de Mínima Varianza ({titulo_h})")
    cols_min = [c for c in cols_activas + ["Rp", "vp", "des_p"]]
    st.dataframe(
        min_port[[c for c in cols_min if c in min_port.columns]].style.format(
            formatos_mx if is_mx else formatos_usa
        ),
        use_container_width=True,
    )

    st.subheader(f"6. Optimización por Ratio de Sharpe ({titulo_h})")
    st.plotly_chart(
        construir_figura_animada_con_minima(
            anim_data, min_port, cols_activas, titulo_h, lbl_tasa, is_mx=is_mx
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
        max_port[[c for c in cols_sh if c in max_port.columns]].style.format(
            formatos_mx if is_mx else formatos_usa
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
        df_stats_sharpe.style.format({"Valor": "{:.4f}"}),
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
        pd.concat([min_mx_1, min_mx_2, min_mx_3])[
            [c for c in cols_min_mx if c in min_mx_1.columns]
        ].style.format(formatos_mx),
        use_container_width=True,
    )
    st.markdown(
        "### B. Tabla Comparativa - Criterio de Máximo Sharpe Ratio (1, 2 y 3"
        " Años)"
    )
    st.dataframe(
        pd.concat([max_mx_1, max_mx_2, max_mx_3])[
            [c for c in cols_sh_mx if c in max_mx_1.columns]
        ].style.format(formatos_mx),
        use_container_width=True,
    )


# ==============================================================================
# PESTAÑA 2: ESTADOS UNIDOS
# ==============================================================================
with tab_usa:
  st.header("🇺🇸 Análisis de Portafolios - Activos de EE. UU. y T-Bills")
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
        pd.concat([min_usa_1, min_usa_2, min_usa_3])[
            [c for c in cols_min_usa if c in min_usa_1.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown(
        "### B. Tabla Comparativa - Criterio de Máximo Sharpe Ratio (1, 2 y 3"
        " Años)"
    )
    st.dataframe(
        pd.concat([max_usa_1, max_usa_2, max_usa_3])[
            [c for c in cols_sh_usa if c in max_usa_1.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )


# ==============================================================================
# PESTAÑA 3: SIMULACIÓN POR ENFOQUE (DINÁMICA CON BOOTSTRAPPING 10,000 + STOP-LOSS)
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
            "Full México (Solo Acciones MX + CETES)",
            "Full EE. UU. (Solo Acciones USA + T-Bills + TC)",
            "Global Unificado (México + EE. UU. Combinados)",
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

  # CALCULAR CAPITALES DINÁMICOS (NO FIJOS)
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

  # TABLA GRANDE DE DISTRIBUCIÓN DINÁMICA DEL CAPITAL TOTAL
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
      df_dist_dinamica.style.format({
          "Peso Dinámico (Mín. Varianza)": "{:.2%}",
          "Inversión Asignada (MXN)": "${:,.2f}",
      }),
      use_container_width=True,
  )

  st.markdown(f"### 🎯 Asignación de Pesos y Títulos ({titulo_enf})")
  precios_mx_actuales = {
      k: obtener_precio_actual(v, 100.0) for k, v in tickers_mx.items()
  }
  precios_usa_actuales = {
      k: obtener_precio_actual(v, 150.0) for k, v in tickers_usa.items()
  }

  if mostrar_mx:
    st.markdown(f"#### 🇲🇽 Portafolio Accionario México (${m_mx_acc:,.2f} MXN)")
    activos_validos_mx = [e for e in empresas_mx if e in s_mx.columns]
    pesos_mx_opt = s_mx[activos_validos_mx].values[0]
    df_t_mx = pd.DataFrame({
        "Empresa": activos_validos_mx,
        "Peso Óptimo": pesos_mx_opt,
        "Capital Asignado (MXN)": pesos_mx_opt * m_mx_acc,
        "Precio Actual (MXN)": [
            precios_mx_actuales[e] for e in activos_validos_mx
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
        "Precio Actual (USD)": [
            precios_usa_actuales[e] for e in activos_validos_usa
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

  # 1. EJECUCIÓN DE LAS 10,000 SIMULACIONES BOOTSTRAP + STOP-LOSS INDIVIDUAL (-10%)
  (
      df_est_enf,
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

  st.subheader(
      "📊 Tabla de Estadísticos & Riesgo (10,000 Simulaciones Bootstrapping +"
      " Stop-Loss Individual 10%)"
  )

  def formateador_tabla_riesgo(val, col_name):
    if isinstance(val, (int, float)):
      if "CUTOFF" in col_name or "VAR" in col_name:
        return f"${val:,.2f}"
      else:
        return f"{val:.2%}"
    return val

  st.dataframe(
      df_est_enf.style.format({
          col: lambda v, c=col: formateador_tabla_riesgo(v, c)
          for col in df_est_enf.columns
      }),
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

  # TABLA DE ACCIONES QUE SALIERON MÁS (STOP-LOSS -10% INDIVIDUAL)
  st.subheader(
      "🛡️ Ranking de Salidas por Stop-Loss (-10% Individual → Refugio en CETES / T-Bills)"
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
        df_sl_ranking.style.format({
            "Veces que Salió (-10%)": "{:,d}",
            "% de Simulaciones": "{:.2%}",
        }),
        use_container_width=True,
    )

  st.markdown("---")

  # TABLA ESTILO EXCEL (MUESTRA DE LAS 10,000 SIMULACIONES)
  st.subheader(
      "📋 Tabla Resumen de 10,000 Simulaciones Bootstrap -"
      " Primeras 15 Filas"
  )
  st.dataframe(
      df_excel_10k.head(15).style.format({
          "Rendimiento en Pesos": "{:.2%}",
          "Rendimiento en Dólares": "{:.2%}",
          "Inicio del Dólar": "${:,.2f}",
          "Final del Dólar": "${:,.2f}",
          "Ganancia Cambiaria": "{:.2%}",
          "Rendimiento Global": "{:.2%}",
      }),
      use_container_width=True,
  )

  st.markdown("---")
  st.subheader(
      f"📈 Gráficas de Distribución y Comportamiento Estocástico ({titulo_enf})"
  )
  fig_hist = px.histogram(
      x=val_enf_sim,
      nbins=50,
      labels={"x": "Valor Final en MXN"},
      title=(
          "Distribución de Probabilidad del Valor Final (10,000 Corridas"
          " Bootstrap)"
      ),
      color_discrete_sequence=["#2980b9"],
  )
  fig_hist.update_layout(
      xaxis_title="Valor Final (MXN)", template="plotly_white"
  )
  st.plotly_chart(fig_hist, use_container_width=True)

  # VIOLIN PLOT Y GRÁFICA DE RENDIMIENTOS (LADO A LADO)
  col_g1, col_g2 = st.columns(2)
  with col_g1:
    fig_violin = px.violin(
        y=val_enf_sim / 1000,
        box=True,
        points="all",
        title="Violin Plot - Dispersión del Valor Final (Miles MXN)",
        labels={"y": "Valor Final (Miles MXN)"},
        color_discrete_sequence=["#27ae60"],
    )
    fig_violin.update_layout(template="plotly_white")
    st.plotly_chart(fig_violin, use_container_width=True)

  with col_g2:
    media_r = np.mean(rend_enf_sim) * 100
    cutoff_r = np.percentile(rend_enf_sim, alpha_nivel * 100) * 100

    fig_10k_mov = go.Figure()
    fig_10k_mov.add_trace(
        go.Scatter(
            x=np.arange(1, len(rend_enf_sim) + 1),
            y=rend_enf_sim * 100,
            mode="markers",
            marker=dict(
                size=4,
                color=rend_enf_sim * 100,
                colorscale="Viridis",
                opacity=0.6,
            ),
            name="Rendimiento (%)",
        )
    )
    fig_10k_mov.add_hline(
        y=media_r,
        line_dash="solid",
        line_color="#27ae60",
        annotation_text=f"Media: {media_r:.2f}%",
        annotation_position="top right",
    )
    fig_10k_mov.add_hline(
        y=cutoff_r,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Cutoff ({alpha_nivel*100:.1f}%): {cutoff_r:.2f}%",
        annotation_position="bottom right",
    )
    fig_10k_mov.update_layout(
        title="Rendimientos en las 10,000 Simulaciones",
        xaxis_title="Simulación (#1 a #10,000)",
        yaxis_title="Rendimiento Final (%)",
        template="plotly_white",
        showlegend=False,
    )
    st.plotly_chart(fig_10k_mov, use_container_width=True)

  # ============================================================================
  # GRÁFICA DE RATIO DE SHARPE COMBINADO (EN PUNTOS Y LEYENDA SUPERIOR)
  # ============================================================================
  st.markdown("---")
  st.subheader(
      "📈 Gráfica de Optimización Ratio de Sharpe (Portafolio Combinado MX +"
      " USA)"
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
  tasa_rf_global = 0.08 * w_mx_din + 0.037 * w_usa_din
  sharpe_comb_grid = (ret_comb_grid - tasa_rf_global) / riesgo_comb_grid

  idx_best_sh = np.argmax(sharpe_comb_grid)
  idx_min_var_g = np.argmin(riesgo_comb_grid)

  fig_sh_global = go.Figure()
  fig_sh_global.add_trace(
      go.Scatter(
          x=riesgo_comb_grid,
          y=ret_comb_grid,
          mode="markers",
          marker=dict(
              size=8,
              color=sharpe_comb_grid,
              colorscale="Viridis",
              showscale=True,
              colorbar=dict(title="Sharpe", x=1.02),
              line=dict(width=0.5, color="white"),
          ),
          name="Portafolios Combinados",
      )
  )
  fig_sh_global.add_trace(
      go.Scatter(
          x=[riesgo_comb_grid[idx_best_sh]],
          y=[ret_comb_grid[idx_best_sh]],
          mode="markers",
          marker=dict(
              size=16,
              color="#e74c3c",
              symbol="star",
              line=dict(width=1.5, color="black"),
          ),
          name="★ Máximo Sharpe Global",
      )
  )
  fig_sh_global.add_trace(
      go.Scatter(
          x=[riesgo_comb_grid[idx_min_var_g]],
          y=[ret_comb_grid[idx_min_var_g]],
          mode="markers",
          marker=dict(
              size=12,
              color="blue",
              symbol="circle",
              line=dict(width=1.5, color="black"),
          ),
          name="🔵 Mínima Varianza Global",
      )
  )

  x_cal_g = np.linspace(0, riesgo_comb_grid[idx_best_sh] * 1.25, 30)
  y_cal_g = tasa_rf_global + sharpe_comb_grid[idx_best_sh] * x_cal_g
  fig_sh_global.add_trace(
      go.Scatter(
          x=x_cal_g,
          y=y_cal_g,
          mode="lines",
          line=dict(color="red", width=1.5, dash="dash"),
          name="Línea de Asignación de Capital (CAL)",
      )
  )
  fig_sh_global.update_layout(
      title="Frontera Eficiente & Ratio de Sharpe (Combinación MX + EE. UU.)",
      xaxis_title="Volatilidad (Riesgo Anual)",
      yaxis_title="Rendimiento Anual Esperado",
      template="plotly_white",
      legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.02,
          xanchor="center",
          x=0.5,
      ),
      margin=dict(t=90, r=80),
  )
  st.plotly_chart(fig_sh_global, use_container_width=True)

  # TABLA DE SIMULACIÓN DIARIA ESTOCÁSTICA (CON CAPITALES DINÁMICOS)
  st.markdown("---")
  st.subheader(f"📈 Tabla de Simulación Diaria Estocástica ({titulo_enf})")
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
      "Prob CETES": "{:.4f}",
      "Tasa CETES (Triang)": "{:.2%}",
      "Valor CETES ($)": "${:,.2f}",
      "Rend. USA (Boot)": "{:.2%}",
      "Portafolio Acc. USA (USD)": "${:,.2f}",
      "Prob T-Bills": "{:.4f}",
      "Tasa T-Bills (Triang)": "{:.2%}",
      "Valor T-Bills (USD)": "${:,.2f}",
      "Tipo Cambio USD/MXN": "${:,.4f}",
      "Valor Total USA (MXN)": "${:,.2f}",
      "VALOR TOTAL PORTAFOLIO GLOBAL (MXN)": "${:,.2f}",
      "VALOR TOTAL MÉXICO (MXN)": "${:,.2f}",
      "VALOR TOTAL USA (MXN)": "${:,.2f}",
  }
  st.dataframe(
      df_excel_demo.style.format(
          {c: v for c, v in formatos_tabla.items() if c in df_excel_demo.columns}
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

  st.subheader(
      f"🇲🇽 Asignación de Títulos - Portafolio México (${m_mx_ex:,.2f} MXN)"
  )
  activos_validos_mx = [e for e in empresas_mx if e in mx_p.columns]
  pesos_mx_opt = mx_p[activos_validos_mx].values[0]
  df_titulos_mx = pd.DataFrame({
      "Empresa": activos_validos_mx,
      "Peso Óptimo": pesos_mx_opt,
      "Capital Asignado (MXN)": pesos_mx_opt * m_mx_ex,
      "Precio Actual (MXN)": [
          precios_mx_actuales[e] for e in activos_validos_mx
      ],
  })
  df_titulos_mx["Títulos a Comprar"] = np.floor(
      df_titulos_mx["Capital Asignado (MXN)"]
      / df_titulos_mx["Precio Actual (MXN)"]
  ).astype(int)
  st.dataframe(
      df_titulos_mx.style.format({
          "Peso Óptimo": "{:.2%}",
          "Capital Asignado (MXN)": "${:,.2f}",
          "Precio Actual (MXN)": "${:,.2f}",
          "Títulos a Comprar": "{:d}",
      }),
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
      "Precio Actual (USD)": [
          precios_usa_actuales[e] for e in activos_validos_usa
      ],
  })
  df_titulos_usa["Títulos a Comprar"] = np.floor(
      df_titulos_usa["Capital Asignado (USD)"]
      / df_titulos_usa["Precio Actual (USD)"]
  ).astype(int)
  st.dataframe(
      df_titulos_usa.style.format({
          "Peso Óptimo": "{:.2%}",
          "Capital Asignado (USD)": "${:,.2f}",
          "Precio Actual (USD)": "${:,.2f}",
          "Títulos a Comprar": "{:d}",
      }),
      use_container_width=True,
  )


# ==============================================================================
# PESTAÑA 5: ANEXO DE FÓRMULAS QUANTITATIVE
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
