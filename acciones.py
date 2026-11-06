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

st.title(
    "📊 Simulador Integral de Portafolios, Optimización y Riesgo Estocástico"
)
st.markdown(
    "Modelo cuantitativo multianual con Markowitz, Frontera Eficiente, Simulación"
    " Monte Carlo y Métricas de Riesgo."
)

# ==============================================================================
# 1. BARRA LATERAL (SIDEBAR): PARÁMETROS GENERALES
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

# Activos actualizados
empresas_mx = [
    "Grupo Lamosa",
    "Walmart México",
    "América Movil",
    "Bimbo",
    "Grupo México",
]
tickers_mx = {
    "Grupo Lamosa": "LAMOSA.MX",
    "Walmart México": "WALMEX.MX",
    "América Movil": "AMXB.MX",
    "Bimbo": "BIMBOA.MX",
    "Grupo México": "GMEXICOB.MX",
}

empresas_usa = ["Microsoft", "Netflix", "Alphabet", "Amazon", "NVIDIA"]
tickers_usa = {
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "NVIDIA": "NVDA",
}

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
# 2. FUNCIÓN DE SIMULACIÓN Y OPTIMIZACIÓN (MARKOWITZ)
# ==============================================================================
@st.cache_data
def ejecutar_simulacion_mercado(
    tickers_dict, lista_empresas, start, end, n_sim, n_corr, tipo_tasa="cetes"
):
  datos = yf.download(list(tickers_dict.values()), start=start, end=end)[
      "Close"
  ]
  if isinstance(datos.columns, pd.MultiIndex):
    datos.columns = datos.columns.droplevel(0)
  datos.rename(columns={v: k for k, v in tickers_dict.items()}, inplace=True)
  datos = datos[lista_empresas].dropna()

  rendimientos = np.log(datos / datos.shift(1)).dropna()
  promedio = rendimientos.mean()
  dias_ano = 252
  Sigma = rendimientos[lista_empresas].cov(ddof=0).values * dias_ano

  if tipo_tasa == "cetes":
    a, b, c = 0.04, 0.08, 0.10
  else:
    a, b, c = 0.0360, 0.0370, 0.0400

  k = (b - a) / (c - a)
  mejores_sharpe = []
  datos_animacion = []
  n_animacion_pasos = min(25, n_corr)
  colores_anim = px.colors.sample_colorscale(
      "turbo", np.linspace(0.1, 0.9, n_animacion_pasos)
  )

  w_cetes = 0.30
  w_portafolio = 0.70
  df_sim_global = None

  for i in range(1, n_corr + 1):
    p_tasa = np.random.uniform(0, 1)
    if p_tasa < k:
      TL = a + np.sqrt(p_tasa * (b - a) * (c - a))
    else:
      TL = c - np.sqrt((1 - p_tasa) * (c - a) * (c - b))

    diario = (1 + TL) ** (1 / 360) - 1
    tasa_anual = TL

    aleatorios_sim = np.random.uniform(0, 1, size=(n_sim, len(lista_empresas)))
    wi_sim = aleatorios_sim / aleatorios_sim.sum(axis=1, keepdims=True)
    df_sim = pd.DataFrame(wi_sim, columns=lista_empresas)

    df_sim["Rp"] = df_sim[lista_empresas].dot(promedio) * dias_ano
    W_sim = df_sim[lista_empresas].values
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
  idx_min_desp = df_sim_global["des_p"].idxmin()
  min_port = df_sim_global.loc[[idx_min_desp]].copy()

  idx_max_global = tabla_sharpe_max["sharpe_ratio"].idxmax()
  max_sharpe_port = tabla_sharpe_max.loc[[idx_max_global]].copy()

  return min_port, max_sharpe_port, datos_animacion, tabla_sharpe_max, datos, Sigma


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
# 3. CONSTRUCTOR DE GRÁFICA ANIMADA (EJE Y DE -0.5 A 0.5)
# ==============================================================================
def construir_figura_animada(
    datos_anim, titulo_horizonte, lbl_tasa="CETES", is_mx=False
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
    frames.append(go.Frame(data=f_data, name=f"Paso {f_idx + 1}"))

  for idx, t_data in enumerate(frames[0].data):
    fig.data[idx].x = t_data.x
    fig.data[idx].y = t_data.y

  y_range = [-0.5, 0.5]

  fig.update_layout(
      title=f"Simulación Monte Carlo Paso a Paso ({titulo_horizonte})",
      template="plotly_white",
      xaxis=dict(title="Volatility (Riesgo Anual)", range=[0, 1.2]),
      yaxis=dict(title="Return (Rendimiento Anual)", range=y_range),
      updatemenus=[{
          "type": "buttons",
          "buttons": [
              {
                  "label": "▶ Reproducir",
                  "method": "animate",
                  "args": [
                      None,
                      {
                          "frame": {"duration": 300, "redraw": False},
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
                          "frame": {"duration": 200, "redraw": False},
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


def resaltar_maximo_sharpe(df, col_sharpe="sharpe_ratio"):
  max_idx = df[col_sharpe].idxmax()

  def estilo_fila(row):
    if row.name == max_idx:
      return [
          "background-color: #d4edda; font-weight: bold; color: #155724"
          for _ in row
      ]
    return ["" for _ in row]

  return df.style.apply(estilo_fila, axis=1)


# ==============================================================================
# 4. SIMULADOR ESTOCÁSTICO DIARIO Y ESTADÍSTICOS (CORREGIDO)
# ==============================================================================
def generar_simulacion_y_estadisticos(
    max_mx, max_usa, n_dias, alpha, num_simulaciones_futuras=1000
):
  tc_inicial = 18.44
  shape_k = 28.52977
  scale_lambda = 18.32123

  monto_mx_acc = 500000.0
  monto_cetes = 150000.0
  monto_usa_acc_mxn = 245000.0
  monto_tbills_mxn = 105000.0

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

  rend_global = (val_final_global_mxn - 1000000.0) / 1000000.0
  rend_mx_acc = (val_final_mx_acc - monto_mx_acc) / monto_mx_acc
  rend_usa_usd = (val_final_usa_acc_usd - monto_usa_acc_usd) / monto_usa_acc_usd
  rend_cetes = (val_final_cetes - monto_cetes) / monto_cetes
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

  return (
      df_estadisticos,
      rend_global,
      rend_mx_acc,
      rend_usa_usd,
  )  # Variable corregida


# ==============================================================================
# 5. INTERFAZ EN PESTAÑAS PRINCIPALES
# ==============================================================================
tab_mx, tab_usa, tab_excel, tab_stats, tab_charts, tab_anexo = st.tabs([
    "🇲🇽 Mercado Mexicano (BMV)",
    "🇺🇸 Mercado EE. UU. (Tech)",
    "📑 Modelo Excel Estocástico & Títulos",
    "📊 Estadísticos & Riesgo",
    "📈 Gráficas & Covarianza",
    "📐 Anexo de Fórmulas Quantitative",
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
    "Tasa_anual": "{:.2%}",
    "sharpe_ratio": "{:.4f}",
})
cols_min_usa = [c for c in empresas_usa + ["Rp", "vp", "des_p"]]
cols_sh_usa = [
    c
    for c in empresas_usa
    + ["Rp", "vp", "des_p", "Tasa_diaria", "Tasa_anual", "sharpe_ratio"]
]

# ------------------------------------------------------------------------------
# PESTAÑA 1: MÉXICO
# ------------------------------------------------------------------------------
with tab_mx:
  st.header("🇲🇽 Análisis de Portafolios - Activos Mexicanos y CETES")
  sub_mx_1, sub_mx_2, sub_mx_3, sub_mx_comp = st.tabs(
      ["📅 1 Año", "📅 2 Años", "📅 3 Años", "📋 Tablas Comparativas (MX)"]
  )

  with sub_mx_1:
    col_g, col_ctrl = st.columns([3, 1])
    with col_ctrl:
      st.subheader("🎛️ Controles")
      st.info("Utiliza los controles superiores para ajustar las iteraciones.")
    with col_g:
      st.plotly_chart(
          construir_figura_animada(
              anim_mx_1, "1 Año (México)", "CETES", is_mx=True
          ),
          use_container_width=True,
      )

    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_mx_1[[c for c in cols_min_mx if c in min_mx_1.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Portafolio Óptimo Global (Máximo Sharpe Absoluto)")
    st.dataframe(
        max_mx_1[[c for c in cols_sh_mx if c in max_mx_1.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios de las Corridas (Top 10)")
    st.dataframe(
        resaltar_maximo_sharpe(
            t1000_mx_1.head(n_corridas)[
                [c for c in cols_sh_mx if c in t1000_mx_1.columns]
            ]
        ).format(formatos_mx),
        use_container_width=True,
    )

  with sub_mx_2:
    st.plotly_chart(
        construir_figura_animada(anim_mx_2, "2 Años (México)", "CETES", is_mx=True),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_mx_2[[c for c in cols_min_mx if c in min_mx_2.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Portafolio Óptimo Global (Máximo Sharpe Absoluto)")
    st.dataframe(
        max_mx_2[[c for c in cols_sh_mx if c in max_mx_2.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios de las Corridas (Top 10)")
    st.dataframe(
        resaltar_maximo_sharpe(
            t1000_mx_2.head(n_corridas)[
                [c for c in cols_sh_mx if c in t1000_mx_2.columns]
            ]
        ).format(formatos_mx),
        use_container_width=True,
    )

  with sub_mx_3:
    st.plotly_chart(
        construir_figura_animada(anim_mx_3, "3 Años (México)", "CETES", is_mx=True),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_mx_3[[c for c in cols_min_mx if c in min_mx_3.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Portafolio Óptimo Global (Máximo Sharpe Absoluto)")
    st.dataframe(
        max_mx_3[[c for c in cols_sh_mx if c in max_mx_3.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios de las Corridas (Top 10)")
    st.dataframe(
        resaltar_maximo_sharpe(
            t1000_mx_3.head(n_corridas)[
                [c for c in cols_sh_mx if c in t1000_mx_3.columns]
            ]
        ).format(formatos_mx),
        use_container_width=True,
    )

  with sub_mx_comp:
    st.subheader("📋 Tablas Comparativas (México)")
    st.markdown("### Mínima Desviación")
    st.dataframe(
        pd.concat([min_mx_1, min_mx_2, min_mx_3])[
            [c for c in cols_min_mx if c in min_mx_1.columns]
        ].style.format(formatos_mx),
        use_container_width=True,
    )
    st.markdown("### Máximo Sharpe Ratios")
    st.dataframe(
        pd.concat([max_mx_1, max_mx_2, max_mx_3])[
            [c for c in cols_sh_mx if c in max_mx_1.columns]
        ].style.format(formatos_mx),
        use_container_width=True,
    )

# ------------------------------------------------------------------------------
# PESTAÑA 2: ESTADOS UNIDOS
# ------------------------------------------------------------------------------
with tab_usa:
  st.header("🇺🇸 Análisis de Portafolios - Activos de EE. UU. y T-Bills")
  sub_usa_1, sub_usa_2, sub_usa_3, sub_usa_comp = st.tabs(
      ["📅 1 Año", "📅 2 Años", "📅 3 Años", "📋 Tablas Comparativas (USA)"]
  )

  with sub_usa_1:
    st.plotly_chart(
        construir_figura_animada(
            anim_usa_1, "1 Año (EE. UU.)", "T-Bills", is_mx=False
        ),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_usa_1[
            [c for c in cols_min_usa if c in min_usa_1.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Portafolio Óptimo Global (Máximo Sharpe Absoluto)")
    st.dataframe(
        max_usa_1[
            [c for c in cols_sh_usa if c in max_usa_1.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios de las Corridas (Top 10)")
    st.dataframe(
        resaltar_maximo_sharpe(
            t1000_usa_1.head(n_corridas)[
                [c for c in cols_sh_usa if c in t1000_usa_1.columns]
            ]
        ).format(formatos_usa),
        use_container_width=True,
    )

  with sub_usa_2:
    st.plotly_chart(
        construir_figura_animada(
            anim_usa_2, "2 Años (EE. UU.)", "T-Bills", is_mx=False
        ),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_usa_2[
            [c for c in cols_min_usa if c in min_usa_2.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Portafolio Óptimo Global (Máximo Sharpe Absoluto)")
    st.dataframe(
        max_usa_2[
            [c for c in cols_sh_usa if c in max_usa_2.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios de las Corridas (Top 10)")
    st.dataframe(
        resaltar_maximo_sharpe(
            t1000_usa_2.head(n_corridas)[
                [c for c in cols_sh_usa if c in t1000_usa_2.columns]
            ]
        ).format(formatos_usa),
        use_container_width=True,
    )

  with sub_usa_3:
    st.plotly_chart(
        construir_figura_animada(
            anim_usa_3, "3 Años (EE. UU.)", "T-Bills", is_mx=False
        ),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_usa_3[
            [c for c in cols_min_usa if c in min_usa_3.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Portafolio Óptimo Global (Máximo Sharpe Absoluto)")
    st.dataframe(
        max_usa_3[
            [c for c in cols_sh_usa if c in max_usa_3.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios de las Corridas (Top 10)")
    st.dataframe(
        resaltar_maximo_sharpe(
            t1000_usa_3.head(n_corridas)[
                [c for c in cols_sh_usa if c in t1000_usa_3.columns]
            ]
        ).format(formatos_usa),
        use_container_width=True,
    )

  with sub_usa_comp:
    st.subheader("📋 Tablas Comparativas (EE. UU.)")
    st.markdown("### Mínima Desviación")
    st.dataframe(
        pd.concat([min_usa_1, min_usa_2, min_usa_3])[
            [c for c in cols_min_usa if c in min_usa_1.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("### Máximo Sharpe Ratios")
    st.dataframe(
        pd.concat([max_usa_1, max_usa_2, max_usa_3])[
            [c for c in cols_sh_usa if c in max_usa_1.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )

# ------------------------------------------------------------------------------
# PESTAÑA 3: MODELO EXCEL ESTOCÁSTICO & TÍTULOS ENTEROS (NP.FLOOR)
# ------------------------------------------------------------------------------
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
    mx_p, usa_p = max_mx_1, max_usa_1
  elif horizonte_excel == "2 años":
    mx_p, usa_p = max_mx_2, max_usa_2
  else:
    mx_p, usa_p = max_mx_3, max_usa_3

  precios_mx_actuales = {}
  for k, v in tickers_mx.items():
    try:
      df_p = yf.download(v, period="10d", progress=False)["Close"]
      if isinstance(df_p, pd.DataFrame):
        df_p = df_p.iloc[:, 0]
      precios_mx_actuales[k] = float(df_p.dropna().iloc[-1])
    except:
      precios_mx_actuales[k] = 100.0

  precios_usa_actuales = {}
  for k, v in tickers_usa.items():
    try:
      df_p = yf.download(v, period="10d", progress=False)["Close"]
      if isinstance(df_p, pd.DataFrame):
        df_p = df_p.iloc[:, 0]
      precios_usa_actuales[k] = float(df_p.dropna().iloc[-1])
    except:
      precios_usa_actuales[k] = 150.0

  # TABLA 1: ACCIONES MÉXICO ($500,000 MXN)
  st.subheader("🇲🇽 Asignación de Títulos - Portafolio México ($500,000 MXN)")
  pesos_mx_opt = mx_p[empresas_mx].values[0]
  capital_mx_acc = 500000.0

  df_titulos_mx = pd.DataFrame({
      "Empresa": empresas_mx,
      "Peso Óptimo": pesos_mx_opt,
      "Capital Asignado (MXN)": pesos_mx_opt * capital_mx_acc,
      "Precio Actual (MXN)": [precios_mx_actuales[e] for e in empresas_mx],
  })

  # Redondeo entero hacia abajo (tipo 5.46 -> 5 títulos)
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

  # TABLA 2: ACCIONES EE. UU. ($245,000 MXN / TC)
  st.subheader(
      "🇺🇸 Asignación de Títulos - Portafolio EE. UU. ($245,000 MXN / USD)"
  )
  pesos_usa_opt = usa_p[empresas_usa].values[0]
  tc_inicial = 18.44
  capital_usa_usd = 245000.0 / tc_inicial

  df_titulos_usa = pd.DataFrame({
      "Empresa": empresas_usa,
      "Peso Óptimo": pesos_usa_opt,
      "Capital Asignado (USD)": pesos_usa_opt * capital_usa_usd,
      "Precio Actual (USD)": [precios_usa_actuales[e] for e in empresas_usa],
  })

  # Redondeo entero hacia abajo
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

  st.subheader("📉 Simulación Diaria Estocástica (Tasas Estables)")
  dias_muestra = 10
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
      tc = tc_base
    else:
      val_mx *= 1 + np.random.normal(0.0005, 0.012)
      val_cetes *= 1 + tasa_c / 360
      val_usa_usd *= 1 + np.random.normal(0.0006, 0.015)
      val_tb_usd *= 1 + tasa_t / 360
      u_w = np.random.uniform(0.3, 0.7)
      tc = 18.32123 * (u_w / (1 - u_w)) ** (1 / 28.52977)

    total_usa_mxn = (val_usa_usd + val_tb_usd) * tc
    total_global = val_mx + val_cetes + total_usa_mxn

    filas_excel.append({
        "Fecha": f,
        "Portafolio Acc. MX ($)": val_mx,
        "Tasa CETES (Triang)": tasa_c,
        "Valor CETES ($)": val_cetes,
        "Portafolio Acc. USA (USD)": val_usa_usd,
        "Tasa T-Bills (Triang)": tasa_t,
        "Valor T-Bills (USD)": val_tb_usd,
        "Tipo Cambio USD/MXN": tc,
        "Valor Total USA (MXN)": total_usa_mxn,
        "VALOR TOTAL GLOBAL (MXN)": total_global,
    })

  df_excel_demo = pd.DataFrame(filas_excel)
  st.dataframe(
      df_excel_demo.style.format({
          "Portafolio Acc. MX ($)": "${:,.2f}",
          "Tasa CETES (Triang)": "{:.2%}",
          "Valor CETES ($)": "${:,.2f}",
          "Portafolio Acc. USA (USD)": "${:,.2f}",
          "Tasa T-Bills (Triang)": "{:.2%}",
          "Valor T-Bills (USD)": "${:,.2f}",
          "Tipo Cambio USD/MXN": "{:.4f}",
          "Valor Total USA (MXN)": "${:,.2f}",
          "VALOR TOTAL GLOBAL (MXN)": "${:,.2f}",
      }),
      use_container_width=True,
  )

# ------------------------------------------------------------------------------
# PESTAÑA 4: ESTADÍSTICOS & RIESGO
# ------------------------------------------------------------------------------
with tab_stats:
  st.header("📊 Tabla de Estadísticos & Métricas de Riesgo Estocástico")
  horizonte_stats = st.selectbox(
      "Horizonte para los estadísticos:",
      ["1 año", "2 años", "3 años"],
      key="sel_stats",
  )

  if horizonte_stats == "1 año":
    m_mx, m_usa = max_mx_1, max_usa_1
  elif horizonte_stats == "2 años":
    m_mx, m_usa = max_mx_2, max_usa_2
  else:
    m_mx, m_usa = max_mx_3, max_usa_3

  df_stats, rend_glob_sim, rend_mx_sim, rend_usa_sim = (
      generar_simulacion_y_estadisticos(
          m_mx, m_usa, dias_habiles, alpha_nivel
      )
  )

  formatos_stats = {col: "{:.2%}" for col in df_stats.columns}
  st.dataframe(df_stats.style.format(formatos_stats), use_container_width=True)

# ------------------------------------------------------------------------------
# PESTAÑA 5: GRÁFICAS DE PRECIOS ESCALADOS & MATRIZ DE COVARIANZA DINÁMICAS
# ------------------------------------------------------------------------------
with tab_charts:
  st.header("📈 Gráficas Escaladas de Precios y Matrices de Covarianza")

  col_sel1, col_sel2 = st.columns(2)
  with col_sel1:
    mercado_chart = st.selectbox(
        "Selecciona el Mercado:", ["México (BMV)", "EE. UU. (Tech)"]
    )
  with col_sel2:
    horiz_chart = st.selectbox(
        "Selecciona el Horizonte:", ["1 año", "2 años", "3 años"]
    )

  if mercado_chart == "México (BMV)":
    if horiz_chart == "1 año":
      d_m, c_m, min_h, empresas_act = (
          datos_mx_1,
          cov_mx_1,
          min_mx_1,
          empresas_mx,
      )
    elif horiz_chart == "2 años":
      d_m, c_m, min_h, empresas_act = (
          datos_mx_2,
          cov_mx_2,
          min_mx_2,
          empresas_mx,
      )
    else:
      d_m, c_m, min_h, empresas_act = (
          datos_mx_3,
          cov_mx_3,
          min_mx_3,
          empresas_mx,
      )
    titulo_mercado = "Mercado Mexicano"
  else:
    if horiz_chart == "1 año":
      d_m, c_m, min_h, empresas_act = (
          datos_usa_1,
          cov_usa_1,
          min_usa_1,
          empresas_usa,
      )
    elif horiz_chart == "2 años":
      d_m, c_m, min_h, empresas_act = (
          datos_usa_2,
          cov_usa_2,
          min_usa_2,
          empresas_usa,
      )
    else:
      d_m, c_m, min_h, empresas_act = (
          datos_usa_3,
          cov_usa_3,
          min_usa_3,
          empresas_usa,
      )
    titulo_mercado = "Mercado EE. UU."

  st.subheader(
      f"📉 {titulo_mercado} - Precios Normalizados (Base 100) [{horiz_chart}]"
  )
  precios_norm = (d_m / d_m.iloc[0]) * 100
  fig_norm = px.line(
      precios_norm,
      title=(
          f"Evolución Comparativa de Precios Escalados ({titulo_mercado} -"
          f" {horiz_chart})"
      ),
  )

  ret_diarios = np.log(d_m / d_m.shift(1)).dropna()
  w_min = min_h[empresas_act].values[0]
  port_min_var = (
      ret_diarios[empresas_act].dot(w_min).cumsum().apply(np.exp) * 100
  )

  fig_norm.add_trace(
      go.Scatter(
          x=port_min_var.index,
          y=port_min_var,
          mode="lines",
          line=dict(color="blue", width=3, dash="dash"),
          name="🔵 Mínima Varianza (Portafolio)",
      )
  )
  st.plotly_chart(fig_norm, use_container_width=True)

  st.subheader(f"🔷 Matriz de Covarianza Anualizada ({titulo_mercado})")
  fig_cov = px.imshow(
      c_m,
      x=empresas_act,
      y=empresas_act,
      color_continuous_scale="Viridis",
      text_auto=".4f",
      title=f"Matriz de Covarianza (Sigma) - {horiz_chart}",
  )
  st.plotly_chart(fig_cov, use_container_width=True)

# ------------------------------------------------------------------------------
# PESTAÑA 6: ANEXO DE FÓRMULAS QUANTITATIVE & RIESGO
# ------------------------------------------------------------------------------
with tab_anexo:
  st.header("📐 Anexo Cuantitativo: Fórmulas y Modelos Matemáticos")
  st.markdown(
      "Resumen técnico de las ecuaciones fundamentales implementadas en la"
      " aplicación."
  )

  col1, col2 = st.columns(2)

  with col1:
    st.markdown("### 1. Rendimiento y Volatilidad del Portafolio")
    st.latex(r"R_p = \sum_{i=1}^{n} w_i \cdot \mu_i = \mathbf{w}^T \boldsymbol{\mu}")
    st.latex(
        r"\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w} = \sum_{i=1}^n"
        r" \sum_{j=1}^n w_i w_j \sigma_{ij}"
    )
    st.latex(r"\sigma_p = \sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}")

    st.markdown("### 2. Ratio de Sharpe & Línea de Asignación de Capital (CAL)")
    st.latex(r"Sharpe = \frac{R_p - R_f}{\sigma_p}")
    st.latex(r"E(R_c) = R_f + Sharpe \cdot \sigma_c")

  with col2:
    st.markdown("### 3. Distribución Triangular (CETES / T-Bills)")
    st.latex(r"k = \frac{b - a}{c - a}")
    st.latex(r"""
        T_L = \begin{cases} 
        a + \sqrt{U \cdot (b - a)(c - a)} & \text{si } U < k \\ 
        c - \sqrt{(1 - U)(c - a)(c - b)} & \text{si } U \ge k 
        \end{cases}
        """)

    st.markdown("### 4. Métricas de Riesgo (Cutoff & VaR)")
    st.latex(r"Cutoff_{\alpha} = Q_{\alpha}(R_{global})")
    st.latex(r"VaR_{\alpha} = \mu_{global} - Cutoff_{\alpha}")

  st.markdown("---")
  st.markdown("### 5. Tipo de Cambio Estocástico (Distribución Weibull)")
  st.latex(r"TC = \lambda \cdot \left( \frac{U}{1 - U} \right)^{\frac{1}{k}}")
