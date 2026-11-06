import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Configuración de la página en Streamlit
st.set_page_config(
    page_title="Simulador de Portafolios Internacionales & Monte Carlo",
    page_icon="📈",
    layout="wide",
)

st.title(
    "📊 Simulador de Portafolios de Inversión"
)
st.markdown(
    "Modelo cuantitativo multianual con optimización para activos de la BMV"
    " (CETES) y de EE. UU. (T-Bills)."
)

# ==============================================================================
# 1. SIDEBAR: PARÁMETROS GENERALES
# ==============================================================================
st.sidebar.header("⚙️ Configuración del Análisis")

n_simulaciones = st.sidebar.slider(
    "Número de Portafolios (Simulación):", 1000, 10000, 5000, 500
)
n_corridas = st.sidebar.slider(
    "Corridas de Optimización (Sharpe):", 50, 500, 200, 50
)

# Definición de mercados
empresas_mx = ["Grupo México", "Lamosa", "América Movil", "Bimbo", "Kimbera"]
tickers_mx = {
    "Grupo México": "GMEXICOB.MX",
    "Lamosa": "LAMOSA.MX",
    "América Movil": "AMXB.MX",
    "Bimbo": "BIMBOA.MX",
    "Kimbera": "KIMBERA.MX",
}

empresas_usa = ["Microsoft", "Netflix", "Spotify", "Intel", "Apple"]
tickers_usa = {
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Spotify": "SPOT",
    "Intel": "INTC",
    "Apple": "AAPL",
}

fechas_horizontes = {
    "1 año": {"inicio": "2025-09-18", "fin": "2026-09-16"},
    "2 años": {"inicio": "2024-09-18", "fin": "2026-09-16"},
    "3 años": {"inicio": "2023-09-18", "fin": "2026-09-16"},
}


# ==============================================================================
# 2. FUNCIÓN DE SIMULACIÓN GENERAL (GENÉRICA PARA AMBOS MERCADOS)
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
    
  aleatorios_base = np.random.uniform(
      0, 1, size=(len(datos) - 1, len(lista_empresas))
  )
  wi_base = aleatorios_base / aleatorios_base.sum(axis=1, keepdims=True)
  matriz_cov_wi = pd.DataFrame(wi_base, columns=lista_empresas).cov(ddof=0)
  Sigma_wi = matriz_cov_wi.values

  # Configuración de tasa libre de riesgo (Triangular)
  if tipo_tasa == "cetes":
    a, b, c = 0.04, 0.08, 0.10
  else:  # t-bills
    a, b, c = 0.0360, 0.0370, 0.0400

  k = (b - a) / (c - a)
  dias_ano = 252

  mejores_sharpe = []
  datos_animacion = []
  n_animacion_pasos = min(20, n_corr)
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

    df_sim["Rp"] = df_sim[lista_empresas].dot(promedio)
    W_sim = df_sim[lista_empresas].values
    vp_sim = np.einsum("ij,jk,ik->i", W_sim, Sigma_wi, W_sim)
    df_sim["vp"] = vp_sim
    df_sim["des_p"] = np.sqrt(vp_sim)
    df_sim["sharpe_ratio"] = (df_sim["Rp"] - diario) / df_sim["des_p"]

    if i == 1:
      df_sim_global = df_sim.copy()

    idx_max_sharpe = df_sim["sharpe_ratio"].idxmax()
    mejor_fila = df_sim.loc[[idx_max_sharpe]].copy()
    mejor_fila.index = [f"{i}"]
    mejor_fila["Tasa_diaria"] = diario
    mejores_sharpe.append(mejor_fila)

    if i <= n_animacion_pasos:
      Rp_anual_sim = df_sim["Rp"] * dias_ano
      des_p_anual_sim = df_sim["des_p"] * np.sqrt(dias_ano)
      sharpe_anual_sim = (Rp_anual_sim - tasa_anual) / des_p_anual_sim
      idx_opt = sharpe_anual_sim.idxmax()

      riesgo_opt_anual = des_p_anual_sim.loc[idx_opt]
      ret_opt_anual = Rp_anual_sim.loc[idx_opt]
      max_sh_anual = sharpe_anual_sim.loc[idx_opt]

      x_cal = np.linspace(0, riesgo_opt_anual * 1.25, 30)
      y_cal = tasa_anual + max_sh_anual * x_cal

      r_comb = w_portafolio * riesgo_opt_anual
      ret_comb = (w_cetes * tasa_anual) + (w_portafolio * ret_opt_anual)

      df_vis = df_sim.sample(
          n=min(3000, len(df_sim)), random_state=42
      ).copy()

      datos_animacion.append({
          "tasa_anual": tasa_anual,
          "nube_x": (df_vis["des_p"] * np.sqrt(dias_ano)).values,
          "nube_y": (df_vis["Rp"] * dias_ano).values,
          "x_cal": x_cal,
          "y_cal": y_cal,
          "star_x": riesgo_opt_anual,
          "star_y": ret_opt_anual,
          "mix_x": r_comb,
          "mix_y": ret_comb,
          "color": colores_anim[i - 1],
      })

  tabla_sharpe_max = pd.concat(mejores_sharpe)
  idx_min_desp = df_sim_global["des_p"].idxmin()
  min_port = df_sim_global.loc[[idx_min_desp]].copy()

  idx_max_global = tabla_sharpe_max["sharpe_ratio"].idxmax()
  max_sharpe_port = tabla_sharpe_max.loc[[idx_max_global]].copy()

  return min_port, max_sharpe_port, datos_animacion, tabla_sharpe_max


with st.spinner(
    "Calculando simulaciones para México y Estados Unidos (1, 2 y 3 años)..."
):
  # Resultados México (CETES)
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

  # Resultados USA (T-Bills)
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

  # Desempaquetar MX
  min_mx_1, max_mx_1, anim_mx_1, t1000_mx_1 = res_mx_1
  min_mx_2, max_mx_2, anim_mx_2, t1000_mx_2 = res_mx_2
  min_mx_3, max_mx_3, anim_mx_3, t1000_mx_3 = res_mx_3

  # Desempaquetar USA
  min_usa_1, max_usa_1, anim_usa_1, t1000_usa_1 = res_usa_1
  min_usa_2, max_usa_2, anim_usa_2, t1000_usa_2 = res_usa_2
  min_usa_3, max_usa_3, anim_usa_3, t1000_usa_3 = res_usa_3

# Asignar índices de periodo
for m in [min_mx_1, max_mx_1, t1000_mx_1]:
  m.index = ["1 año" if len(m) == 1 else m.index]
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
# 3. FUNCIÓN DE GRÁFICA ANIMADA
# ==============================================================================
def construir_figura_animada(datos_anim, titulo_horizonte, lbl_tasa="CETES"):
  fig = go.Figure()
  n_pasos = len(datos_anim)

  fig.add_trace(
      go.Scatter(
          x=[],
          y=[],
          mode="markers",
          marker=dict(size=3.5, color="lightgreen", opacity=0.45),
          name="Portafolios Simulados",
      )
  )
  for j in range(n_pasos):
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="red", width=1.2, dash="dash"),
            opacity=0.4,
            showlegend=False,
        )
    )
  fig.add_trace(
      go.Scatter(
          x=[],
          y=[],
          mode="markers",
          marker=dict(
              size=11, color="gold", symbol="star", line=dict(width=1, color="black")
          ),
          name="Máximo Sharpe",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=[],
          y=[],
          mode="markers",
          marker=dict(
              size=9,
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
  for f_idx in range(n_pasos):
    f_data = []
    curr = datos_anim[f_idx]
    f_data.append(
        go.Scatter(
            x=curr["nube_x"],
            y=curr["nube_y"],
            marker=dict(size=3.5, color=curr["color"]),
        )
    )
    for j in range(n_pasos):
      if j <= f_idx:
        f_data.append(
            go.Scatter(x=datos_anim[j]["x_cal"], y=datos_anim[j]["y_cal"])
        )
      else:
        f_data.append(go.Scatter(x=[], y=[]))
    f_data.append(
        go.Scatter(
            x=[datos_anim[j]["star_x"] for j in range(f_idx + 1)],
            y=[datos_anim[j]["star_y"] for j in range(f_idx + 1)],
        )
    )
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

  fig.update_layout(
      title=f"Simulación Monte Carlo Paso a Paso ({titulo_horizonte})",
      template="plotly_white",
      xaxis=dict(title="Volatility (Riesgo Anual)", range=[0, 1.2]),
      yaxis=dict(title="Return (Rendimiento Anual)", range=[-0.05, 0.35]),
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


# ==============================================================================
# 4. INTERFAZ PRINCIPAL CON PESTAÑAS PRINCIPALES (MÉXICO VS. USA)
# ==============================================================================
tab_mx, tab_usa = st.tabs(
    ["🇲🇽 Mercado Mexicano (BMV / CETES)", "🇺🇸 Mercado Estadounidense (EE. UU. / T-Bills)"]
)

# ------------------------------------------------------------------------------
# PESTAÑA MÉXICO
# ------------------------------------------------------------------------------
with tab_mx:
  st.header("🇲🇽 Análisis de Portafolios - Activos Mexicanos y CETES")
  sub_mx_1, sub_mx_2, sub_mx_3, sub_mx_comp = st.tabs(
      ["📅 1 Año", "📅 2 Años", "📅 3 Años", "📋 Tablas Comparativas (MX)"]
  )

  formatos_mx = {c: "{:.2%}" for c in empresas_mx}
  formatos_mx.update({
      "Rp": "{:.4%}",
      "des_p": "{:.4%}",
      "Tasa_diaria": "{:.4%}",
      "sharpe_ratio": "{:.4f}",
  })
  cols_min_mx = [c for c in empresas_mx + ["Rp", "vp", "des_p"]]
  cols_sh_mx = [
      c for c in empresas_mx + ["Rp", "vp", "des_p", "Tasa_diaria", "sharpe_ratio"]
  ]

  with sub_mx_1:
    st.plotly_chart(
        construir_figura_animada(anim_mx_1, "1 Año (México)", "CETES"),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_mx_1[[c for c in cols_min_mx if c in min_mx_1.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Máximo Sharpe")
    st.dataframe(
        max_mx_1[[c for c in cols_sh_mx if c in max_mx_1.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios (Top 10)")
    st.dataframe(
        t1000_mx_1.head(10)[
            [c for c in cols_sh_mx if c in t1000_mx_1.columns]
        ].style.format(formatos_mx),
        use_container_width=True,
    )

  with sub_mx_2:
    st.plotly_chart(
        construir_figura_animada(anim_mx_2, "2 Años (México)", "CETES"),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_mx_2[[c for c in cols_min_mx if c in min_mx_2.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Máximo Sharpe")
    st.dataframe(
        max_mx_2[[c for c in cols_sh_mx if c in max_mx_2.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios (Top 10)")
    st.dataframe(
        t1000_mx_2.head(10)[
            [c for c in cols_sh_mx if c in t1000_mx_2.columns]
        ].style.format(formatos_mx),
        use_container_width=True,
    )

  with sub_mx_3:
    st.plotly_chart(
        construir_figura_animada(anim_mx_3, "3 Años (México)", "CETES"),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_mx_3[[c for c in cols_min_mx if c in min_mx_3.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Máximo Sharpe")
    st.dataframe(
        max_mx_3[[c for c in cols_sh_mx if c in max_mx_3.columns]].style.format(
            formatos_mx
        ),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios (Top 10)")
    st.dataframe(
        t1000_mx_3.head(10)[
            [c for c in cols_sh_mx if c in t1000_mx_3.columns]
        ].style.format(formatos_mx),
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
# PESTAÑA USA
# ------------------------------------------------------------------------------
with tab_usa:
  st.header("🇺🇸 Análisis de Portafolios - Activos de EE. UU. y T-Bills")
  sub_usa_1, sub_usa_2, sub_usa_3, sub_usa_comp = st.tabs(
      ["📅 1 Año", "📅 2 Años", "📅 3 Años", "📋 Tablas Comparativas (USA)"]
  )

  formatos_usa = {c: "{:.2%}" for c in empresas_usa}
  formatos_usa.update({
      "Rp": "{:.4%}",
      "des_p": "{:.4%}",
      "Tasa_diaria": "{:.4%}",
      "sharpe_ratio": "{:.4f}",
  })
  cols_min_usa = [c for c in empresas_usa + ["Rp", "vp", "des_p"]]
  cols_sh_usa = [
      c
      for c in empresas_usa
      + ["Rp", "vp", "des_p", "Tasa_diaria", "sharpe_ratio"]
  ]

  with sub_usa_1:
    st.plotly_chart(
        construir_figura_animada(anim_usa_1, "1 Año (EE. UU.)", "T-Bills"),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_usa_1[
            [c for c in cols_min_usa if c in min_usa_1.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Máximo Sharpe")
    st.dataframe(
        max_usa_1[
            [c for c in cols_sh_usa if c in max_usa_1.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios (Top 10)")
    st.dataframe(
        t1000_usa_1.head(10)[
            [c for c in cols_sh_usa if c in t1000_usa_1.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )

  with sub_usa_2:
    st.plotly_chart(
        construir_figura_animada(anim_usa_2, "2 Años (EE. UU.)", "T-Bills"),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_usa_2[
            [c for c in cols_min_usa if c in min_usa_2.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Máximo Sharpe")
    st.dataframe(
        max_usa_2[
            [c for c in cols_sh_usa if c in max_usa_2.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios (Top 10)")
    st.dataframe(
        t1000_usa_2.head(10)[
            [c for c in cols_sh_usa if c in t1000_usa_2.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )

  with sub_usa_3:
    st.plotly_chart(
        construir_figura_animada(anim_usa_3, "3 Años (EE. UU.)", "T-Bills"),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Mínima Desviación")
    st.dataframe(
        min_usa_3[
            [c for c in cols_min_usa if c in min_usa_3.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Portafolio de Máximo Sharpe")
    st.dataframe(
        max_usa_3[
            [c for c in cols_sh_usa if c in max_usa_3.columns]
        ].style.format(formatos_usa),
        use_container_width=True,
    )
    st.markdown("#### Mejores Portafolios (Top 10)")
    st.dataframe(
        t1000_usa_3.head(10)[
            [c for c in cols_sh_usa if c in t1000_usa_3.columns]
        ].style.format(formatos_usa),
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
