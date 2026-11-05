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
    "Modelo cuantitativo multianual con optimización de portafolios de la BMV y CETES."
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

empresas = ["Grupo México", "Lamosa", "América Movil", "Bimbo", "Kimbera"]
tickers = {
    "Grupo México": "GMEXICOB.MX",
    "Lamosa": "LAMOSA.MX",
    "América Movil": "AMXB.MX",
    "Bimbo": "BIMBOA.MX",
    "Kimbera": "KIMBERA.MX",
}

fechas_horizontes = {
    "1 año": {"inicio": "2025-09-18", "fin": "2026-09-16"},
    "2 años": {"inicio": "2024-09-18", "fin": "2026-09-16"},
    "3 años": {"inicio": "2023-09-18", "fin": "2026-09-16"},
}


# ==============================================================================
# 2. FUNCIÓN DE SIMULACIÓN Y ANIMACIÓN PASO A PASO
# ==============================================================================
@st.cache_data
def ejecutar_simulacion_completa(start, end, n_sim, n_corr):
  datos = yf.download(list(tickers.values()), start=start, end=end)["Close"]
  if isinstance(datos.columns, pd.MultiIndex):
    datos.columns = datos.columns.droplevel(0)
  datos.rename(
      columns={v: k for k, v in tickers.items()}, inplace=True
  )
  datos = datos[empresas].dropna()

  rendimientos = np.log(datos / datos.shift(1)).dropna()
  promedio = rendimientos.mean()

  np.random.seed(42)
  aleatorios_base = np.random.uniform(
      0, 1, size=(len(datos) - 1, len(empresas))
  )
  wi_base = aleatorios_base / aleatorios_base.sum(axis=1, keepdims=True)
  matriz_cov_wi = pd.DataFrame(wi_base, columns=empresas).cov(ddof=0)
  Sigma_wi = matriz_cov_wi.values

  a, b, c = 0.04, 0.08, 0.10
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
    p_cetes = np.random.uniform(0, 1)
    if p_cetes < k:
      TL = a + np.sqrt(p_cetes * (b - a) * (c - a))
    else:
      TL = c - np.sqrt((1 - p_cetes) * (c - a) * (c - b))

    diario = (1 + TL) ** (1 / 360) - 1
    cetes_anual = TL

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
      df_sim_global = df_sim.copy()

    idx_max_sharpe = df_sim["sharpe_ratio"].idxmax()
    mejor_fila = df_sim.loc[[idx_max_sharpe]].copy()
    mejor_fila.index = [f"{i}"]
    mejor_fila["CETES_diario"] = diario
    mejores_sharpe.append(mejor_fila)

    if i <= n_animacion_pasos:
      Rp_anual_sim = df_sim["Rp"] * dias_ano
      des_p_anual_sim = df_sim["des_p"] * np.sqrt(dias_ano)
      sharpe_anual_sim = (Rp_anual_sim - cetes_anual) / des_p_anual_sim
      idx_opt = sharpe_anual_sim.idxmax()

      riesgo_opt_anual = des_p_anual_sim.loc[idx_opt]
      ret_opt_anual = Rp_anual_sim.loc[idx_opt]
      max_sh_anual = sharpe_anual_sim.loc[idx_opt]

      x_cal = np.linspace(0, riesgo_opt_anual * 1.25, 30)
      y_cal = cetes_anual + max_sh_anual * x_cal

      r_comb = w_portafolio * riesgo_opt_anual
      ret_comb = (w_cetes * cetes_anual) + (w_portafolio * ret_opt_anual)

      df_vis = df_sim.sample(
          n=min(3000, len(df_sim)), random_state=42
      ).copy()

      datos_animacion.append({
          "cetes_anual": cetes_anual,
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


with st.spinner("Calculando simulaciones y optimizaciones Monte Carlo..."):
  res_1 = ejecutar_simulacion_completa(
      fechas_horizontes["1 año"]["inicio"],
      fechas_horizontes["1 año"]["fin"],
      n_simulaciones,
      n_corridas,
  )
  res_2 = ejecutar_simulacion_completa(
      fechas_horizontes["2 años"]["inicio"],
      fechas_horizontes["2 años"]["fin"],
      n_simulaciones,
      n_corridas,
  )
  res_3 = ejecutar_simulacion_completa(
      fechas_horizontes["3 años"]["inicio"],
      fechas_horizontes["3 años"]["fin"],
      n_simulaciones,
      n_corridas,
  )

  min_1yr, maxshare1, anim_1, tabla_1000_1yr = res_1
  min_2yr, maxshare2, anim_2, tabla_1000_2yr = res_2
  min_3yr, maxshare3, anim_3, tabla_1000_3yr = res_3

min_1yr.index, min_2yr.index, min_3yr.index = ["1 año"], ["2 años"], ["3 años"]
maxshare1.index, maxshare2.index, maxshare3.index = (
    ["1 año"],
    ["2 años"],
    ["3 años"],
)


def construir_figura_animada(datos_anim, titulo_horizonte):
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
          name="Mix 30% CETES / 70% Portafolio",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=[],
          y=[],
          mode="markers",
          marker=dict(size=8, color="darkred", symbol="circle"),
          name="CETES",
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
    f_data.append(go.Scatter(x=[0], y=[curr["cetes_anual"]]))
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
# 3. INTERFAZ DE PESTAÑAS (TABS)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📅 1 Año",
    "📅 2 Años",
    "📅 3 Años",
    "📋 Tablas Comparativas Generales",
])

formatos_porcentaje = {c: "{:.2%}" for c in empresas}
formatos_porcentaje.update({
    "Rp": "{:.4%}",
    "des_p": "{:.4%}",
    "CETES_diario": "{:.4%}",
    "sharpe_ratio": "{:.4f}",
})

cols_min = [c for c in empresas + ["Rp", "vp", "des_p"]]
cols_sharpe = [
    c for c in empresas + ["Rp", "vp", "des_p", "CETES_diario", "sharpe_ratio"]
]

# PESTAÑA 1: 1 AÑO
with tab1:
  st.subheader("Simulación y Frontera Eficiente - 1 Año")
  fig_1 = construir_figura_animada(anim_1, "1 Año")
  st.plotly_chart(fig_1, use_container_width=True)

  st.markdown("---")
  st.subheader("📑 Resultados Individuales (1 Año)")

  st.markdown("#### Portafolio de Mínima Desviación")
  st.dataframe(
      min_1yr[[c for c in cols_min if c in min_1yr.columns]].style.format(
          formatos_porcentaje
      ),
      use_container_width=True,
  )

  st.markdown("#### Portafolio de Máximo Sharpe (Individual)")
  st.dataframe(
      maxshare1[
          [c for c in cols_sharpe if c in maxshare1.columns]
      ].style.format(formatos_porcentaje),
      use_container_width=True,
  )

  st.markdown("#### Mejores Portafolios (Máximo Sharpe - Top 10)")
  st.dataframe(
      tabla_1000_1yr.head(10)[
          [c for c in cols_sharpe if c in tabla_1000_1yr.columns]
      ].style.format(formatos_porcentaje),
      use_container_width=True,
  )

# PESTAÑA 2: 2 AÑOS
with tab2:
  st.subheader("Simulación y Frontera Eficiente - 2 Años")
  fig_2 = construir_figura_animada(anim_2, "2 Años")
  st.plotly_chart(fig_2, use_container_width=True)

  st.markdown("---")
  st.subheader("📑 Resultados Individuales (2 Años)")

  st.markdown("#### Portafolio de Mínima Desviación")
  st.dataframe(
      min_2yr[[c for c in cols_min if c in min_2yr.columns]].style.format(
          formatos_porcentaje
      ),
      use_container_width=True,
  )

  st.markdown("#### Portafolio de Máximo Sharpe (Individual)")
  st.dataframe(
      maxshare2[
          [c for c in cols_sharpe if c in maxshare2.columns]
      ].style.format(formatos_porcentaje),
      use_container_width=True,
  )

  st.markdown("#### Mejores Portafolios (Máximo Sharpe - Top 10)")
  st.dataframe(
      tabla_1000_2yr.head(10)[
          [c for c in cols_sharpe if c in tabla_1000_2yr.columns]
      ].style.format(formatos_porcentaje),
      use_container_width=True,
  )

# PESTAÑA 3: 3 AÑOS
with tab3:
  st.subheader("Simulación y Frontera Eficiente - 3 Años")
  fig_3 = construir_figura_animada(anim_3, "3 Años")
  st.plotly_chart(fig_3, use_container_width=True)

  st.markdown("---")
  st.subheader("📑 Resultados Individuales (3 Años)")

  st.markdown("#### Portafolio de Mínima Desviación")
  st.dataframe(
      min_3yr[[c for c in cols_min if c in min_3yr.columns]].style.format(
          formatos_porcentaje
      ),
      use_container_width=True,
  )

  st.markdown("#### Portafolio de Máximo Sharpe (Individual)")
  st.dataframe(
      maxshare3[
          [c for c in cols_sharpe if c in maxshare3.columns]
      ].style.format(formatos_porcentaje),
      use_container_width=True,
  )

  st.markdown("#### Mejores Portafolios (Máximo Sharpe - Top 10)")
  st.dataframe(
      tabla_1000_3yr.head(10)[
          [c for c in cols_sharpe if c in tabla_1000_3yr.columns]
      ].style.format(formatos_porcentaje),
      use_container_width=True,
  )

# PESTAÑA 4: TABLAS COMPARATIVAS GENERALES
with tab4:
  st.subheader("📋 Tablas Comparativas Multianuales")

  st.markdown("### 📉 Tabla Comparativa: Portafolios de Mínima Desviación")
  tabla_min = pd.concat([min_1yr, min_2yr, min_3yr])
  st.dataframe(
      tabla_min[[c for c in cols_min if c in tabla_min.columns]].style.format(
          formatos_porcentaje
      ),
      use_container_width=True,
  )

  st.markdown("---")
  st.markdown("### 🏆 Tabla Comparativa: Máximos Sharpe Ratios (1, 2 y 3 Años)")
  tabla_sharpe = pd.concat([maxshare1, maxshare2, maxshare3])
  st.dataframe(
      tabla_sharpe[
          [c for c in cols_sharpe if c in tabla_sharpe.columns]
      ].style.format(formatos_porcentaje),
      use_container_width=True,
  )
