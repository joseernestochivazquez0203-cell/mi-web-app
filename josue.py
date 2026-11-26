# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import yfinance as yf

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="Modelación Portafolio Mexicano (70/30)",
    page_icon="📈",
    layout="wide",
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

EMPRESAS_DEFAULT = ["AMXB.MX", "BIMBOA.MX", "CEMEXCPO.MX", "KIMBERA.MX", "WALMEX.MX"]
PERIODOS_ORDEN = ["1 Año", "2 Años", "3 Años"]

TASAS_CETES_TRIANGULAR = {
    "1 Año": {"min": 6.13, "moda": 6.81, "max": 7.47},
    "2 Años": {"min": 6.13, "moda": 6.81, "max": 10.40},
    "3 Años": {"min": 6.13, "moda": 11.00, "max": 11.30},
}


# ==========================================
# 2. FUNCIONES AUXILIARES Y CÁLCULO (CACHEADAS)
# ==========================================
def obtener_fechas_periodos(fecha_fin_dt: pd.Timestamp) -> dict:
    return {
        "1 Año": fecha_fin_dt - pd.DateOffset(years=1),
        "2 Años": fecha_fin_dt - pd.DateOffset(years=2),
        "3 Años": fecha_fin_dt - pd.DateOffset(years=3),
    }


@st.cache_data(show_spinner="Descargando datos históricos de Yahoo Finance...")
def descargar_precios(empresas: tuple, fecha_inicio_str: str, fecha_fin_str: str) -> pd.DataFrame:
    raw = yf.download(
        list(empresas),
        start=fecha_inicio_str,
        end=fecha_fin_str,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        datos = raw["Close"]
    else:
        datos = raw
    return datos.dropna(how="all").dropna()


def construir_bloque_periodo(datos: pd.DataFrame, cov_anual: pd.DataFrame, rend_anual: pd.Series) -> dict:
    base100 = (datos / datos.dropna().bfill().iloc[0]) * 100
    desc = datos.describe()
    tickers = list(datos.columns)

    estadisticas = [
        {
            "ticker": t,
            "inicio": round(float(datos[t].dropna().iloc[0]), 4),
            "fin": round(float(datos[t].dropna().iloc[-1]), 4),
            "media": round(float(desc.loc["mean", t]), 4),
            "std": round(float(desc.loc["std", t]), 4),
            "min": round(float(desc.loc["min", t]), 4),
            "max": round(float(desc.loc["max", t]), 4),
            "rend_anual_pct": round(float(rend_anual[t] * 100), 2),
            "vol_anual_pct": round(float(np.sqrt(cov_anual.loc[t, t]) * 100), 2),
        }
        for t in tickers
    ]

    return {
        "rango_fechas": f"{datos.index[0].strftime('%Y-%m-%d')} a {datos.index[-1].strftime('%Y-%m-%d')}",
        "observaciones": int(len(datos)),
        "fechas": [d.strftime("%Y-%m-%d") for d in base100.index],
        "base100": {t: [round(float(v), 2) for v in base100[t]] for t in tickers},
        "estadisticas": estadisticas,
        "tickers": tickers,
        "covarianza": [[round(float(cov_anual.loc[r, c]), 6) for c in tickers] for r in tickers],
    }


@st.cache_data(show_spinner="Ejecutando macro-simulación Monte Carlo...")
def ejecutar_macro_simulacion(
    empresas: tuple,
    fecha_fin_str: str,
    num_bloques: int,
    sims_por_bloque: int,
    ponderacion_acciones: float,
    semilla: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ponderacion_cetes = round(1.0 - ponderacion_acciones, 4)
    fecha_fin_dt = pd.to_datetime(fecha_fin_str)
    periodos = obtener_fechas_periodos(fecha_fin_dt)

    rng = np.random.default_rng(semilla)
    df_pesos = pd.DataFrame({"Empresa / Activo": list(empresas)})
    resumen_periodos = {}
    num_activos = len(empresas)

    for periodo in PERIODOS_ORDEN:
        fecha_inicio = periodos[periodo]
        datos_periodo = descargar_precios(
            empresas, fecha_inicio.strftime("%Y-%m-%d"), fecha_fin_dt.strftime("%Y-%m-%d")
        )
        r_log = np.log(datos_periodo / datos_periodo.shift(1)).dropna()
        cov_anual = r_log.cov() * 252
        rend_anual = r_log.mean() * 252

        c_min = TASAS_CETES_TRIANGULAR[periodo]["min"] / 100.0
        c_moda = TASAS_CETES_TRIANGULAR[periodo]["moda"] / 100.0
        c_max = TASAS_CETES_TRIANGULAR[periodo]["max"] / 100.0

        maximos_locales_sharpe = np.zeros(num_bloques)
        mejores_pesos_por_bloque = np.zeros((num_bloques, num_activos))
        mejores_rf_por_bloque = np.zeros(num_bloques)
        mejores_rend_por_bloque = np.zeros(num_bloques)
        mejores_vol_por_bloque = np.zeros(num_bloques)

        cov_vals = cov_anual.values
        rend_vals = rend_anual.values

        for b in range(num_bloques):
            rf_simuladas = rng.triangular(left=c_min, mode=c_moda, right=c_max, size=sims_por_bloque)

            pesos_sim = rng.random((sims_por_bloque, num_activos))
            pesos_sim /= pesos_sim.sum(axis=1, keepdims=True)

            rend_acciones_sim = pesos_sim @ rend_vals
            var_acciones_sim = np.einsum("ij,jk,ik->i", pesos_sim, cov_vals, pesos_sim)
            vol_acciones_sim = np.sqrt(var_acciones_sim)

            rend_comb_sim = (ponderacion_acciones * rend_acciones_sim) + (ponderacion_cetes * rf_simuladas)
            vol_comb_sim = ponderacion_acciones * vol_acciones_sim

            sharpe_comb_sim = np.where(
                vol_comb_sim > 0, (rend_comb_sim - rf_simuladas) / vol_comb_sim, -np.inf
            )

            idx_max_local = int(np.argmax(sharpe_comb_sim))
            maximos_locales_sharpe[b] = sharpe_comb_sim[idx_max_local]
            mejores_pesos_por_bloque[b, :] = pesos_sim[idx_max_local, :]
            mejores_rf_por_bloque[b] = rf_simuladas[idx_max_local]
            mejores_rend_por_bloque[b] = rend_comb_sim[idx_max_local]
            mejores_vol_por_bloque[b] = vol_comb_sim[idx_max_local]

        idx_max_global = int(np.argmax(maximos_locales_sharpe))
        pesos_optimos_raw = mejores_pesos_por_bloque[idx_max_global, :]

        pesos_decimales = np.round(pesos_optimos_raw * 100, 2)
        dif = round(100.0 - float(pesos_decimales.sum()), 2)
        if dif != 0.0:
            pesos_decimales[int(np.argmax(pesos_decimales))] = round(
                float(pesos_decimales[int(np.argmax(pesos_decimales))]) + dif, 2
            )

        pesos_finales_acc = (pesos_decimales / 100.0) * ponderacion_acciones * 100.0
        df_pesos[f"Peso {periodo} (%)"] = np.round(pesos_finales_acc, 2)

        resumen_periodos[periodo] = {
            "Max_Sharpe_Global": float(maximos_locales_sharpe[idx_max_global]),
            "Tasa_CETES_Optima (%)": float(mejores_rf_por_bloque[idx_max_global] * 100),
            "Rendimiento_Comb (%)": float(mejores_rend_por_bloque[idx_max_global] * 100),
            "Volatilidad_Comb (%)": float(mejores_vol_por_bloque[idx_max_global] * 100),
            "Vector_Maximos": maximos_locales_sharpe,
            "Volatilidad_Bloques": mejores_vol_por_bloque,
            "Rendimiento_Bloques": mejores_rend_por_bloque,
            "Rf_Bloques": mejores_rf_por_bloque,
            "Pesos_Decimales_100": pesos_decimales,
        }

    fila_cetes = {"Empresa / Activo": "CETES (28 días)"}
    for periodo in PERIODOS_ORDEN:
        fila_cetes[f"Peso {periodo} (%)"] = round(ponderacion_cetes * 100.0, 2)

    df_pesos = pd.concat([df_pesos, pd.DataFrame([fila_cetes])], ignore_index=True)

    df_metricas = pd.DataFrame(
        {
            "Periodo Histórico": list(resumen_periodos.keys()),
            "Máximo Sharpe Total": [round(v["Max_Sharpe_Global"], 4) for v in resumen_periodos.values()],
            "Tasa CETES Usada (%)": [round(v["Tasa_CETES_Optima (%)"], 2) for v in resumen_periodos.values()],
            "Rend. Esperado (%)": [round(v["Rendimiento_Comb (%)"], 2) for v in resumen_periodos.values()],
            "Volatilidad (%)": [round(v["Volatilidad_Comb (%)"], 2) for v in resumen_periodos.values()],
        }
    )

    return df_pesos, df_metricas, resumen_periodos


def crear_figura_animacion_monte_carlo(periodo_anim: str, datos_periodo_macro: dict, num_bloques: int) -> go.Figure:
    maximos_locales_sharpe = datos_periodo_macro["Vector_Maximos"]
    mejores_vol_por_bloque = datos_periodo_macro["Volatilidad_Bloques"]
    mejores_rend_por_bloque = datos_periodo_macro["Rendimiento_Bloques"]
    mejores_rf_por_bloque = datos_periodo_macro["Rf_Bloques"]

    paso = max(1, num_bloques // 4)
    pasos_frames = list(range(paso, num_bloques + 1, paso))
    if pasos_frames[-1] != num_bloques:
        pasos_frames.append(num_bloques)

    x_init = mejores_vol_por_bloque[:paso] * 100
    y_init = mejores_rend_por_bloque[:paso] * 100
    s_init = maximos_locales_sharpe[:paso]
    rf_init = mejores_rf_por_bloque[:paso] * 100

    idx_max_init = int(np.argmax(s_init))
    x_max_init = float(x_init[idx_max_init])
    y_max_init = float(y_init[idx_max_init])
    rf_max_init = float(rf_init[idx_max_init])
    s_max_init = float(s_init[idx_max_init])

    x_cml_init = [0.0, x_max_init, x_max_init * 1.25]
    y_cml_init = [rf_max_init, y_max_init, rf_max_init + (s_max_init * x_max_init * 1.25)]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_init,
                y=y_init,
                mode="markers",
                marker=dict(
                    size=5,
                    color=s_init,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio"),
                    opacity=0.65,
                ),
                text=[
                    f"Rendimiento: {r:.2f}%<br>Volatilidad: {v:.2f}%<br>Sharpe: {s:.3f}<br>CETES: {rf:.2f}%"
                    for r, v, s, rf in zip(y_init, x_init, s_init, rf_init)
                ],
                hoverinfo="text",
                name="Óptimos Locales",
            ),
            go.Scatter(
                x=[0.0],
                y=[rf_max_init],
                mode="markers+text",
                marker=dict(size=10, color="green", symbol="square"),
                text=[f"CETES: {rf_max_init:.2f}%"],
                textposition="top right",
                hoverinfo="text",
                name="Tasa Libre de Riesgo (CETES)",
            ),
            go.Scatter(
                x=[x_max_init],
                y=[y_max_init],
                mode="markers",
                marker=dict(size=14, color="red", symbol="star", line=dict(color="black", width=1)),
                text=[
                    f"<b>MÁXIMO SHARPE PROGRESIVO</b><br>Rendimiento: {y_max_init:.2f}%<br>"
                    f"Volatilidad: {x_max_init:.2f}%<br>Sharpe: {s_max_init:.4f}<br>CETES: {rf_max_init:.2f}%"
                ],
                hoverinfo="text",
                name="Portafolio Óptimo",
            ),
            go.Scatter(
                x=x_cml_init,
                y=y_cml_init,
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                text=[f"Línea Tangente (CML)<br>Sharpe Max: {s_max_init:.4f}"] * 3,
                hoverinfo="text",
                name="Línea Tangente (CML)",
            ),
        ],
        layout=go.Layout(
            title=f"<b>Construcción Dinámica Monte Carlo Alineada ({periodo_anim})</b>",
            xaxis=dict(
                title="Volatilidad / Riesgo (%)",
                range=[-0.5, float(np.max(mejores_vol_por_bloque * 100)) + 2.0],
            ),
            yaxis=dict(
                title="Rendimiento Esperado (%)",
                range=[
                    float(np.min(mejores_rend_por_bloque * 100)) - 2.0,
                    float(np.max(mejores_rend_por_bloque * 100)) + 3.0,
                ],
            ),
            template="plotly_white",
            height=580,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.05,
                    y=1.15,
                    buttons=[
                        dict(
                            label="▶ Reproducir",
                            method="animate",
                            args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)],
                        ),
                        dict(
                            label="⏸ Pausar",
                            method="animate",
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                        ),
                    ],
                )
            ],
        ),
    )

    frames = []
    for k in pasos_frames:
        x_k = mejores_vol_por_bloque[:k] * 100
        y_k = mejores_rend_por_bloque[:k] * 100
        s_k = maximos_locales_sharpe[:k]
        rf_k = mejores_rf_por_bloque[:k] * 100

        idx_max_k = int(np.argmax(s_k))
        x_max_k = float(x_k[idx_max_k])
        y_max_k = float(y_k[idx_max_k])
        rf_max_k = float(rf_k[idx_max_k])
        s_max_k = float(s_k[idx_max_k])

        x_cml_k = [0.0, x_max_k, x_max_k * 1.25]
        y_cml_k = [rf_max_k, y_max_k, rf_max_k + (s_max_k * x_max_k * 1.25)]

        hover_nube = [
            f"Rendimiento: {r:.2f}%<br>Volatilidad: {v:.2f}%<br>Sharpe: {s:.3f}<br>CETES: {rf:.2f}%"
            for r, v, s, rf in zip(y_k, x_k, s_k, rf_k)
        ]
        hover_cml = [
            f"Línea Tangente (CML)<br>Sharpe Max: {s_max_k:.4f}<br>CETES Usado: {rf_max_k:.2f}%"
        ] * 3

        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=x_k, y=y_k, marker=dict(color=s_k), text=hover_nube),
                    go.Scatter(x=[0.0], y=[rf_max_k], text=[f"CETES: {rf_max_k:.2f}%"]),
                    go.Scatter(
                        x=[x_max_k],
                        y=[y_max_k],
                        text=[
                            f"<b>MÁXIMO SHARPE PROGRESIVO</b><br>Bloques evaluados: {k:,}<br>"
                            f"Rendimiento: {y_max_k:.2f}%<br>Volatilidad: {x_max_k:.2f}%<br>"
                            f"Sharpe: {s_max_k:.4f}<br>CETES: {rf_max_k:.2f}%"
                        ],
                    ),
                    go.Scatter(x=x_cml_k, y=y_cml_k, text=hover_cml),
                ],
                name=f"frame_{periodo_anim}_{k}",
            )
        )

    fig.frames = frames
    return fig


# ==========================================
# 3. BARRA LATERAL (PARÁMETROS)
# ==========================================
st.title("🇲🇽 Modelación y Optimización de Portafolio Mexicano")
st.caption("Análisis histórico (Base 100 y Covarianza) + Macro-simulación Monte Carlo con CETES")

with st.sidebar:
    st.header("Parámetros Generales")
    empresas_sel = st.multiselect(
        "Empresas (Tickers)",
        options=EMPRESAS_DEFAULT,
        default=EMPRESAS_DEFAULT,
    )
    if len(empresas_sel) < 2:
        st.error("Selecciona al menos 2 empresas para calcular la matriz de covarianza.")
        st.stop()

    fecha_fin_input = st.date_input("Fecha de corte (fin)", value=pd.to_datetime("2026-09-18").date())
    fecha_fin_dt = pd.to_datetime(fecha_fin_input)

    st.divider()
    st.header("Macro-Simulación Monte Carlo")
    ponderacion_acciones = st.slider(
        "Ponderación Renta Variable (%)",
        min_value=10,
        max_value=100,
        value=70,
        step=5,
    ) / 100.0
    st.caption(f"Ponderación CETES: **{(1.0 - ponderacion_acciones) * 100:.0f}%**")

    num_bloques = st.number_input("Número de bloques", min_value=50, max_value=2000, value=1000, step=50)
    sims_por_bloque = st.number_input(
        "Simulaciones por bloque", min_value=500, max_value=15000, value=5000, step=500
    )
    semilla_aleatoria = st.number_input("Semilla aleatoria", min_value=0, max_value=999999, value=42, step=1)

empresas_tuple = tuple(empresas_sel)
fecha_fin_str = fecha_fin_dt.strftime("%Y-%m-%d")
periodos = obtener_fechas_periodos(fecha_fin_dt)

# ==========================================
# 4. PESTAÑAS PRINCIPALES
# ==========================================
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Análisis Histórico y JSON",
        "🎯 Macro-Simulación y Pesos",
        "🎬 Animación Monte Carlo (CML)",
    ]
)

# ------------------------------------------
# TAB 1: ANÁLISIS HISTÓRICO POR PERIODO
# ------------------------------------------
with tab1:
    payload_portafolio = {}
    periodo_tab = st.selectbox("Selecciona el periodo a inspeccionar", PERIODOS_ORDEN, key="sel_periodo_hist")

    # Construir JSON completo de los 3 periodos para descarga
    for nombre_p in PERIODOS_ORDEN:
        f_ini = periodos[nombre_p]
        d_p = descargar_precios(empresas_tuple, f_ini.strftime("%Y-%m-%d"), fecha_fin_str)
        r_log_p = np.log(d_p / d_p.shift(1)).dropna()
        cov_p = r_log_p.cov() * 252
        rend_p = r_log_p.mean() * 252
        payload_portafolio[nombre_p.upper()] = construir_bloque_periodo(d_p, cov_p, rend_p)

    st.download_button(
        label="⬇️ Descargar portafolio_mexicano.json",
        data=json.dumps(payload_portafolio, ensure_ascii=False, indent=2),
        file_name="portafolio_mexicano.json",
        mime="application/json",
    )

    fecha_inicio_sel = periodos[periodo_tab]
    st.subheader(f"Periodo: {periodo_tab} ({fecha_inicio_sel.strftime('%Y-%m-%d')} a {fecha_fin_str})")

    datos = descargar_precios(empresas_tuple, fecha_inicio_sel.strftime("%Y-%m-%d"), fecha_fin_str)
    rendimientos_log = np.log(datos / datos.shift(1)).dropna()
    cov_anualizada = rendimientos_log.cov() * 252

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Primeras filas de precios de cierre**")
        st.dataframe(datos.head(), use_container_width=True)
    with col_b:
        st.markdown("**Resumen estadístico de precios**")
        st.dataframe(datos.describe(), use_container_width=True)

    # Gráfico 1: Base 100
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for columna in datos.columns:
        serie_norm = (datos[columna] / datos[columna].dropna().iloc[0]) * 100
        ax1.plot(datos.index, serie_norm, label=columna, linewidth=1.6)
    ax1.set_title(f"Rendimiento Relativo (Base 100) - {periodo_tab}", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Valor Normalizado (Base 100)")
    ax1.legend(title="Empresas", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # Matriz de covarianza y heatmap
    st.markdown(f"**Matriz de Covarianza Anualizada ({periodo_tab})**")
    st.dataframe(cov_anualizada, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.heatmap(cov_anualizada, annot=True, cmap="YlGnBu", fmt=".4f", ax=ax2)
    ax2.set_title(f"Matriz de Covarianza Anualizada ({periodo_tab})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ------------------------------------------
# TAB 2: MACRO-SIMULACIÓN Y PESOS ÓPTIMOS
# ------------------------------------------
with tab2:
    df_pesos_macro, df_metricas_macro, resumen_macro_periodos = ejecutar_macro_simulacion(
        empresas=empresas_tuple,
        fecha_fin_str=fecha_fin_str,
        num_bloques=int(num_bloques),
        sims_por_bloque=int(sims_por_bloque),
        ponderacion_acciones=ponderacion_acciones,
        semilla=int(semilla_aleatoria),
    )

    st.subheader("Distribución Final de Pesos Combinados")
    st.dataframe(df_pesos_macro, use_container_width=True)

    st.subheader("Resumen de Métricas del Máximo Global Absoluto")
    st.dataframe(df_metricas_macro, use_container_width=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_bar_rend, ax_r = plt.subplots(figsize=(8, 5))
        bars = sns.barplot(
            x="Periodo Histórico",
            y="Rend. Esperado (%)",
            data=df_metricas_macro,
            palette="viridis",
            hue="Periodo Histórico",
            legend=False,
            ax=ax_r,
        )
        ax_r.set_title("Comparativa de Rendimientos Esperados por Periodo", fontsize=12, fontweight="bold")
        ax_r.set_ylabel("Rendimiento Esperado Anual (%)")
        ax_r.set_ylim(0, max(df_metricas_macro["Rend. Esperado (%)"]) + 5)
        for p in bars.patches:
            h = p.get_height()
            ax_r.annotate(
                f"{h:.2f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                xytext=(0, 4),
                textcoords="offset points",
            )
        plt.tight_layout()
        st.pyplot(fig_bar_rend)
        plt.close(fig_bar_rend)

    with col_g2:
        df_melted = pd.melt(
            df_pesos_macro,
            id_vars=["Empresa / Activo"],
            value_vars=[f"Peso {p} (%)" for p in PERIODOS_ORDEN],
            var_name="Periodo",
            value_name="Ponderación (%)",
        )
        df_melted["Periodo"] = df_melted["Periodo"].str.replace("Peso ", "", regex=False).str.replace(
            " (%)", "", regex=False
        )

        fig_bar_pesos, ax_w = plt.subplots(figsize=(9, 5))
        sns.barplot(
            x="Empresa / Activo",
            y="Ponderación (%)",
            hue="Periodo",
            data=df_melted,
            palette="Set2",
            ax=ax_w,
        )
        ax_w.set_title("Asignación de Activos (Pesos %) por Periodo", fontsize=12, fontweight="bold")
        ax_w.set_ylabel("Ponderación en el Portafolio (%)")
        ax_w.set_ylim(0, max(df_melted["Ponderación (%)"]) + 8)
        for p in ax_w.patches:
            h = p.get_height()
            if h > 0:
                ax_w.annotate(
                    f"{h:.1f}%",
                    (p.get_x() + p.get_width() / 2.0, h),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    xytext=(0, 2),
                    textcoords="offset points",
                )
        plt.tight_layout()
        st.pyplot(fig_bar_pesos)
        plt.close(fig_bar_pesos)

    st.subheader("Distribución en Gráficos de Pastel por Periodo")
    fig_pie, axs_pie = plt.subplots(1, 3, figsize=(16, 5))
    colores_pie = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0", "#ffb3e6"]

    for idx, periodo in enumerate(PERIODOS_ORDEN):
        col_name = f"Peso {periodo} (%)"
        datos_pastel = df_pesos_macro[df_pesos_macro[col_name] > 0]
        axs_pie[idx].pie(
            datos_pastel[col_name],
            labels=datos_pastel["Empresa / Activo"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colores_pie[: len(datos_pastel)],
            textprops={"fontsize": 9, "weight": "bold"},
            wedgeprops={"edgecolor": "w", "linewidth": 1.2},
        )
        axs_pie[idx].set_title(f"Asignación MSR - {periodo}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig_pie)
    plt.close(fig_pie)

# ------------------------------------------
# TAB 3: ANIMACIÓN MONTE CARLO (CML)
# ------------------------------------------
with tab3:
    _, _, resumen_anim = ejecutar_macro_simulacion(
        empresas=empresas_tuple,
        fecha_fin_str=fecha_fin_str,
        num_bloques=int(num_bloques),
        sims_por_bloque=int(sims_por_bloque),
        ponderacion_acciones=ponderacion_acciones,
        semilla=int(semilla_aleatoria),
    )

    periodo_anim = st.selectbox("Selecciona el periodo para la animación CML", PERIODOS_ORDEN, key="sel_periodo_anim")
    fig_anim = crear_figura_animacion_monte_carlo(
        periodo_anim=periodo_anim,
        datos_periodo_macro=resumen_anim[periodo_anim],
        num_bloques=int(num_bloques),
    )
    st.plotly_chart(fig_anim, use_container_width=True)
