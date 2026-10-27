import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Futuros USD/MXN", page_icon="📈", layout="wide"
)

# Título Principal
st.title("📊 Simulador Financiero de Futuros USD/MXN")
st.write(
    "Selecciona el modelo de simulación diaria detallada para visualizar la trayectoria de precios, valuación y requerimientos de margen."
)

# --- 1. SELECCIÓN DE MODELO EN LA BARRA LATERAL ---
st.sidebar.header("🛠️ Seleccionar Modelo")
modelo_seleccionado = st.sidebar.selectbox(
    "Elige el enfoque de simulación:",
    [
        "1. Simulación Diaria - Modelo Log-logístico",
        "2. Simulación Diaria - Método de Fisher (MBG)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros de Simulación")

precio_futuro_inicial = st.sidebar.number_input(
    "Precio Futuro Inicial ($)", min_value=15.0, max_value=30.0, value=19.0, step=0.1
)

capital_inicial = st.sidebar.number_input(
    "Capital Inicial ($)", min_value=10000.0, max_value=100000.0, value=15000.0, step=1000.0
)

tamano_contrato = st.sidebar.number_input(
    "Tamaño del Contrato", min_value=1000, max_value=100000, value=10000, step=1000
)


# --- 2. FUNCIONES DE CÁLCULO DE LAS SIMULACIONES ---

def simular_log_logistico(precio_inicial, cap_inicial, t_contrato):
    B3 = 28.52977
    B4 = 18.32123

    fechas = pd.date_range(start="2026-10-15", end="2026-12-17", freq="D")
    n = len(fechas)

    np.random.seed(42)  # Semilla fija para reproducibilidad visual
    U = np.random.uniform(0.0001, 0.9999, n)
    precios = B4 * (U / (1 - U)) ** (1 / B3)
    precios[0] = 18.44

    variacion_diaria = pd.Series(precios).pct_change()
    variacion_diaria.iloc[0] = 0.0

    precio_futuro = np.zeros(n)
    precio_futuro[0] = precio_inicial
    for i in range(1, n):
        precio_futuro[i] = precio_futuro[i - 1] * (1 + variacion_diaria.iloc[i])

    valuacion_contrato = precio_futuro * t_contrato
    ganancia_perdida_diaria = pd.Series(valuacion_contrato).diff()
    ganancia_perdida_diaria.iloc[0] = 0.0

    comisiones = np.zeros(n)
    comisiones[0] = 50.0
    comisiones[-1] = 50.0

    margen = np.zeros(n)
    depositos_llamadas = np.zeros(n)
    capital_neto_inicial = cap_inicial - comisiones[0]

    for i in range(n):
        margen_prev = capital_neto_inicial if i == 0 else margen[i - 1]
        gp = ganancia_perdida_diaria.iloc[i]
        com = comisiones[i] if i == n - 1 else 0.0

        m_temp = margen_prev + gp - com

        dep = 0.0
        if m_temp <= 0:
            dep = cap_inicial + abs(m_temp)
            m_temp = m_temp + dep

        margen[i] = m_temp
        depositos_llamadas[i] = dep

    df = pd.DataFrame({
        "Fecha": fechas.strftime("%d/%m/%Y"),
        "Precio_Simulado": precios,
        "Variacion_Diaria": variacion_diaria,
        "Precio_Futuro": precio_futuro,
        "Valuacion_Contrato": valuacion_contrato,
        "Ganancia_Perdida_Diaria": ganancia_perdida_diaria,
        "Comisiones": comisiones,
        "Depositos_por_llamadas": depositos_llamadas,
        "Margen": margen,
    })
    return df, fechas


def simular_fisher(precio_inicial, cap_inicial, t_contrato):
    B5_desv = 17.69391793  # Desviación estándar anual (%)
    B10 = 0.01736111       # Cetes
    B11 = 0.01119444       # T-Bills

    fechas = pd.date_range(start="2026-10-15", end="2026-12-17", freq="D")
    n = len(fechas)

    sigma = B5_desv / 100.0
    dt = 1 / 252.0

    np.random.seed(42)  # Semilla fija para estabilidad de la curva
    Z = np.random.standard_normal(n)

    precios = np.zeros(n)
    precios[0] = 18.44

    for i in range(1, n):
        factor_exponente = (B10 - B11 - 0.5 * (sigma**2)) * dt + (sigma / np.sqrt(252)) * Z[i]
        precios[i] = precios[i - 1] * np.exp(factor_exponente)

    variacion_diaria = pd.Series(precios).pct_change()
    variacion_diaria.iloc[0] = 0.0

    precio_futuro = np.zeros(n)
    precio_futuro[0] = precio_inicial
    for i in range(1, n):
        precio_futuro[i] = precio_futuro[i - 1] * (1 + variacion_diaria.iloc[i])

    valuacion_contrato = precio_futuro * t_contrato
    ganancia_perdida_diaria = pd.Series(valuacion_contrato).diff()
    ganancia_perdida_diaria.iloc[0] = 0.0

    comisiones = np.zeros(n)
    comisiones[0] = 50.0
    comisiones[-1] = 50.0

    margen = np.zeros(n)
    depositos_llamadas = np.zeros(n)
    capital_neto_inicial = cap_inicial - comisiones[0]

    for i in range(n):
        margen_prev = capital_neto_inicial if i == 0 else margen[i - 1]
        gp = ganancia_perdida_diaria.iloc[i]
        com = comisiones[i] if i == n - 1 else 0.0

        m_temp = margen_prev + gp - com

        dep = 0.0
        if m_temp <= 0:
            dep = cap_inicial + abs(m_temp)
            m_temp = m_temp + dep

        margen[i] = m_temp
        depositos_llamadas[i] = dep

    df = pd.DataFrame({
        "Fecha": fechas.strftime("%d/%m/%Y"),
        "Precio_Simulado": precios,
        "Variacion_Diaria": variacion_diaria,
        "Precio_Futuro": precio_futuro,
        "Valuacion_Contrato": valuacion_contrato,
        "Ganancia_Perdida_Diaria": ganancia_perdida_diaria,
        "Comisiones": comisiones,
        "Depositos_por_llamadas": depositos_llamadas,
        "Margen": margen,
    })
    return df, fechas


def generar_df_resumen(df, cap_inicial, t_contrato, p_inicial):
    idx_max_dep = df["Depositos_por_llamadas"].idxmax()
    max_deposito = df.loc[idx_max_dep, "Depositos_por_llamadas"]

    if max_deposito > 0:
        fecha_max_dep = df.loc[idx_max_dep, "Fecha"]
        precio_dolar_max_dep = df.loc[idx_max_dep, "Precio_Futuro"]
    else:
        fecha_max_dep = "N/A (Sin llamadas)"
        precio_dolar_max_dep = "N/A"

    precio_final = df["Precio_Futuro"].iloc[-1]
    ganancia_bruta_mercado = (precio_final - p_inicial) * t_contrato
    total_llamadas = df["Depositos_por_llamadas"].sum()
    capital_inyectado_total = cap_inicial + total_llamadas
    total_comisiones = df["Comisiones"].sum()
    efectivo_retirable = df["Margen"].iloc[-1]

    df_resumen = pd.DataFrame({
        "Métrica / Concepto": [
            "Máximo depósito por llamada",
            "Fecha del máximo depósito",
            "Precio del dólar ese día",
            "Precio del dólar al final",
            "Inversión Inicial",
            "Ganancia Bruta del Mercado",
            "Total depósitos por llamadas",
            "Capital Inyectado Total",
            "Comisiones Totales",
            "Efectivo Total Retirable (Margen Final)",
        ],
        "Valor": [
            f"${max_deposito:,.2f}",
            str(fecha_max_dep),
            f"${precio_dolar_max_dep:,.4f}" if isinstance(precio_dolar_max_dep, float) else precio_dolar_max_dep,
            f"${precio_final:,.4f}",
            f"${cap_inicial:,.2f}",
            f"${ganancia_bruta_mercado:,.2f}",
            f"${total_llamadas:,.2f}",
            f"${capital_inyectado_total:,.2f}",
            f"${total_comisiones:,.2f}",
            f"${efectivo_retirable:,.2f}",
        ],
    })
    return df_resumen


# --- 3. EJECUCIÓN Y RENDERIZADO ---

if "Log-logístico" in modelo_seleccionado:
    st.subheader("📌 Modelo Log-logístico — Simulación Diaria")
    df, fechas = simular_log_logistico(precio_futuro_inicial, capital_inicial, tamano_contrato)
    color_grafico = "#2980b9"
else:
    st.subheader("📌 Método de Fisher (MBG) — Simulación Diaria")
    df, fechas = simular_fisher(precio_futuro_inicial, capital_inicial, tamano_contrato)
    color_grafico = "#8e44ad"

df_resumen = generar_df_resumen(df, capital_inicial, tamano_contrato, precio_futuro_inicial)

# Tarjetas de Métricas Rápidas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio Dólar Final", df_resumen.loc[3, "Valor"])
col2.metric("Ganancia Bruta Mercado", df_resumen.loc[5, "Valor"])
col3.metric("Capital Inyectado Total", df_resumen.loc[7, "Valor"])
col4.metric("Efectivo Retirable Final", df_resumen.loc[9, "Valor"])

st.markdown("---")

# Layout de 2 columnas para Tablas
col_izq, col_der = st.columns([3, 2])

with col_izq:
    st.markdown("### 📋 Tabla de Simulación Diaria")
    st.dataframe(df, use_container_width=True, height=450)

with col_der:
    st.markdown("### 📊 Resumen de Inversión y Resultados")
    st.table(df_resumen)

# Gráfico Interactivo de Trayectoria
st.markdown("### 📈 Gráfica de Trayectoria del Precio Futuro y Margen")
fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.set_title("Evolución Diaria del Precio Futuro (USD/MXN) vs. Margen Neto", fontsize=13, fontweight="bold", pad=15)
ax1.plot(fechas, df["Precio_Futuro"], color=color_grafico, linewidth=2.5, label="Precio Futuro ($)")
ax1.set_xlabel("Fecha", fontsize=11)
ax1.set_ylabel("Precio Dólar ($)", color=color_grafico, fontsize=11)
ax1.tick_params(axis="y", labelcolor=color_grafico)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(fechas, df["Margen"], color="#27ae60", linewidth=1.8, linestyle="--", label="Margen Neto ($)")
ax2.set_ylabel("Margen Neto ($)", color="#27ae60", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#27ae60")

fig.tight_layout()
st.pyplot(fig)
