import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Futuros USD/MXN", page_icon="📈", layout="wide"
)

# Título Principal
st.title("📊 Simulador Financiero Completo de Futuros USD/MXN")
st.write(
    "Selecciona el modelo para analizar tanto la **trayectoria diaria detallada** como el **análisis de riesgo por Monte Carlo**."
)

# --- 1. SELECCIÓN DE MODELO EN LA BARRA LATERAL ---
st.sidebar.header("🛠️ Seleccionar Modelo")
modelo_seleccionado = st.sidebar.selectbox(
    "Elige el enfoque de simulación:",
    [
        "1. Modelo Log-logístico (Distribución Base)",
        "2. Método de Fisher (Movimiento Geométrico Browniano)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros de Simulación")

precio_futuro_inicial = st.sidebar.slider(
    "Precio Futuro Inicial ($)", min_value=15.0, max_value=25.0, value=19.0, step=0.1
)

margen_llamada = st.sidebar.slider(
    "Margen de Llamada ($)",
    min_value=5000.0,
    max_value=15000.0,
    value=12000.0,
    step=500.0,
)

capital_inicial = st.sidebar.number_input(
    "Capital Inicial ($)", min_value=10000.0, max_value=100000.0, value=15000.0, step=1000.0
)

tamano_contrato = st.sidebar.number_input(
    "Tamaño del Contrato", min_value=1000, max_value=100000, value=10000, step=1000
)

st.sidebar.markdown("---")
st.sidebar.header("🎲 Parámetros de Monte Carlo")

alpha_seleccionado = st.sidebar.slider(
    "Alpha (p) para VaR", min_value=0.01, max_value=0.90, value=0.05, step=0.01
)

num_simulaciones = st.sidebar.slider(
    "Número de Simulaciones",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
)


# ==============================================================================
# --- 2. FUNCIONES DE SIMULACIÓN DIARIA DETALLADA ---
# ==============================================================================

def simular_diario_log_logistico(precio_inicial, cap_inicial, t_contrato):
    B3 = 28.52977
    B4 = 18.32123

    fechas = pd.date_range(start="2026-10-15", end="2026-12-17", freq="D")
    n = len(fechas)

    np.random.seed(42)  # Para estabilidad en la curva diaria
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


def simular_diario_fisher(precio_inicial, cap_inicial, t_contrato):
    B5_desv = 17.69391793
    B10 = 0.01736111
    B11 = 0.01119444

    fechas = pd.date_range(start="2026-10-15", end="2026-12-17", freq="D")
    n = len(fechas)

    sigma = B5_desv / 100.0
    dt = 1 / 252.0

    np.random.seed(42)  # Para estabilidad en la curva diaria
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


# ==============================================================================
# --- 3. FUNCIONES DE SIMULACIÓN DE MONTE CARLO ---
# ==============================================================================

def ejecutar_monte_carlo(tipo_modelo, precio_inicial, margen_llamada_val, cap_inicial, t_contrato, alpha, num_sims):
    fechas = pd.date_range(start="2026-10-15", end="2026-12-17", freq="D")
    n = len(fechas)

    B3 = 28.52977
    B4 = 18.32123

    B5_desv = 17.69391793
    B10 = 0.01736111
    B11 = 0.01119444
    sigma = B5_desv / 100.0
    dt = 1 / 252.0

    resultados_simulaciones = []

    for sim in range(num_sims):
        if tipo_modelo == "log_logistico":
            U = np.random.uniform(0.0001, 0.9999, n)
            precios = B4 * (U / (1 - U)) ** (1 / B3)
            precios[0] = 18.44
        else:
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
        gp_diaria = np.diff(valuacion_contrato, prepend=valuacion_contrato[0])
        gp_diaria[0] = 0.0

        comisiones = np.zeros(n)
        comisiones[0] = 50.0
        comisiones[-1] = 50.0
        total_comisiones = np.sum(comisiones)

        margen = np.zeros(n)
        depositos_llamadas = np.zeros(n)
        capital_neto_inicial = cap_inicial - comisiones[0]

        for i in range(n):
            margen_prev = capital_neto_inicial if i == 0 else margen[i - 1]
            gp = gp_diaria[i]
            com = comisiones[i] if i == n - 1 else 0.0

            m_temp = margen_prev + gp - com

            dep = 0.0
            if m_temp < margen_llamada_val:
                dep = cap_inicial - m_temp
                m_temp = cap_inicial

            margen[i] = m_temp
            depositos_llamadas[i] = dep

        precio_final = precio_futuro[-1]
        ganancia_bruta_mercado = (precio_final - precio_inicial) * t_contrato
        total_depositos_llamadas = np.sum(depositos_llamadas)
        total_capital_inyectado = cap_inicial + total_depositos_llamadas
        efectivo_retirable = margen[-1]

        max_deposito = np.max(depositos_llamadas)
        if max_deposito > 0:
            idx_max = np.argmax(depositos_llamadas)
            dia_llamada_max = fechas[idx_max].strftime("%d/%m/%Y")
            precio_dolar_llamada_max = precio_futuro[idx_max]
        else:
            dia_llamada_max = "Sin llamadas"
            precio_dolar_llamada_max = 0.0

        resultados_simulaciones.append({
            "Simulación (#)": sim + 1,
            "Inversión inicial": cap_inicial,
            "Ganancia bruta del mercado": ganancia_bruta_mercado,
            "Depósitos por llamadas": total_depositos_llamadas,
            "Total de capital inyectado": total_capital_inyectado,
            "Comisiones totales": total_comisiones,
            "Efectivo total retirable": efectivo_retirable,
            "Llamada máxima": max_deposito,
            "Día de llamada máxima": dia_llamada_max,
            "Precio dólar (llamada máx.)": precio_dolar_llamada_max,
            "Precio dólar (final)": precio_final,
        })

    df_excel = pd.DataFrame(resultados_simulaciones)

    columnas_numericas = [
        "Ganancia bruta del mercado",
        "Depósitos por llamadas",
        "Total de capital inyectado",
        "Efectivo total retirable",
        "Llamada máxima",
        "Precio dólar (llamada máx.)",
        "Precio dólar (final)",
    ]

    dic_estadisticos = []
    for col in columnas_numericas:
        serie = df_excel[col]
        moda_res = stats.mode(np.round(serie, 2), keepdims=True)
        moda_val = moda_res.mode[0] if len(moda_res.mode) > 0 else 0.0

        dic_estadisticos.append({
            "Métrica": col,
            "Mínimo": serie.min(),
            "Q1 (25%)": serie.quantile(0.25),
            "Mediana (50%)": serie.median(),
            "Promedio": serie.mean(),
            "Q3 (75%)": serie.quantile(0.75),
            "Máximo": serie.max(),
            "Moda": moda_val,
            "Desviación Estándar": serie.std(),
        })

    df_estadisticos = pd.DataFrame(dic_estadisticos)

    valor_inicial = 190000.0
    df_excel["Valor Total Final"] = valor_inicial + df_excel["Ganancia bruta del mercado"]
    p = alpha
    etiqueta = f"{int(p*100)}%"
    serie_valor = df_excel["Valor Total Final"]

    val_cutoff = serie_valor.quantile(p)
    val_var = valor_inicial - val_cutoff

    df_cut_var_valor = pd.DataFrame([{
        "Métrica": "Valor Total del Contrato",
        "Tipo Cutoff": val_cutoff,
        "Tipo VaR": val_var,
        "Nivel Alpha (%)": etiqueta,
    }])

    return df_excel, df_estadisticos, df_cut_var_valor, etiqueta


# ==============================================================================
# --- 4. RENDERIZADO GENERAL Y DESPLIEGUE CONTINUO ---
# ==============================================================================

if "Log-logístico" in modelo_seleccionado:
    st.header("📌 MODELO 1: LOG-LOGÍSTICO")
    df_diario, fechas = simular_diario_log_logistico(precio_futuro_inicial, capital_inicial, tamano_contrato)
    color_grafico = "#2980b9"
    tipo_mc = "log_logistico"
else:
    st.header("📌 MODELO 2: MÉTODO DE FISHER (MBG)")
    df_diario, fechas = simular_diario_fisher(precio_futuro_inicial, capital_inicial, tamano_contrato)
    color_grafico = "#8e44ad"
    tipo_mc = "fisher"

df_resumen = generar_df_resumen(df_diario, capital_inicial, tamano_contrato, precio_futuro_inicial)

# --- A. METRICAS RAPIDAS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio Dólar Final", df_resumen.loc[3, "Valor"])
col2.metric("Ganancia Bruta Mercado", df_resumen.loc[5, "Valor"])
col3.metric("Capital Inyectado Total", df_resumen.loc[7, "Valor"])
col4.metric("Efectivo Retirable Final", df_resumen.loc[9, "Valor"])

st.markdown("---")

# --- B. SIMULACION DIARIA Y RESUMEN ---
col_izq, col_der = st.columns([3, 2])

with col_izq:
    st.markdown("### 📋 Tabla de Simulación Diaria")
    st.dataframe(df_diario, use_container_width=True, height=420)

with col_der:
    st.markdown("### 📊 Resumen de Inversión y Resultados")
    st.table(df_resumen)

# --- C. GRAFICA DE TRAYECTORIA ---
st.markdown("### 📈 Gráfica de Trayectoria del Precio Futuro y Margen")
fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.set_title("Evolución Diaria del Precio Futuro (USD/MXN) vs. Margen Neto", fontsize=13, fontweight="bold", pad=15)
ax1.plot(fechas, df_diario["Precio_Futuro"], color=color_grafico, linewidth=2.5, label="Precio Futuro ($)")
ax1.set_xlabel("Fecha", fontsize=11)
ax1.set_ylabel("Precio Dólar ($)", color=color_grafico, fontsize=11)
ax1.tick_params(axis="y", labelcolor=color_grafico)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(fechas, df_diario["Margen"], color="#27ae60", linewidth=1.8, linestyle="--", label="Margen Neto ($)")
ax2.set_ylabel("Margen Neto ($)", color="#27ae60", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#27ae60")

fig.tight_layout()
st.pyplot(fig)

st.markdown("---")

# --- D. SIMULACION DE MONTE CARLO (JUSTO DEBAJO) ---
st.header("🎲 ANÁLISIS DE MONTE CARLO Y VAR (Múltiples Corridas)")

df_excel, df_estadisticos, df_cut_var_valor, etiqueta = ejecutar_monte_carlo(
    tipo_mc, precio_futuro_inicial, margen_llamada, capital_inicial, tamano_contrato, alpha_seleccionado, num_simulaciones
)

st.markdown("### 📋 Vista Previa de Simulaciones Monte Carlo (Primeras 10)")
st.dataframe(df_excel.head(10), use_container_width=True)

col_est, col_var = st.columns([3, 2])

with col_est:
    st.markdown("### 📊 Tabla de Estadísticos")
    st.dataframe(df_estadisticos, use_container_width=True)

with col_var:
    st.markdown(f"### 🛡️ Tabla de Cutoff y VaR (Alpha = {etiqueta} | Base: 190,000)")
    st.dataframe(df_cut_var_valor, use_container_width=True)

# --- E. HISTOGRAMAS 2X2 DE MONTE CARLO ---
st.markdown("### 📊 Histogramas Dinámicos de Distribución (Monte Carlo)")

fig_mc, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].hist(df_excel["Efectivo total retirable"], bins=50, color="#3498db", edgecolor="#2980b9", alpha=0.85)
axs[0, 0].set_title(f"Efectivo Total Retirable (Margen: ${margen_llamada:,.0f})", fontweight="bold")
axs[0, 0].set_xlabel("Monto ($)")
axs[0, 0].set_ylabel("Frecuencia")
axs[0, 0].grid(True, linestyle="--", alpha=0.5)

axs[0, 1].hist(df_excel["Ganancia bruta del mercado"], bins=50, color="#2ecc71", edgecolor="#27ae60", alpha=0.85)
axs[0, 1].set_title(f"Ganancia Bruta del Mercado (Futuro: {precio_futuro_inicial})", fontweight="bold")
axs[0, 1].set_xlabel("Monto ($)")
axs[0, 1].set_ylabel("Frecuencia")
axs[0, 1].grid(True, linestyle="--", alpha=0.5)

axs[1, 0].hist(df_excel["Total de capital inyectado"], bins=50, color="#e67e22", edgecolor="#d35400", alpha=0.85)
axs[1, 0].set_title("Capital Inyectado Total", fontweight="bold")
axs[1, 0].set_xlabel("Monto ($)")
axs[1, 0].set_ylabel("Frecuencia")
axs[1, 0].grid(True, linestyle="--", alpha=0.5)

axs[1, 1].hist(df_excel["Llamada máxima"], bins=50, color="#e74c3c", edgecolor="#c0392b", alpha=0.85)
axs[1, 1].set_title(f"Llamada Máxima (Margen: ${margen_llamada:,.0f})", fontweight="bold")
axs[1, 1].set_xlabel("Monto ($)")
axs[1, 1].set_ylabel("Frecuencia")
axs[1, 1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
st.pyplot(fig_mc)
