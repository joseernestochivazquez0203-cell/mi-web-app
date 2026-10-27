import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Futuros USD/MXN", page_icon="📈", layout="wide"
)

# Título principal de la aplicación web
st.title("📊 Simulador Financiero y de Monte Carlo - Futuros USD/MXN")
st.write(
    "Selecciona el tipo de análisis o modelo que deseas ejecutar y ajusta los parámetros en la barra lateral."
)

# --- 1. SELECCIÓN DE MODELO EN LA BARRA LATERAL ---
st.sidebar.header("🛠️ Panel de Control")
modelo_seleccionado = st.sidebar.selectbox(
    "Elige la vista o modelo:",
    [
        "Modelo 1: Simulación Diaria Detallada (Log-logística)",
        "Modelo 2: Monte Carlo - Log-logístico (Múltiples corridas)",
        "Modelo 3: Monte Carlo - Movimiento Geométrico Browniano (Fisher)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros Globales")

precio_futuro_inicial = st.sidebar.slider(
    "Futuro Inicial ($)", min_value=19.0, max_value=22.0, value=19.0, step=0.1
)

margen_llamada = st.sidebar.slider(
    "Margen de Llamada ($)",
    min_value=5000.0,
    max_value=15000.0,
    value=12000.0,
    step=500.0,
)

alpha_seleccionado = st.sidebar.slider(
    "Alpha (p) para VaR", min_value=0.01, max_value=0.90, value=0.05, step=0.01
)

num_simulaciones = st.sidebar.slider(
    "Número de Simulaciones (Monte Carlo)",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
)


# --- 2. FUNCIONES DE LOS MODELOS ---


def ejecutar_modelo_diario_detallado(precio_inicial, margen_llamada_val):
    # Parámetros
    B3 = 28.52977
    B4 = 18.32123
    tamano_contrato = 10000
    capital_inicial = 15000.0

    fechas = pd.date_range(start="2026-10-15", end="2026-12-17", freq="D")
    n = len(fechas)

    np.random.seed(42)  # Para estabilidad visual
    U = np.random.uniform(0.0001, 0.9999, n)
    precios = B4 * (U / (1 - U)) ** (1 / B3)
    precios[0] = 18.44

    variacion_diaria = pd.Series(precios).pct_change()
    variacion_diaria.iloc[0] = 0.0

    precio_futuro = np.zeros(n)
    precio_futuro[0] = precio_inicial
    for i in range(1, n):
        precio_futuro[i] = precio_futuro[i - 1] * (
            1 + variacion_diaria.iloc[i]
        )

    valuacion_contrato = precio_futuro * tamano_contrato
    ganancia_perdida_diaria = pd.Series(valuacion_contrato).diff()
    ganancia_perdida_diaria.iloc[0] = 0.0

    comisiones = np.zeros(n)
    comisiones[0] = 50.0
    comisiones[-1] = 50.0

    margen = np.zeros(n)
    depositos_llamadas = np.zeros(n)
    capital_neto_inicial = capital_inicial - comisiones[0]

    for i in range(n):
        margen_prev = capital_neto_inicial if i == 0 else margen[i - 1]
        gp = ganancia_perdida_diaria.iloc[i]
        com = comisiones[i] if i == n - 1 else 0.0

        m_temp = margen_prev + gp - com

        dep = 0.0
        if m_temp <= 0:
            dep = 15000.0 + abs(m_temp)
            m_temp = m_temp + dep

        margen[i] = m_temp
        depositos_llamadas[i] = dep

    df = pd.DataFrame(
        {
            "Fecha": fechas.strftime("%d/%m/%Y"),
            "Precio_Simulado": precios,
            "Variacion_Diaria": variacion_diaria,
            "Precio_Futuro": precio_futuro,
            "Valuacion_Contrato": valuacion_contrato,
            "Ganancia_Perdida_Diaria": ganancia_perdida_diaria,
            "Comisiones": comisiones,
            "Depositos_por_llamadas": depositos_llamadas,
            "Margen": margen,
        }
    )
    return df, fechas, precio_futuro, ganancia_perdida_diaria


def ejecutar_monte_carlo_modelo1(precio_inicial, margen_llamada_val, alpha, num_sims):
    B3 = 28.52977
    B4 = 18.32123
    tamano_contrato = 10000
    capital_inicial = 15000.0

    fechas = pd.date_range(start="2026-10-15", end="2026-12-17", freq="D")
    n = len(fechas)
    resultados_simulaciones = []

    for sim in range(num_sims):
        U = np.random.uniform(0.0001, 0.9999, n)
        precios = B4 * (U / (1 - U)) ** (1 / B3)
        precios[0] = 18.44

        variacion_diaria = pd.Series(precios).pct_change()
        variacion_diaria.iloc[0] = 0.0

        precio_futuro = np.zeros(n)
        precio_futuro[0] = precio_inicial
        for i in range(1, n):
            precio_futuro[i] = precio_futuro[i - 1] * (
                1 + variacion_diaria.iloc[i]
            )

        valuacion_contrato = precio_futuro * tamano_contrato
        gp_diaria = np.diff(valuacion_contrato, prepend=valuacion_contrato[0])
        gp_diaria[0] = 0.0

        comisiones = np.zeros(n)
        comisiones[0] = 50.0
        comisiones[-1] = 50.0
        total_comisiones = np.sum(comisiones)

        margen = np.zeros(n)
        depositos_llamadas = np.zeros(n)
        capital_neto_inicial = capital_inicial - comisiones[0]

        for i in range(n):
            margen_prev = (
                capital_neto_inicial if i == 0 else margen[i - 1]
            )  # pylint: disable=cell-var-from-loop
            gp = gp_diaria[i]
            com = comisiones[i] if i == n - 1 else 0.0

            m_temp = margen_prev + gp - com

            dep = 0.0
            if m_temp < margen_llamada_val:
                dep = 15000.0 - m_temp
                m_temp = 15000.0

            margen[i] = m_temp
            depositos_llamadas[i] = dep

        precio_final = precio_futuro[-1]
        ganancia_bruta_mercado = (
            precio_final - precio_inicial
        ) * tamano_contrato
        total_depositos_llamadas = np.sum(depositos_llamadas)
        total_capital_inyectado = capital_inicial + total_depositos_llamadas
        efectivo_retirable = margen[-1]

        max_deposito = np.max(depositos_llamadas)
        if max_deposito > 0:
            idx_max = np.argmax(depositos_llamadas)
            dia_llamada_max = fechas[idx_max].strftime("%d/%m/%Y")
            precio_dolar_llamada_max = precio_futuro[idx_max]
        else:
            dia_llamada_max = "Sin llamadas"
            precio_dolar_llamada_max = 0.0

        resultados_simulaciones.append(
            {
                "Simulación (#)": sim + 1,
                "Inversión inicial": capital_inicial,
                "Ganancia bruta del mercado": ganancia_bruta_mercado,
                "Depósitos por llamadas": total_depositos_llamadas,
                "Total de capital inyectado": total_capital_inyectado,
                "Comisiones totales": total_comisiones,
                "Efectivo total retirable": efectivo_retirable,
                "Llamada máxima": max_deposito,
                "Día de llamada máxima": dia_llamada_max,
                "Precio dólar (llamada máx.)": precio_dolar_llamada_max,
                "Precio dólar (final)": precio_final,
            }
        )

    return procesar_resultados(
        resultados_simulaciones, alpha, capital_inicial
    )


def ejecutar_monte_carlo_modelo2(
    precio_inicial, margen_llamada_val, alpha, num_sims
):
    B10 = 0.018
    B11 = 0.01612
    B5_desv = 24.69
    sigma = B5_desv / 100.0
    dt = 1 / 252.0
    tamano_contrato = 10000
    capital_inicial = 15000.0

    fechas = pd.date_range(start="2026-10-15", end="2026-12-17", freq="D")
    n = len(fechas)
    resultados_simulaciones = []

    for sim in range(num_sims):
        Z = np.random.standard_normal(n)
        precios = np.zeros(n)
        precios[0] = 18.44

        for i in range(1, n):
            factor_exponente = (
                B10 - B11 - 0.5 * (sigma**2)
            ) * dt + (sigma / np.sqrt(252)) * Z[i]
            precios[i] = precios[i - 1] * np.exp(factor_exponente)

        variacion_diaria = pd.Series(precios).pct_change()
        variacion_diaria.iloc[0] = 0.0

        precio_futuro = np.zeros(n)
        precio_futuro[0] = precio_inicial
        for i in range(1, n):
            precio_futuro[i] = precio_futuro[i - 1] * (
                1 + variacion_diaria.iloc[i]
            )

        valuacion_contrato = precio_futuro * tamano_contrato
        ganancia_perdida_diaria = pd.Series(valuacion_contrato).diff()
        ganancia_perdida_diaria.iloc[0] = 0.0

        comisiones = np.zeros(n)
        comisiones[0] = 50.0
        comisiones[-1] = 50.0
        total_comisiones = np.sum(comisiones)

        margen = np.zeros(n)
        depositos_llamadas = np.zeros(n)
        capital_neto_inicial = capital_inicial - comisiones[0]

        for i in range(n):
            margen_prev = (
                capital_neto_inicial if i == 0 else margen[i - 1]
            )  # pylint: disable=cell-var-from-loop
            gp = ganancia_perdida_diaria.iloc[i]
            com = comisiones[i] if i == n - 1 else 0.0

            m_temp = margen_prev + gp - com

            dep = 0.0
            if m_temp < margen_llamada_val:
                dep = 15000.0 - m_temp
                m_temp = 15000.0

            margen[i] = m_temp
            depositos_llamadas[i] = dep

        precio_final = precio_futuro[-1]
        ganancia_bruta_mercado = (
            precio_final - precio_inicial
        ) * tamano_contrato
        total_depositos_llamadas = np.sum(depositos_llamadas)
        total_capital_inyectado = capital_inicial + total_depositos_llamadas
        efectivo_retirable = margen[-1]

        max_deposito = np.max(depositos_llamadas)
        if max_deposito > 0:
            idx_max = np.argmax(depositos_llamadas)
            dia_llamada_max = fechas[idx_max].strftime("%d/%m/%Y")
            precio_dolar_llamada_max = precio_futuro[idx_max]
        else:
            dia_llamada_max = "Sin llamadas"
            precio_dolar_llamada_max = 0.0

        resultados_simulaciones.append(
            {
                "Simulación (#)": sim + 1,
                "Inversión inicial": capital_inicial,
                "Ganancia bruta del mercado": ganancia_bruta_mercado,
                "Depósitos por llamadas": total_depositos_llamadas,
                "Total de capital inyectado": total_capital_inyectado,
                "Comisiones totales": total_comisiones,
                "Efectivo total retirable": efectivo_retirable,
                "Llamada máxima": max_deposito,
                "Día de llamada máxima": dia_llamada_max,
                "Precio dólar (llamada máx.)": precio_dolar_llamada_max,
                "Precio dólar (final)": precio_final,
            }
        )

    return procesar_resultados(
        resultados_simulaciones, alpha, capital_inicial
    )


def procesar_resultados(resultados_simulaciones, alpha_seleccionado, capital_inicial):  # pylint: disable=redefined-outer-name,unused-argument
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

        dic_estadisticos.append(
            {
                "Métrica": col,
                "Mínimo": serie.min(),
                "Q1 (25%)": serie.quantile(0.25),
                "Mediana (50%)": serie.median(),
                "Promedio": serie.mean(),
                "Q3 (75%)": serie.quantile(0.75),
                "Máximo": serie.max(),
                "Moda": moda_val,
                "Desviación Estándar": serie.std(),
            }
        )

    df_estadisticos = pd.DataFrame(dic_estadisticos)

    valor_inicial = 190000.0
    df_excel["Valor Total Final"] = (
        valor_inicial + df_excel["Ganancia bruta del mercado"]
    )
    p = alpha_seleccionado
    etiqueta = f"{int(p*100)}%"
    serie_valor = df_excel["Valor Total Final"]

    val_cutoff = serie_valor.quantile(p)
    val_var = valor_inicial - val_cutoff

    df_cut_var_valor = pd.DataFrame(
        [
            {
                "Métrica": "Valor Total del Contrato",
                "Tipo Cutoff": val_cutoff,
                "Tipo VaR": val_var,
                "Nivel Alpha (%)": etiqueta,
            }
        ]
    )

    return df_excel, df_estadisticos, df_cut_var_valor, etiqueta


# --- 3. RENDERIZADO SEGÚN LA OPCIÓN SELECCIONADA ---

if "Modelo 1: Simulación Diaria Detallada" in modelo_seleccionado:
    st.subheader("📅 Detalle de Simulación Diaria (Trayectoria Única)")

    df, fechas, precio_futuro, ganancia_perdida_diaria = (
        ejecutar_modelo_diario_detallado(precio_futuro_inicial, margen_llamada)
    )

    # Mostrar métricas clave en tarjetas coloridas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Precio Final", f"${df['Precio_Futuro'].iloc[-1]:,.4f}"
    )  # pylint: disable=no-member
    col2.metric(
        "Margen Final", f"${df['Margen'].iloc[-1]:,.2f}"
    )  # pylint: disable=no-member
    col3.metric(
        "Depósitos Totales",
        f"${df['Depositos_por_llamadas'].sum():,.2f}",
    )  # pylint: disable=no-member
    col4.metric(
        "Máx. Depósito Llamada",
        f"${df['Depositos_por_llamadas'].max():,.2f}",
    )  # pylint: disable=no-member

    st.markdown("---")
    st.markdown("### 📋 Tabla de Simulación Diaria")
    st.dataframe(df, use_container_width=True)

    # Gráfico colorido de la trayectoria del precio futuro y margen
    st.markdown("### 📈 Gráfica de Trayectoria y Márgenes")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_title(
        "Evolución del Precio Futuro y Márgenes Diarios",
        fontsize=14,
        color="darkblue",
        fontweight="bold",
    )
    ax1.plot(
        fechas,
        precio_futuro,
        color="#1f77b4",
        linewidth=2,
        label="Precio Futuro ($)",
    )
    ax1.set_xlabel("Fecha", fontsize=11)
    ax1.set_ylabel("Precio Dólar ($)", color="#1f77b4", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(
        fechas,
        df["Margen"],
        color="#2ca02c",
        linewidth=1.5,
        linestyle="--",
        label="Margen ($)",
    )
    ax2.set_ylabel("Margen Neto ($)", color="#2ca02c", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    fig.tight_layout()
    st.pyplot(fig)

else:
    # Modelos de Monte Carlo (Modelo 2 o 3)
    if "Modelo 2" in modelo_seleccionado:
        st.subheader(
            "🎲 Modelo Monte Carlo - Log-logístico (Múltiples Corridas)"
        )
        df_excel, df_estadisticos, df_cut_var_valor, etiqueta = (
            ejecutar_monte_carlo_modelo1(
                precio_futuro_inicial,
                margen_llamada,
                alpha_seleccionado,
                num_simulaciones,
            )
        )
    else:
        st.subheader("🎲 Modelo Monte Carlo - Movimiento Geométrico Browniano")
        df_excel, df_estadisticos, df_cut_var_valor, etiqueta = (
            ejecutar_monte_carlo_modelo2(
                precio_futuro_inicial,
                margen_llamada,
                alpha_seleccionado,
                num_simulaciones,
            )
        )

    st.markdown("---")
    st.markdown("### 📋 Vista Previa de Simulaciones (Primeras 10)")
    st.dataframe(df_excel.head(10), use_container_width=True)

    st.markdown("### 📊 Tabla de Estadísticos")
    st.dataframe(df_estadisticos, use_container_width=True)

    st.markdown(
        f"### 🛡️ Tabla de Cutoff y VaR (Alpha = {etiqueta} | Base: 190,000)"
    )
    st.dataframe(df_cut_var_valor, use_container_width=True)

    # Histogramas coloridos
    st.markdown("### 📊 Histogramas de Distribución de Resultados")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].hist(
        df_excel["Efectivo total retirable"],
        bins=50,
        color="#3498db",
        edgecolor="#2980b9",
        alpha=0.85,
    )
    axs[0, 0].set_title(
        f"Efectivo Total Retirable (Margen: ${margen_llamada:,.0f})",
        fontweight="bold",
    )
    axs[0, 0].set_xlabel("Monto ($)")
    axs[0, 0].set_ylabel("Frecuencia")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)

    axs[0, 1].hist(
        df_excel["Ganancia bruta del mercado"],
        bins=50,
        color="#2ecc71",
        edgecolor="#27ae60",
        alpha=0.85,
    )
    axs[0, 1].set_title(
        f"Ganancia Bruta del Mercado (Futuro: {precio_futuro_inicial})",
        fontweight="bold",
    )
    axs[0, 1].set_xlabel("Monto ($)")
    axs[0, 1].set_ylabel("Frecuencia")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)

    axs[1, 0].hist(
        df_excel["Total de capital inyectado"],
        bins=50,
        color="#e67e22",
        edgecolor="#d35400",
        alpha=0.85,
    )
    axs[1, 0].set_title(
        "Capital Inyectado Total", fontweight="bold"
    )
    axs[1, 0].set_xlabel("Monto ($)")
    axs[1, 0].set_ylabel("Frecuencia")
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)

    axs[1, 1].hist(
        df_excel["Llamada máxima"],
        bins=50,
        color="#e74c3c",
        edgecolor="#c0392b",
        alpha=0.85,
    )
    axs[1, 1].set_title(
        f"Llamada Máxima (Margen: ${margen_llamada:,.0f})",
        fontweight="bold",
    )
    axs[1, 1].set_xlabel("Monto ($)")
    axs[1, 1].set_ylabel("Frecuencia")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
