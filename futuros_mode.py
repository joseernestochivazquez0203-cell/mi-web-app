import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Simulador Financiero Integral USD/MXN", page_icon="📈", layout="wide"
)

# Título Principal
st.title("📊 Simulador Financiero Completo: Futuros, Opciones (Binomial) y Monte Carlo")
st.write(
    "Selecciona el enfoque en la barra lateral para alternar entre el **análisis de futuros (con historial superpuesto y Monte Carlo)** y el **Modelo Binomial de Opciones**."
)

# --- INICIALIZACIÓN DE SESSION STATE PARA HISTORIALES ---
if "historial_trayectorias" not in st.session_state:
    st.session_state["historial_trayectorias"] = []

if "historial_mc" not in st.session_state:
    st.session_state["historial_mc"] = []

# --- 1. SELECCIÓN DE MODELO EN LA BARRA LATERAL ---
st.sidebar.header("🛠️ Seleccionar Enfoque")
modelo_seleccionado = st.sidebar.selectbox(
    "Elige el modelo de análisis:",
    [
        "1. Modelo Log-logístico (Distribución Base)",
        "2. Método de Fisher (Movimiento Geométrico Browniano)",
        "3. Modelo Binomial (Árbol CRR para Opciones)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parámetros Generales")

precio_futuro_inicial = st.sidebar.slider(
    "Precio Inicial / Subyacente ($)", min_value=10.0, max_value=30.0, value=19.0, step=0.1
)

capital_inicial = st.sidebar.number_input(
    "Capital Inicial ($)", min_value=10000.0, max_value=100000.0, value=15000.0, step=1000.0
)

tamano_contrato = st.sidebar.number_input(
    "Tamaño del Contrato", min_value=1000, max_value=100000, value=10000, step=1000
)

# Parámetros específicos según el modelo elegido
if "Binomial" in modelo_seleccionado:
    st.sidebar.markdown("---")
    st.sidebar.header("🌳 Parámetros del Árbol Binomial")
    strike_k = st.sidebar.slider("Precio de Ejercicio (K) ($)", min_value=10.0, max_value=30.0, value=19.0, step=0.1)
    tasa_r = st.sidebar.slider("Tasa Libre de Riesgo (r)", min_value=0.01, max_value=0.20, value=0.065, step=0.005)
    volatilidad_sigma = st.sidebar.slider("Volatilidad (σ)", min_value=0.05, max_value=0.80, value=0.20, step=0.01)
    tiempo_t = st.sidebar.slider("Tiempo al Vencimiento (T en años)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    num_pasos_n = st.sidebar.slider("Número de Pasos (N)", min_value=2, max_value=20, value=5, step=1)
    tipo_op = st.sidebar.selectbox("Tipo de Opción", ["CALL", "PUT"])
    estilo_op = st.sidebar.selectbox("Estilo de Opción", ["EUROPEA", "AMERICANA"])
else:
    st.sidebar.markdown("---")
    st.sidebar.header("🎲 Parámetros de Monte Carlo y Riesgo")
    margen_llamada = st.sidebar.slider(
        "Margen de Llamada ($)", min_value=5000.0, max_value=15000.0, value=12000.0, step=500.0
    )
    alpha_seleccionado = st.sidebar.slider(
        "Alpha (p) para VaR", min_value=0.01, max_value=0.90, value=0.05, step=0.01
    )
    num_simulaciones = st.sidebar.slider(
        "Número de Simulaciones", min_value=100, max_value=10000, value=1000, step=100
    )


# ==============================================================================
# --- 2. FUNCIONES DE CÁLCULO Y SIMULACIÓN ---
# ==============================================================================

def simular_diario_log_logistico(precio_inicial, cap_inicial, t_contrato):
    B3 = 28.52977
    B4 = 18.32123
    fechas = pd.date_range(start="2026-10-15", end="2026-12-17", freq="D")
    n = len(fechas)

    np.random.seed(42)  
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

    np.random.seed(42)  
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


def simular_modelo_binomial(S0, K, r, sigma, T, N, op_tipo, op_estilo):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    df = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    S = np.zeros((N + 1, N + 1))
    f = np.zeros((N + 1, N + 1))

    for j in range(N + 1):
        for i in range(j + 1):
            S[i, j] = S0 * (u ** (j - i)) * (d ** i)

    for i in range(N + 1):
        if op_tipo == "CALL":
            f[i, N] = max(S[i, N] - K, 0)
        else:
            f[i, N] = max(K - S[i, N], 0)

    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            f_hold = df * (p * f[i, j + 1] + (1 - p) * f[i + 1, j + 1])
            if op_estilo == "AMERICANA":
                if op_tipo == "CALL":
                    f_exec = max(S[i, j] - K, 0)
                else:
                    f_exec = max(K - S[i, j], 0)
                f[i, j] = max(f_hold, f_exec)
            else:
                f[i, j] = f_hold

    resumen_pasos = []
    for j in range(N + 1):
        precios_nodo = S[:j + 1, j]
        opciones_nodo = f[:j + 1, j]
        resumen_pasos.append({
            "Paso": j,
            "Tiempo (t)": round(j * dt, 4),
            "Precio Mínimo ($)": round(np.min(precios_nodo), 4),
            "Precio Máximo ($)": round(np.max(precios_nodo), 4),
            "Precio Promedio ($)": round(np.mean(precios_nodo), 4),
            "Valor Opción (Promedio)": round(np.mean(opciones_nodo), 4),
            "Probabilidad (p)": round(p, 4)
        })

    df_resumen = pd.DataFrame(resumen_pasos)
    cols = [f"Paso {j}" for j in range(N + 1)]
    
    S_display = np.full((N + 1, N + 1), np.nan)
    f_display = np.full((N + 1, N + 1), np.nan)
    for j in range(N + 1):
        for i in range(j + 1):
            S_display[i, j] = round(S[i, j], 4)
            f_display[i, j] = round(f[i, j], 4)

    return df_resumen, pd.DataFrame(S_display, columns=cols), pd.DataFrame(f_display, columns=cols), p, u, d, dt, f[0, 0]


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
    matriz_precios_futuros = np.zeros((num_sims, n))

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

        matriz_precios_futuros[sim, :] = precio_futuro

        valuacion_contrato = precio_futuro * t_contrato
        gp_diaria = np.diff(valuacion_contrato, prepend=valuacion_contrato[0])
        gp_diaria[0] = 0.0

        comisiones = np.zeros(n)
        comisiones[0] = 50.0
        comisiones[-1] = 50.0

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
        else:
            dia_llamada_max = "Sin llamadas"

        resultados_simulaciones.append({
            "Simulación (#)": sim + 1,
            "Ganancia bruta del mercado": ganancia_bruta_mercado,
            "Depósitos por llamadas": total_depositos_llamadas,
            "Total de capital inyectado": total_capital_inyectado,
            "Efectivo total retirable": efectivo_retirable,
            "Llamada máxima": max_deposito,
        })

    df_excel = pd.DataFrame(resultados_simulaciones)

    columnas_numericas = [
        "Ganancia bruta del mercado",
        "Depósitos por llamadas",
        "Total de capital inyectado",
        "Efectivo total retirable",
        "Llamada máxima",
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
    val_cutoff = df_excel["Valor Total Final"].quantile(p)
    val_var = valor_inicial - val_cutoff

    df_cut_var_valor = pd.DataFrame([{
        "Métrica": "Valor Total del Contrato",
        "Tipo Cutoff": val_cutoff,
        "Tipo VaR": val_var,
        "Nivel Alpha (%)": etiqueta,
    }])

    return df_excel, df_estadisticos, df_cut_var_valor, etiqueta, matriz_precios_futuros, fechas


# ==============================================================================
# --- 3. RENDERIZADO EN STREAMLIT ---
# ==============================================================================

if "Binomial" in modelo_seleccionado:
    st.header("🌳 MODELO BINOMIAL (Árbol CRR para Opciones)")
    st.write(f"Evaluando opción tipo **{tipo_op}** con estilo **{estilo_op}** a **{num_pasos_n} pasos**.")

    df_resumen_pasos, df_arbol_S, df_arbol_f, p_val, u_val, d_val, dt_val, precio_opcion = simular_modelo_binomial(
        precio_futuro_inicial, strike_k, tasa_r, volatilidad_sigma, tiempo_t, num_pasos_n, tipo_op, estilo_op
    )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Precio Teórico Opción", f"${precio_opcion:,.4f}")
    col_m2.metric("Probabilidad Risk-Neutral (p)", f"{p_val:,.4f}")
    col_m3.metric("Factor de Subida (u)", f"{u_val:,.4f}")
    col_m4.metric("Factor de Bajada (d)", f"{d_val:,.4f}")

    st.markdown("---")
    st.markdown("### 📋 Tabla Resumen por Paso del Árbol Binomial")
    st.dataframe(df_resumen_pasos, use_container_width=True)

    col_arbol1, col_arbol2 = st.columns(2)
    with col_arbol1:
        st.markdown("### 📈 1. Árbol de Precios del Subyacente (S)")
        st.dataframe(df_arbol_S, use_container_width=True)

    with col_arbol2:
        st.markdown(f"### 📉 2. Árbol de Valor de la Opción ({tipo_op} {estilo_op})")
        st.dataframe(df_arbol_f, use_container_width=True)

else:
    if "Log-logístico" in modelo_seleccionado:
        st.header("📌 MODELO 1: LOG-LOGÍSTICO")
        df_diario, fechas = simular_diario_log_logistico(precio_futuro_inicial, capital_inicial, tamano_contrato)
        nombre_mod = "Log-Logístico"
        tipo_mc = "log_logistico"
    else:
        st.header("📌 MODELO 2: MÉTODO DE FISHER (MBG)")
        df_diario, fechas = simular_diario_fisher(precio_futuro_inicial, capital_inicial, tamano_contrato)
        nombre_mod = "Fisher (MBG)"
        tipo_mc = "fisher"

    df_resumen = generar_df_resumen(df_diario, capital_inicial, tamano_contrato, precio_futuro_inicial)

    trayectoria_actual = {
        "etiqueta": f"{nombre_mod} | P0=${precio_futuro_inicial:.1f} | Cap=${capital_inicial:,.0f}",
        "fechas": fechas,
        "precio_futuro": df_diario["Precio_Futuro"].copy(),
        "margen": df_diario["Margen"].copy()
    }

    if not st.session_state["historial_trayectorias"] or st.session_state["historial_trayectorias"][-1]["etiqueta"] != trayectoria_actual["etiqueta"]:
        st.session_state["historial_trayectorias"].append(trayectoria_actual)

    col_izq, col_der = st.columns([3, 2])
    with col_izq:
        st.markdown("### 📋 Tabla de Simulación Diaria")
        st.dataframe(df_diario, use_container_width=True, height=420)

    with col_der:
        st.markdown("### 📊 Resumen de Inversión y Resultados")
        st.table(df_resumen)

    df_excel, df_estadisticos, df_cut_var_valor, etiqueta, matriz_precios, fechas_mc = ejecutar_monte_carlo(
        tipo_mc, precio_futuro_inicial, margen_llamada, capital_inicial, tamano_contrato, alpha_seleccionado, num_simulaciones
    )

    mc_actual = {
        "etiqueta": f"{nombre_mod} | P0=${precio_futuro_inicial:.1f} | Margen=${margen_llamada:,.0f}",
        "df_excel": df_excel.copy(),
        "matriz_precios": matriz_precios.copy(),
        "fechas": fechas_mc.copy()
    }

    if not st.session_state["historial_mc"] or st.session_state["historial_mc"][-1]["etiqueta"] != mc_actual["etiqueta"]:
        st.session_state["historial_mc"].append(mc_actual)

    st.markdown("---")
    col_tit, col_btn = st.columns([3, 1])
    with col_tit:
        st.markdown("### 📈 Gráfica de Trayectorias Comparativas (Sobrepuestas)")
    with col_btn:
        if st.button("🧹 Limpiar Historial de Gráficas", use_container_width=True):
            st.session_state["historial_trayectorias"] = [trayectoria_actual]
            st.session_state["historial_mc"] = [mc_actual]
            st.rerun()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_title("Comparativa de Trayectorias del Precio Futuro (USD/MXN)", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("Fecha", fontsize=11)
    ax1.set_ylabel("Precio Dólar ($)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)

    for i, tray in enumerate(st.session_state["historial_trayectorias"]):
        es_ultima = (i == len(st.session_state["historial_trayectorias"]) - 1)
        ax1.plot(
            tray["fechas"], tray["precio_futuro"], 
            linewidth=2.5 if es_ultima else 1.2, 
            alpha=1.0 if es_ultima else 0.35, 
            label=tray["etiqueta"]
        )
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.header("🎲 ANÁLISIS DE MONTE CARLO Y VAR (Múltiples Corridas)")

    col_est, col_var = st.columns([3, 2])
    with col_est:
        st.markdown("### 📊 Tabla de Estadísticos")
        st.dataframe(df_estadisticos, use_container_width=True)
    with col_var:
        st.markdown(f"### 🛡️ Tabla de Cutoff y VaR (Alpha = {etiqueta} | Base: 190,000)")
        st.dataframe(df_cut_var_valor, use_container_width=True)

    # Gráfica de Abanico
    st.markdown("### 🌪️ Gráfica de Abanico y Cuantiles de Monte Carlo (Trayectorias Simuladas)")
    fig_fan, ax_fan = plt.subplots(figsize=(12, 5))
    num_muestras = min(100, matriz_precios.shape[0])
    for s in range(num_muestras):
        ax_fan.plot(fechas_mc, matriz_precios[s, :], color='gray', alpha=0.15, linewidth=0.8)

    mediana_tray = np.median(matriz_precios, axis=0)
    q25_tray = np.percentile(matriz_precios, 25, axis=0)
    q75_tray = np.percentile(matriz_precios, 75, axis=0)
    p5_tray = np.percentile(matriz_precios, 5, axis=0)
    p95_tray = np.percentile(matriz_precios, 95, axis=0)

    ax_fan.plot(fechas_mc, mediana_tray, color='#2c3e50', linewidth=2.5, label='Mediana (50%)')
    ax_fan.plot(fechas_mc, q25_tray, color='#2980b9', linestyle='--', linewidth=1.5, label='Cuantil 25% (Q1)')
    ax_fan.plot(fechas_mc, q75_tray, color='#2980b9', linestyle='--', linewidth=1.5, label='Cuantil 75% (Q3)')
    ax_fan.fill_between(fechas_mc, p5_tray, p95_tray, color='#3498db', alpha=0.15, label='Rango P5 - P95')

    ax_fan.set_title(f"Abanico de Trayectorias Monte Carlo ({num_muestras} muestras visualizadas) - {mc_actual['etiqueta']}", fontsize=12, fontweight="bold", pad=15)
    ax_fan.set_xlabel("Fecha", fontsize=11)
    ax_fan.set_ylabel("Precio Futuro ($)", fontsize=11)
    ax_fan.grid(True, linestyle="--", alpha=0.4)
    ax_fan.legend(loc="upper left", fontsize=9, framealpha=0.8)
    fig_fan.tight_layout()
    st.pyplot(fig_fan)

    # Histogramas Superpuestos con sus Leyendas Independientes en cada Subgráfico
    st.markdown("### 📊 Histogramas Dinámicos de Distribución (Monte Carlo Superpuestos con Estadísticos)")
    fig_mc, axs = plt.subplots(2, 2, figsize=(14, 9))
    metricas_mc = [
        ("Efectivo total retirable", "Efectivo Total Retirable"),
        ("Ganancia bruta del mercado", "Ganancia Bruta del Mercado"),
        ("Total de capital inyectado", "Capital Inyectado Total"),
        ("Llamada máxima", "Llamada Máxima")
    ]
    paleta_colores = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c", "#34495e", "#d35400"]

    for idx, (col_name, titulo_base) in enumerate(metricas_mc):
        ax = axs[idx // 2, idx % 2]
        for i, item in enumerate(st.session_state["historial_mc"]):
            df_sim = item["df_excel"]
            es_ultimo = (i == len(st.session_state["historial_mc"]) - 1)
            color_actual = paleta_colores[i % len(paleta_colores)]
            
            ax.hist(
                df_sim[col_name], bins=40, color=color_actual, 
                alpha=0.7 if es_ultimo else 0.4, 
                edgecolor=color_actual, 
                histtype='stepfilled' if es_ultimo else 'step',
                linewidth=1.8 if es_ultimo else 1.2,
                label=item["etiqueta"] if idx == 0 else ""
            )
            
            if es_ultimo:
                serie = df_sim[col_name]
                media_val = serie.mean()
                mediana_val = serie.median()
                q1_val = serie.quantile(0.25)
                q3_val = serie.quantile(0.75)
                
                moda_res = stats.mode(np.round(serie, 2), keepdims=True)
                moda_val = moda_res.mode[0] if len(moda_res.mode) > 0 else serie.iloc[0]

                ax.axvline(media_val, color='#e74c3c', linestyle='-', linewidth=1.5, label='Media')
                ax.axvline(mediana_val, color='#2ecc71', linestyle='--', linewidth=1.5, label='Mediana')
                ax.axvline(q1_val, color='#9b59b6', linestyle=':', linewidth=1.5, label='Q1')
                ax.axvline(q3_val, color='#8e44ad', linestyle=':', linewidth=1.5, label='Q3')
                ax.axvline(moda_val, color='#f39c12', linestyle='-.', linewidth=1.5, label='Moda')

        ax.set_title(titulo_base, fontweight="bold")
        ax.set_xlabel("Monto ($)")
        ax.set_ylabel("Frecuencia")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    if len(st.session_state["historial_mc"]) > 0:
        axs[0, 0].legend(fontsize=7, loc="upper left", framealpha=0.8)

    plt.tight_layout()
    st.pyplot(fig_mc)

    # Tabla de Estadísticos Exactos
    st.markdown("### 📋 Valores Estadísticos Exactos (Última Corrida Activa)")
    for i, item in enumerate(st.session_state["historial_mc"]):
        if i == len(st.session_state["historial_mc"]) - 1:
            df_sim = item["df_excel"]
            resumen_stats = []
            for col_name, titulo_base in metricas_mc:
                serie = df_sim[col_name]
                media_val = serie.mean()
                mediana_val = serie.median()
                q1_val = serie.quantile(0.25)
                q3_val = serie.quantile(0.75)
                
                moda_res = stats.mode(np.round(serie, 2), keepdims=True)
                moda_val = moda_res.mode[0] if len(moda_res.mode) > 0 else serie.iloc[0]
                
                resumen_stats.append({
                    "Métrica": titulo_base,
                    "Moda ($)": f"${moda_val:,.2f}",
                    "Media ($)": f"${media_val:,.2f}",
                    "Mediana (50%) ($)": f"${mediana_val:,.2f}",
                    "Q1 (25%) ($)": f"${q1_val:,.2f}",
                    "Q3 (75%) ($)": f"${q3_val:,.2f}"
                })
            st.subheader(f"Modelo: {item['etiqueta']}")
            st.dataframe(pd.DataFrame(resumen_stats), use_container_width=True)
