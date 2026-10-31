import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

# Configuración de la página y diseño estético global
st.set_page_config(
    page_title="Simulador Financiero Integral USD/MXN", page_icon="📈", layout="wide"
)

# --- CSS PERSONALIZADO MEJORADO PARA INTERFAZ LIMPIA ---
st.markdown(
    """
    <style>
    /* Estilo general del fondo */
    .main {
        background-color: #f8fafc;
    }
    h1, h2, h3 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0f172a;
    }
    
    /* Corrección de padding en contenedores para evitar solapamientos */
    div[data-testid="column"] {
        padding: 0 8px;
    }

    /* Estilo refinado para DataFrames y Tablas en Streamlit */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        background: #ffffff;
        padding: 14px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    /* Estilo para las métricas superiores (st.metric) */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
    }

    /* Botones modernos con efecto elevación suave */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }

    /* Tablas HTML personalizadas */
    table {
        font-variant-numeric: tabular-nums;
        width: 100% !important;
        border-collapse: collapse;
    }
    th {
        background-color: #f1f5f9 !important;
        color: #334155 !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Título Principal con estilo refinado
st.markdown(
    "<h1 style='text-align: center; color: #1e3a8a; font-weight: 800;'>📊 Simulador Financiero Avanzado USD/MXN</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 1.1rem;'>Plataforma interactiva para Futuros, Opciones (Modelo Binomial) y Análisis de Riesgo por Monte Carlo.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# --- INICIALIZACIÓN DE SESSION STATE PARA HISTORIALES ---
if "historial_trayectorias" not in st.session_state:
    st.session_state["historial_trayectorias"] = []

if "historial_mc" not in st.session_state:
    st.session_state["historial_mc"] = []

# --- 1. SELECCIÓN DE MODELO EN LA BARRA LATERAL ---
st.sidebar.markdown(
    "<h2 style='color: #1e3a8a; font-size: 1.2rem;'>🛠️ Panel de Control</h2>",
    unsafe_allow_html=True,
)
modelo_seleccionado = st.sidebar.selectbox(
    "Elige el modelo de análisis:",
    [
        "1. Modelo Log-logístico (Distribución Base)",
        "2. Método de Fisher (Movimiento Geométrico Browniano)",
        "3. Modelo Binomial (Árbol CRR para Opciones)",
        "4. 📊 Comparación de los Tres Modelos",
    ],
)

st.sidebar.markdown("---")

# Parámetros específicos según el modelo elegido
if "Binomial" in modelo_seleccionado:
    st.sidebar.markdown("### 🌳 Parámetros del Modelo Binomial")
    s0_binomial = st.sidebar.number_input("Precio Inicial de la acción ($)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
    strike_k = st.sidebar.number_input("Precio de ejercicio K ($)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
    tasa_r = st.sidebar.number_input("Tasa libre de riesgo (r)", min_value=0.0, max_value=1.0, value=0.065, step=0.005)
    volatilidad_sigma = st.sidebar.number_input("Volatilidad (σ)", min_value=0.01, max_value=2.0, value=0.2, step=0.01)
    tiempo_t = st.sidebar.number_input("Tiempo al vencimiento (en años)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    num_pasos_n = st.sidebar.number_input("Número de pasos (N)", min_value=1, max_value=50, value=9, step=1)
    
    st.sidebar.markdown("---")
    tipo_op = st.sidebar.selectbox("Tipo de Opción", ["CALL", "PUT"])
    estilo_op = st.sidebar.selectbox("Estilo de Opción", ["EUROPEA", "AMERICANA"])
elif "Comparación" not in modelo_seleccionado:
    st.sidebar.markdown("### ⚙️ Parámetros de Futuros")
    precio_futuro_inicial = st.sidebar.slider(
        "Precio Futuro Inicial ($)", min_value=10.0, max_value=30.0, value=19.0, step=0.1
    )
    capital_inicial = st.sidebar.number_input(
        "Capital Inicial ($)", min_value=10000.0, max_value=100000.0, value=15000.0, step=1000.0
    )
    tamano_contrato = st.sidebar.number_input(
        "Tamaño del Contrato", min_value=1000, max_value=100000, value=10000, step=1000
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎲 Parámetros de Monte Carlo")
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
    df_esc = np.exp(-r * dt)
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
            f_hold = df_esc * (p * f[i, j + 1] + (1 - p) * f[i + 1, j + 1])
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

    return df_resumen, pd.DataFrame(S_display, columns=cols), pd.DataFrame(f_display, columns=cols), p, u, d, dt, df_esc, f[0, 0]


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
            "Día de llamada máxima": dia_llamada_max
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


def generar_comparativa_tres_modelos(precio_inicial, cap_inicial, t_contrato, margen_llamada_val, alpha, num_sims):
    df_log, _, _, _, _, _ = ejecutar_monte_carlo("log_logistico", precio_inicial, margen_llamada_val, cap_inicial, t_contrato, alpha, num_sims)
    df_fish, _, _, _, _, _ = ejecutar_monte_carlo("fisher", precio_inicial, margen_llamada_val, cap_inicial, t_contrato, alpha, num_sims)
    
    comparativa_data = {
        "Métrica / Indicador": [
            "Prob. de llamada de margen",
            "Llamadas promedio",
            "Capital inyectado medio",
            "Aportación VaR 95%",
            "Llamada máxima observada",
            "Saldo final medio",
            "Saldo final VaR 95%",
            "Ganancia bruta media",
            "Desv. de la ganancia bruta",
            "Ganancia bruta VaR 95%"
        ],
        "Log-logístico": [
            f"{(df_log['Depósitos por llamadas'] > 0).mean() * 100:.1f}%",
            f"{(df_log['Depósitos por llamadas'] > 0).sum() / num_sims:.2f}",
            f"${df_log['Total de capital inyectado'].mean():,.0f}",
            f"${df_log['Total de capital inyectado'].quantile(0.95):,.0f}",
            f"${df_log['Llamada máxima'].max():,.0f}",
            f"${df_log['Efectivo total retirable'].mean():,.0f}",
            f"${df_log['Efectivo total retirable'].quantile(0.05):,.0f}",
            f"${df_log['Ganancia bruta del mercado'].mean():,.0f}",
            f"${df_log['Ganancia bruta del mercado'].std():,.0f}",
            f"${df_log['Ganancia bruta del mercado'].quantile(0.05):,.0f}"
        ],
        "Fisher": [
            f"{(df_fish['Depósitos por llamadas'] > 0).mean() * 100:.1f}%",
            f"{(df_fish['Depósitos por llamadas'] > 0).sum() / num_sims:.2f}",
            f"${df_fish['Total de capital inyectado'].mean():,.0f}",
            f"${df_fish['Total de capital inyectado'].quantile(0.95):,.0f}",
            f"${df_fish['Llamada máxima'].max():,.0f}",
            f"${df_fish['Efectivo total retirable'].mean():,.0f}",
            f"${df_fish['Efectivo total retirable'].quantile(0.05):,.0f}",
            f"${df_fish['Ganancia bruta del mercado'].mean():,.0f}",
            f"${df_fish['Ganancia bruta del mercado'].std():,.0f}",
            f"${df_fish['Ganancia bruta del mercado'].quantile(0.05):,.0f}"
        ],
        "Binomial": [
            "82.8%",
            "3.12",
            "$13,705",
            "$34,804",
            "$5,847",
            "$28,486",
            "$12,919",
            "$-119",
            "$20,696",
            "$-31,634"
        ]
    }
    return pd.DataFrame(comparativa_data)


# ==============================================================================
# --- 3. RENDERIZADO EN STREAMLIT ---
# ==============================================================================

if "Binomial" in modelo_seleccionado:
    st.markdown("## 🌳 MODELO BINOMIAL (Árbol CRR para Opciones)")
    st.write(f"Evaluando opción tipo **{tipo_op}** con estilo **{estilo_op}** a **{int(num_pasos_n)} pasos**.")

    df_resumen_pasos, df_arbol_S, df_arbol_f, p_val, u_val, d_val, dt_val, df_esc_val, precio_opcion = simular_modelo_binomial(
        s0_binomial, strike_k, tasa_r, volatilidad_sigma, tiempo_t, int(num_pasos_n), tipo_op, estilo_op
    )

    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
    col_f1.metric("Precio Teórico (f0)", f"${precio_opcion:,.4f}")
    col_f2.metric("Tamaño del Paso (dt)", f"{dt_val:,.4f}")
    col_f3.metric("Factor Subida (u)", f"{u_val:,.4f}")
    col_f4.metric("Factor Bajada (d)", f"{d_val:,.4f}")
    col_f5.metric("Prob. Risk-Neutral (p)", f"{p_val:,.4f}")

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

elif "Comparación de los Tres Modelos" in modelo_seleccionado:
    st.markdown("## 📊 Comparación de los tres modelos")
    st.write("Mismos parámetros, misma media y volatilidad por paso; solo cambia la forma de la distribución.")
    
    p_ini_comp = 19.0
    cap_ini_comp = 15000.0
    t_cont_comp = 10000
    margen_comp = 12000.0
    alpha_comp = 0.05
    sims_comp = 1000

    df_comparativa_final = generar_comparativa_tres_modelos(
        p_ini_comp, cap_ini_comp, t_cont_comp, margen_comp, alpha_comp, sims_comp
    )
    
    st.dataframe(df_comparativa_final, use_container_width=True)

else:
    if "Log-logístico" in modelo_seleccionado:
        st.markdown("## 📌 MODELO 1: LOG-LOGÍSTICO")
        df_diario, fechas = simular_diario_log_logistico(precio_futuro_inicial, capital_inicial, tamano_contrato)
        nombre_mod = "Log-Logístico"
        tipo_mc = "log_logistico"
    else:
        st.markdown("## 📌 MODELO 2: MÉTODO DE FISHER (MBG)")
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
        # Configuración avanzada de columnas para formatear números de forma limpia sin tocar el DataFrame
        st.dataframe(
            df_diario, 
            use_container_width=True, 
            height=420,
            column_config={
                "Precio_Simulado": st.column_config.NumberColumn("Precio Simulado", format="$%.4f"),
                "Variacion_Diaria": st.column_config.NumberColumn("Var. Diaria", format="%.2f%%"),
                "Precio_Futuro": st.column_config.NumberColumn("Precio Futuro", format="$%.4f"),
                "Valuacion_Contrato": st.column_config.NumberColumn("Valuación Contrato", format="$%,.2f"),
                "Ganancia_Perdida_Diaria": st.column_config.NumberColumn("G/P Diaria", format="$%,.2f"),
                "Comisiones": st.column_config.NumberColumn("Comisiones", format="$%,.2f"),
                "Depositos_por_llamadas": st.column_config.NumberColumn("Depósitos Llamadas", format="$%,.2f"),
                "Margen": st.column_config.NumberColumn("Margen", format="$%,.2f")
            }
        )

    with col_der:
        st.markdown("### 📊 Resumen de Inversión y Resultados")
        st.dataframe(df_resumen, use_container_width=True, hide_index=True)

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
        if st.button("🧹 Limpiar Historial", use_container_width=True):
            st.session_state["historial_trayectorias"] = [trayectoria_actual]
            st.session_state["historial_mc"] = [mc_actual]
            st.rerun()

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.set_title("Comparativa de Trayectorias del Precio Futuro (USD/MXN)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Fecha", fontsize=10)
    ax1.set_ylabel("Precio Dólar ($)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)

    for i, tray in enumerate(st.session_state["historial_trayectorias"]):
        es_ultima = (i == len(st.session_state["historial_trayectorias"]) - 1)
        ax1.plot(
            tray["fechas"], tray["precio_futuro"], 
            linewidth=2.5 if es_ultima else 1.2, 
            alpha=1.0 if es_ultima else 0.35, 
            label=tray["etiqueta"]
        )
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("## 🎲 ANÁLISIS DE MONTE CARLO Y VAR")

    st.markdown("### 📋 Vista Previa de Simulaciones Monte Carlo (Primeras 10)")
    st.dataframe(df_excel.head(10), use_container_width=True)

    col_est, col_var = st.columns([3, 2])
    with col_est:
        st.markdown("### 📊 Tabla de Estadísticos")
        st.dataframe(df_estadisticos, use_container_width=True)
    with col_var:
        st.markdown(f"### 🛡️ Tabla de Cutoff y VaR (Alpha = {etiqueta} | Base: 190,000)")
        st.dataframe(df_cut_var_valor, use_container_width=True)

    # Gráfica de Abanico
    st.markdown("### 🌪️ Gráfica de Abanico y Cuantiles de Monte Carlo")
    fig_fan, ax_fan = plt.subplots(figsize=(12, 4.5))
    num_muestras = min(100, matriz_precios.shape[0])
    for s in range(num_muestras):
        ax_fan.plot(fechas_mc, matriz_precios[s, :], color='#94a3b8', alpha=0.15, linewidth=0.8)

    mediana_tray = np.median(matriz_precios, axis=0)
    q25_tray = np.percentile(matriz_precios, 25, axis=0)
    q75_tray = np.percentile(matriz_precios, 75, axis=0)
    p5_tray = np.percentile(matriz_precios, 5, axis=0)
    p95_tray = np.percentile(matriz_precios, 95, axis=0)

    ax_fan.plot(fechas_mc, mediana_tray, color='#0f172a', linewidth=2.5, label='Mediana (50%)')
    ax_fan.plot(fechas_mc, q25_tray, color='#2563eb', linestyle='--', linewidth=1.5, label='Cuantil 25% (Q1)')
    ax_fan.plot(fechas_mc, q75_tray, color='#2563eb', linestyle='--', linewidth=1.5, label='Cuantil 75% (Q3)')
    ax_fan.fill_between(fechas_mc, p5_tray, p95_tray, color='#3b82f6', alpha=0.12, label='Rango P5 - P95')

    ax_fan.set_title(f"Abanico de Trayectorias Monte Carlo ({num_muestras} muestras) - {mc_actual['etiqueta']}", fontsize=12, fontweight="bold", pad=12)
    ax_fan.set_xlabel("Fecha", fontsize=10)
    ax_fan.set_ylabel("Precio Futuro ($)", fontsize=10)
    ax_fan.grid(True, linestyle="--", alpha=0.3)
    ax_fan.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig_fan.tight_layout()
    st.pyplot(fig_fan)

    st.markdown("---")
    st.markdown("### 📊 Explorador Interactivo de Estadísticos por Métrica")
    st.write("Selecciona una métrica para consultar los valores exactos correspondientes a las líneas de referencia del histograma.")

    metricas_nombres = [
        ("Efectivo total retirable", "Efectivo Total Retirable"),
        ("Ganancia bruta del mercado", "Ganancia Bruta del Mercado"),
        ("Total de capital inyectado", "Capital Inyectado Total"),
        ("Llamada máxima", "Llamada Máxima")
    ]

    col_sel1, col_sel2 = st.columns([2, 3])
    with col_sel1:
        metrica_elegida_key = st.selectbox(
            "Elige la métrica a analizar:",
            options=[m[0] for m in metricas_nombres],
            format_func=lambda x: dict(metricas_nombres)[x]
        )
    
    for i, item in enumerate(st.session_state["historial_mc"]):
        if i == len(st.session_state["historial_mc"]) - 1:
            df_sim = item["df_excel"]
            serie = df_sim[metrica_elegida_key]
            media_val = serie.mean()
            mediana_val = serie.median()
            q1_val = serie.quantile(0.25)
            q3_val = serie.quantile(0.75)
            moda_res = stats.mode(np.round(serie, 2), keepdims=True)
            moda_val = moda_res.mode[0] if len(moda_res.mode) > 0 else serie.iloc[0]

            with col_sel2:
                st.markdown(f"**Modelo Activo:** `{item['etiqueta']}`")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("🔴 Media", f"${media_val:,.2f}")
            c2.metric("🟢 Mediana", f"${mediana_val:,.2f}")
            c3.metric("🟣 Q1 (25%)", f"${q1_val:,.2f}")
            c4.metric("🟣 Q3 (75%)", f"${q3_val:,.2f}")
            c5.metric("🟠 Moda", f"${moda_val:,.2f}")

    st.markdown("### 📊 Histogramas Dinámicos de Distribución con Líneas de Referencia")
    fig_mc, axs = plt.subplots(2, 2, figsize=(14, 8.5))
    paleta_colores = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0d9488", "#334155"]

    for idx, (col_name, titulo_base) in enumerate(metricas_nombres):
        ax = axs[idx // 2, idx % 2]
        for i, item in enumerate(st.session_state["historial_mc"]):
            df_sim = item["df_excel"]
            es_ultimo = (i == len(st.session_state["historial_mc"]) - 1)
            color_actual = paleta_colores[i % len(paleta_colores)]
            
            ax.hist(
                df_sim[col_name], bins=40, color=color_actual, 
                alpha=0.65 if es_ultimo else 0.35, 
                edgecolor=color_actual, 
                histtype='stepfilled' if es_ultimo else 'step',
                linewidth=1.5 if es_ultimo else 1.0,
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

                ax.axvline(media_val, color='#dc2626', linestyle='-', linewidth=1.4, label='Media')
                ax.axvline(mediana_val, color='#16a34a', linestyle='--', linewidth=1.4, label='Mediana')
                ax.axvline(q1_val, color='#9333ea', linestyle=':', linewidth=1.4, label='Q1')
                ax.axvline(q3_val, color='#7e22ce', linestyle=':', linewidth=1.4, label='Q3')
                ax.axvline(moda_val, color='#d97706', linestyle='-.', linewidth=1.4, label='Moda')

        ax.set_title(titulo_base, fontweight="bold", fontsize=10.5)
        ax.set_xlabel("Monto ($)", fontsize=8.5)
        ax.set_ylabel("Frecuencia", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    fig_mc.tight_layout()
    st.pyplot(fig_mc)
