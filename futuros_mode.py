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
