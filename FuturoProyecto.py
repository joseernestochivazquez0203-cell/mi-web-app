import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
        padding: 12px 12px;
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
        overflow: visible !important;
    }
    /* Evita el recorte con "..." y escala la fuente según el ancho de pantalla */
    div[data-testid="stMetricValue"] > div {
        font-size: clamp(0.95rem, 1.35vw, 1.6rem) !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p {
        overflow: visible !important;
        text-overflow: clip !important;
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

# --- PARÁMETROS FIJOS DE FUTUROS ---
RAZON_APORTACION = 1.5   # $15,000 de aportación por cada $10,000 de tamaño de contrato
COMISION_FIJA = 50.0     # Comisión fija (paso inicial y paso final)


def calcular_aportacion_inicial(tamano_contrato):
    """Aportación inicial proporcional: Tamaño del Contrato × 1.5."""
    return float(tamano_contrato) * RAZON_APORTACION


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
        "5. 📐 Fórmulas Usadas",
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
elif "Comparación" not in modelo_seleccionado and "Fórmulas" not in modelo_seleccionado:
    st.sidebar.markdown("### ⚙️ Parámetros de Futuros")
    precio_futuro_inicial = st.sidebar.slider(
        "Precio Futuro Inicial ($)", min_value=10.0, max_value=30.0, value=19.0, step=0.1
    )
    tamano_contrato = st.sidebar.number_input(
        "Tamaño del Contrato", min_value=1000, max_value=100000, value=10000, step=1000
    )
    # Aportación inicial = Tamaño del Contrato × 1.5 (proporción $15,000 : $10,000)
    capital_inicial = calcular_aportacion_inicial(tamano_contrato)
    st.sidebar.markdown(
        "<div style='background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #2563eb;"
        "border-radius:8px;padding:10px 12px;font-size:0.85rem;color:#334155;line-height:1.6;'>"
        f"<b>Aportación inicial:</b> ${capital_inicial:,.2f}<br>"
        f"<b>Comisión inicial:</b> −${COMISION_FIJA:,.2f}<br>"
        f"<b>Margen inicial:</b> ${capital_inicial - COMISION_FIJA:,.2f}"
        "</div>",
        unsafe_allow_html=True,
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
    comisiones[0] = COMISION_FIJA
    comisiones[-1] = COMISION_FIJA

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
    comisiones[0] = COMISION_FIJA
    comisiones[-1] = COMISION_FIJA

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


def crear_figura_arbol(df_arbol, N, dt, titulo, tipo="precio", S0=None, etiqueta_valor="Precio"):
    """Árbol binomial recombinante en Plotly.

    df_arbol: DataFrame (filas = nº de bajadas i, columnas = paso j) con NaN fuera del árbol.
    Posición de cada nodo: x = paso j, y = (subidas - bajadas) = j - 2i.
    """
    valores = df_arbol.to_numpy(dtype=float)

    if N <= 6:
        tam, fuente = 48, 11
    elif N <= 9:
        tam, fuente = 44, 10
    elif N <= 14:
        tam, fuente = 34, 8
    else:
        tam, fuente = 12, 8
    mostrar_texto = N <= 14

    edge_x, edge_y = [], []
    xs, ys, textos, hovers, vals = [], [], [], [], []
    for j in range(N + 1):
        for i in range(j + 1):
            y = j - 2 * i
            v = valores[i, j]
            xs.append(j)
            ys.append(y)
            vals.append(v)
            textos.append(f"{v:,.2f}" if mostrar_texto else "")
            hovers.append(
                f"<b>Paso {j}</b> (t = {j * dt:.4f})<br>"
                f"Subidas: {j - i} | Bajadas: {i}<br>"
                f"{etiqueta_valor}: <b>{v:,.4f}</b>"
            )
            if j < N:
                # arista hacia el nodo de subida (i, j+1) y hacia el de bajada (i+1, j+1)
                edge_x += [j, j + 1, None, j, j + 1, None]
                edge_y += [y, y + 1, None, y, y - 1, None]

    vals = np.array(vals)
    if tipo == "precio":
        dif = vals - S0
        lim = max(float(np.max(np.abs(dif))), 1e-9)
        color_vals, cmin, cmax = dif, -lim, lim
        escala = [[0.0, "#fecaca"], [0.5, "#f1f5f9"], [1.0, "#bbf7d0"]]
    else:
        color_vals, cmin, cmax = vals, 0.0, max(float(vals.max()), 1e-9)
        escala = [[0.0, "#f1f5f9"], [1.0, "#93c5fd"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#94a3b8", width=1.2),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text" if mostrar_texto else "markers",
        text=textos, textposition="middle center",
        textfont=dict(size=fuente, color="#0f172a"),
        marker=dict(
            size=tam, color=color_vals, colorscale=escala, cmin=cmin, cmax=cmax,
            showscale=False, line=dict(color="#475569", width=1),
        ),
        hovertext=hovers, hoverinfo="text", showlegend=False,
    ))

    px_unidad = max(tam * 0.62, 9)
    altura = int(min(max(2 * N * px_unidad + 140, 480), 1600))
    paso_tick = max(1, N // 10)
    fig.update_layout(
        title=dict(text=titulo, x=0.01, font=dict(size=16, color="#0f172a")),
        template="plotly_white",
        height=altura,
        margin=dict(l=20, r=20, t=60, b=40),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(0, N + 1, paso_tick)),
            ticktext=[f"Paso {j}" for j in range(0, N + 1, paso_tick)],
            showgrid=False, zeroline=False, range=[-0.6, N + 0.6],
        ),
        yaxis=dict(visible=False, range=[-N - 1.5, N + 1.5]),
        hoverlabel=dict(bgcolor="#ffffff", font_size=12),
    )
    return fig


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
        comisiones[0] = COMISION_FIJA
        comisiones[-1] = COMISION_FIJA

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
            "Llamada máxima promedio",
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
            f"${df_log['Llamada máxima'].mean():,.0f}",
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
            f"${df_fish['Llamada máxima'].mean():,.0f}",
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
            "—",
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

    st.caption(
        "Cada nodo es un precio posible: la rama superior es una subida (u) y la inferior una bajada (d). "
        "Pasa el cursor sobre un nodo para ver su valor exacto."
    )

    st.markdown("### 📈 1. Árbol de Precios del Subyacente (S)")
    st.plotly_chart(
        crear_figura_arbol(
            df_arbol_S, int(num_pasos_n), dt_val, "Evolución del subyacente S(t, j)",
            tipo="precio", S0=s0_binomial, etiqueta_valor="Precio S",
        ),
        use_container_width=True, key="arbol_precios",
    )
    with st.expander("Ver tabla numérica del árbol de precios"):
        st.dataframe(df_arbol_S, use_container_width=True)

    st.markdown(f"### 📉 2. Árbol de Valor de la Opción ({tipo_op} {estilo_op})")
    st.plotly_chart(
        crear_figura_arbol(
            df_arbol_f, int(num_pasos_n), dt_val, f"Valor de la opción f(t, j) — {tipo_op} {estilo_op}",
            tipo="opcion", etiqueta_valor="Valor opción f",
        ),
        use_container_width=True, key="arbol_opcion",
    )
    with st.expander("Ver tabla numérica del árbol de la opción"):
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

elif "Fórmulas Usadas" in modelo_seleccionado:
    st.markdown("## 📐 Fórmulas Usadas")
    st.write("Resumen de las ecuaciones que utiliza el simulador, organizadas por módulo.")

    # ---------------- 1. FUTUROS ----------------
    st.markdown("### 1️⃣ Futuros: aportación, margen y efectivo retirable")
    f1, f2 = st.columns(2)
    with f1:
        with st.container(border=True):
            st.markdown("**Aportación inicial proporcional**")
            st.latex(r"A = T \times 1.5")
            st.caption("T = tamaño del contrato. Proporción: $15,000 de aportación por cada $10,000 de contrato.")
            st.markdown("**Comisión y margen inicial**")
            st.latex(r"c_0 = c_n = 50 \qquad M_0 = A - c_0")
            st.caption("Ej.: T = 10,000 → A = 15,000 → M₀ = 15,000 − 50 = 14,950.")
    with f2:
        with st.container(border=True):
            st.markdown("**Precio futuro y valuación del contrato**")
            st.latex(r"F_t = F_{t-1}\,(1 + r_t), \qquad r_t = \frac{P_t}{P_{t-1}} - 1")
            st.latex(r"V_t = F_t \cdot T")
            st.markdown("**Ganancia / pérdida diaria**")
            st.latex(r"GP_t = V_t - V_{t-1}, \qquad GP_0 = 0")

    f3, f4 = st.columns(2)
    with f3:
        with st.container(border=True):
            st.markdown("**Margen diario y llamadas de margen**")
            st.latex(r"\tilde{M}_t = M_{t-1} + GP_t - c_t\,\mathbb{1}_{\{t=n\}}")
            st.latex(
                r"""
                \text{Si } \tilde{M}_t < L:\quad D_t = A - \tilde{M}_t,\;\; M_t = A
                \qquad\text{en otro caso:}\quad D_t = 0,\;\; M_t = \tilde{M}_t
                """
            )
            st.caption(
                "L = margen de llamada (Monte Carlo). En la simulación diaria individual el disparador es "
                "M̃ₜ ≤ 0. D = depósito por llamada."
            )
    with f4:
        with st.container(border=True):
            st.markdown("**Resultados finales**")
            st.latex(r"\text{Efectivo total retirable} = M_n")
            st.latex(r"\text{Capital inyectado total} = A + \sum_{t=0}^{n} D_t")
            st.latex(r"\text{Ganancia bruta del mercado} = (F_n - F_0)\cdot T")
            st.latex(r"\text{Comisiones totales} = \sum_{t=0}^{n} c_t = 100")

    st.markdown("---")

    # ---------------- 2. ESTOCÁSTICOS ----------------
    st.markdown("### 2️⃣ Simulación estocástica del tipo de cambio USD/MXN")
    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            st.markdown("**Distribución Log-Logística**")
            st.latex(r"F(x) = \frac{1}{1 + (x/\beta)^{-\alpha}}")
            st.markdown("Simulación por transformada inversa:")
            st.latex(r"P_t = \beta \left(\frac{U_t}{1 - U_t}\right)^{1/\alpha}, \quad U_t \sim \mathcal{U}(0.0001,\,0.9999)")
            st.caption("α = B3 = 28.52977 · β = B4 = 18.32123 · P₀ = 18.44")
    with g2:
        with st.container(border=True):
            st.markdown("**Modelo de Fisher (Movimiento Browniano Geométrico)**")
            st.latex(r"P_t = P_{t-1}\exp\!\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z_t\right]")
            st.latex(r"Z_t \sim \mathcal{N}(0,1), \quad \Delta t = \tfrac{1}{252}, \quad \mu = B_{10} - B_{11}")
            st.caption("B10 = 0.01736111 · B11 = 0.01119444 · σ = 17.69391793 % (0.1769…) · P₀ = 18.44")

    with st.container(border=True):
        st.markdown("**Cutoff y VaR (Monte Carlo)**")
        v1, v2 = st.columns(2)
        with v1:
            st.latex(r"V^{(k)}_{final} = V_0 + \text{Ganancia bruta}^{(k)}, \quad V_0 = 190{,}000")
            st.latex(r"\text{Cutoff}_\alpha = Q_\alpha\!\left(V^{(k)}_{final}\right)")
            st.latex(r"\text{VaR}_\alpha = V_0 - \text{Cutoff}_\alpha")
        with v2:
            st.markdown("Estadísticos de las métricas simuladas:")
            st.latex(r"\bar{x} = \frac{1}{K}\sum_{k=1}^{K} x_k, \qquad Q_{p}(x) = \text{cuantil de orden } p")
            st.caption("Q1 = Q₀.₂₅, mediana = Q₀.₅, Q3 = Q₀.₇₅, máximo = máxₖ xₖ, moda = valor más frecuente (redondeado a centavos). K = número de simulaciones.")

    st.markdown("---")

    # ---------------- 3. BINOMIAL ----------------
    st.markdown("### 3️⃣ Modelo Binomial de Cox-Ross-Rubinstein (CRR)")
    b1, b2 = st.columns(2)
    with b1:
        with st.container(border=True):
            st.markdown("**Factores del árbol y probabilidad neutral al riesgo**")
            st.latex(r"\Delta t = \frac{T}{N}")
            st.latex(r"u = e^{\sigma\sqrt{\Delta t}}, \qquad d = \frac{1}{u}")
            st.latex(r"p = \frac{e^{r\Delta t} - d}{u - d}")
    with b2:
        with st.container(border=True):
            st.markdown("**Evolución del subyacente**")
            st.latex(r"S_{t,j} = S_0\, u^{\,t-j}\, d^{\,j}")
            st.caption("t = paso (0…N), j = número de bajadas (0…t). El árbol es recombinante.")
            st.markdown("**Pago en el vencimiento (t = N)**")
            st.latex(r"f_{N,j}^{CALL} = \max(S_{N,j} - K,\,0), \qquad f_{N,j}^{PUT} = \max(K - S_{N,j},\,0)")

    c_eu, c_am = st.columns(2)
    with c_eu:
        with st.container(border=True):
            st.markdown("**Valuación hacia atrás: opción EUROPEA**")
            st.latex(r"f_{t,j} = e^{-r\Delta t}\left[\,p\, f_{t+1,j} + (1-p)\, f_{t+1,j+1}\right]")
            st.caption("Aplica igual para Call y Put; solo cambia el pago terminal.")
    with c_am:
        with st.container(border=True):
            st.markdown("**Valuación hacia atrás: opción AMERICANA**")
            st.latex(r"f_{t,j} = \max\!\left\{\, e^{-r\Delta t}\left[p\, f_{t+1,j} + (1-p)\, f_{t+1,j+1}\right],\; g(S_{t,j}) \right\}")
            st.latex(r"g^{CALL}(S) = \max(S-K,0), \qquad g^{PUT}(S) = \max(K-S,0)")
            st.caption("g = valor de ejercicio inmediato en cada nodo.")

    with st.container(border=True):
        st.markdown("**Precio teórico de la opción**")
        st.latex(r"\text{Prima} = f_{0,0}")

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
        "etiqueta": f"{nombre_mod} | P0=${precio_futuro_inicial:.1f} | Margen=${margen_llamada:,.0f} | Cap=${capital_inicial:,.0f}",
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
            
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("🔴 Media", f"${media_val:,.2f}")
            c2.metric("🟢 Mediana", f"${mediana_val:,.2f}")
            c3.metric("🟣 Q1 (25%)", f"${q1_val:,.2f}")
            c4.metric("🟣 Q3 (75%)", f"${q3_val:,.2f}")
            c5.metric("🟠 Moda", f"${moda_val:,.2f}")
            c6.metric("🔵 Máximo", f"${serie.max():,.2f}")

    st.markdown("### 📊 Histogramas Dinámicos de Distribución con Líneas de Referencia")
    fig_mc, axs = plt.subplots(2, 2, figsize=(16, 11))
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

    # Ajuste seguro de subplots para evitar errores de layout en Matplotlib
    # Leyenda única fuera de las gráficas (abajo) para que no tape las barras
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig_mc.legend(
        handles, labels, loc="lower center", ncol=4, fontsize=10,
        framealpha=0.95, bbox_to_anchor=(0.5, 0.0),
    )
    fig_mc.subplots_adjust(hspace=0.35, wspace=0.22, bottom=0.12)
    st.pyplot(fig_mc)
