import os
import json
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# CONFIGURACIÓN INICIAL
# =====================================================
st.set_page_config(page_title="Consulta de Acciones", page_icon="📊", layout="wide")

# ---- Estilos (CSS) sobrios ----
CUSTOM_CSS = """
<style>
.main .block-container {max-width: 1200px; padding-top: 1rem; padding-bottom: 2rem;}
h1, h2, h3 { font-weight: 700; letter-spacing: -0.2px; }
h1 { margin-bottom: 0.25rem; }
.section-subtitle { color: #6b7280; margin-bottom: 1.25rem; }
.hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.25rem 0; }
.stSidebar { border-right: 1px solid #e5e7eb; }
.sidebar-title { font-weight: 700; font-size: 1.1rem; margin-top: 0.5rem; }
.small { color:#6b7280; font-size:0.92rem; }
.stTextInput>div>div>input, .stNumberInput input, .stSelectbox>div>div>div { border-radius: 8px; }
.stat-label { color:#6b7280; font-size:0.85rem; margin-bottom: 0.15rem; }
.stat-value { font-weight:700; font-size:1.1rem; }
.card { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:12px; height:92px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown('<div class="sidebar-title">Menú</div>', unsafe_allow_html=True)
menu = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Consulta de Acciones",
        "Riesgo de Inversión",
        "CAPM",
        "Optimización de Portafolio (Markowitz)",
        "Simulación Monte Carlo",
    ]
)

# =====================================================
# SELECTOR DE PERIODO (global)
# =====================================================
RANGE_OPTIONS = ["1 semana", "1 mes", "6 meses", "1 año", "YTD", "3 años", "5 años"]

def range_to_yf_params(range_key: str):
    mapping = {
        "1 semana": ("7d", "1d"),
        "1 mes": ("1mo", "1d"),
        "6 meses": ("6mo", "1d"),
        "1 año": ("1y", "1d"),
        "YTD": ("ytd", "1d"),
        "3 años": ("3y", "1wk"),
        "5 años": ("5y", "1wk"),
    }
    return mapping.get(range_key, ("6mo", "1d"))

# =====================================================
# HELPERS DE DATOS
# =====================================================
@st.cache_data(ttl=3600)
def get_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna()

def add_smas(df: pd.DataFrame, windows=(20, 50, 200)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"SMA{w}"] = out["Close"].rolling(w).mean()
    return out

def return_metrics(df: pd.DataFrame):
    d = df.copy()
    d["Return"] = d["Close"].pct_change()
    d["LogReturn"] = np.log(d["Close"] / d["Close"].shift(1))
    d = d.dropna()
    avg_daily = d["Return"].mean()
    ann_return = (1 + avg_daily) ** 252 - 1
    ann_vol = d["Return"].std() * np.sqrt(252)
    return d, ann_return, ann_vol

@st.cache_data(ttl=1800)
def load_prices(tickers, period="1y", interval="1d") -> pd.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(tickers, period=period, interval=interval, auto_adjust=True,
                      progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.levels[0]:
            prices = raw["Close"].copy()
        elif "Adj Close" in raw.columns.levels[0]:
            prices = raw["Adj Close"].copy()
        else:
            first = raw.columns.levels[0][0]
            prices = raw[first].copy()
    else:
        if "Close" in raw.columns:
            prices = raw[["Close"]].copy()
        elif "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy().rename(columns={"Adj Close": "Close"})
        else:
            return pd.DataFrame()
        prices.columns = [tickers[0]]

    return prices.dropna(how="all")

def compute_beta_alpha(stock_returns: pd.Series, market_returns: pd.Series):
    merged = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    if merged.shape[0] < 2:
        return np.nan, np.nan
    cov = np.cov(merged.iloc[:, 0], merged.iloc[:, 1])[0][1]
    var_m = merged.iloc[:, 1].var()
    beta = cov / var_m if var_m != 0 else np.nan
    alpha = merged.iloc[:, 0].mean() - beta * merged.iloc[:, 1].mean()
    return beta, alpha

# =====================================================
# FORMATEADORES PARA MÉTRICAS (10 datos clave)
# =====================================================
def human_number(n):
    try:
        n = float(n)
    except Exception:
        return "N/A"
    absn = abs(n)
    if absn >= 1e12:  return f"{n/1e12:.2f}T"
    if absn >= 1e9:   return f"{n/1e9:.2f}B"
    if absn >= 1e6:   return f"{n/1e6:.2f}M"
    if absn >= 1e3:   return f"{n/1e3:.2f}K"
    return f"{n:.2f}"

def pct(x):
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "N/A"

def ratio(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "N/A"

# =====================================================
# ESTILO DE GRÁFICAS
# =====================================================
TEMPLATE = "simple_white"
COLOR_UP = "rgba(22,163,74,1)"
COLOR_UP_FILL = "rgba(22,163,74,0.9)"
COLOR_DOWN = "rgba(220,38,38,1)"
COLOR_DOWN_FILL = "rgba(220,38,38,0.9)"
SMA_COLORS = {"SMA20": "#5546d6", "SMA50": "#f59e0b", "SMA200": "#b78d0a"}

def plot_candles_with_volume(df: pd.DataFrame, title: str, show_sma20=True, show_sma50=True, show_sma200=True):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.72, 0.28])

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLC",
            increasing_line_color=COLOR_UP,
            increasing_fillcolor=COLOR_UP_FILL,
            decreasing_line_color=COLOR_DOWN,
            decreasing_fillcolor=COLOR_DOWN_FILL,
            line=dict(width=1.2),
            whiskerwidth=0.5,
            showlegend=True,
            opacity=0.95,
        ),
        row=1, col=1
    )

    if show_sma20 and "SMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], mode="lines",
                                 line=dict(color=SMA_COLORS["SMA20"], width=1.6),
                                 name="SMA20"), row=1, col=1)
    if show_sma50 and "SMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], mode="lines",
                                 line=dict(color=SMA_COLORS["SMA50"], width=1.6),
                                 name="SMA50"), row=1, col=1)
    if show_sma200 and "SMA200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], mode="lines",
                                 line=dict(color=SMA_COLORS["SMA200"], width=1.6),
                                 name="SMA200"), row=1, col=1)

    vol_colors = [COLOR_UP_FILL if c >= o else COLOR_DOWN_FILL for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors,
                         name="Volumen", opacity=0.85), row=2, col=1)

    fig.update_xaxes(
        showgrid=False,
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="year", stepmode="todate", label="YTD"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
        rangebreaks=[dict(bounds=["sat", "mon"])],
        row=1, col=1
    )
    fig.update_xaxes(showgrid=False, rangebreaks=[dict(bounds=["sat", "mon"])], row=2, col=1)

    fig.update_yaxes(showgrid=True, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Volumen", showgrid=True, zeroline=False, row=2, col=1)

    fig.update_layout(
        title=title,
        template=TEMPLATE,
        height=760,
        margin=dict(l=40, r=20, t=48, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode="pan",
    )
    return fig

# =====================================================
# 1) CONSULTA DE ACCIONES (con 10 datos clave)
# =====================================================
if menu == "Consulta de Acciones":
    st.markdown("<h1>Consulta de Acciones</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Ficha ejecutiva del ticker y gráfica de velas.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c0, c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1, 1.2])
    with c0:
        stonk = st.text_input("Símbolo", "MSFT").strip().upper()
    with c1:
        range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("6 meses"))
    with c2:
        show_sma20 = st.checkbox("SMA20", value=True)
    with c3:
        show_sma50 = st.checkbox("SMA50", value=True)
    with c4:
        show_sma200 = st.checkbox("SMA200", value=False)

    ticker = yf.Ticker(stonk)
    info = ticker.info if hasattr(ticker, "info") else {}

    # 10 datos clave (con fallback)
    key_data = {
        "Sector": info.get("sector", "N/A"),
        "Industria": info.get("industry", "N/A"),
        "País": info.get("country", "N/A"),
        "Market Cap": human_number(info.get("marketCap")),
        "Beta (5y mensual)": ratio(info.get("beta")),
        "P/E (Trailing)": ratio(info.get("trailingPE")),
        "P/E (Forward)": ratio(info.get("forwardPE")),
        "P/B": ratio(info.get("priceToBook")),
        "Dividend Yield": pct(info.get("dividendYield")),
        "Cambio 52 semanas": pct(info.get("52WeekChange")),
    }

    # Grid 2x5
    st.subheader("Indicadores clave")
    cols_top = st.columns(5)
    cols_bottom = st.columns(5)
    keys = list(key_data.keys())
    for i, col in enumerate(cols_top):
        label = keys[i]
        val = key_data[label]
        with col:
            st.markdown(f'<div class="card"><div class="stat-label">{label}</div><div class="stat-value">{val}</div></div>', unsafe_allow_html=True)
    for i, col in enumerate(cols_bottom, start=5):
        label = keys[i]
        val = key_data[label]
        with col:
            st.markdown(f'<div class="card"><div class="stat-label">{label}</div><div class="stat-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Gráfica de velas con volumen")

    period, interval = range_to_yf_params(range_key)
    try:
        hist = ticker.history(period=period, interval=interval, auto_adjust=False)
    except Exception as e:
        hist = pd.DataFrame()
        st.error(f"Error al obtener datos: {e}")

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if hist.empty or not required_cols.issubset(hist.columns):
        st.warning("No se pudo obtener información histórica suficiente. Intenta con otro rango o símbolo válido.")
    else:
        hist = hist.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        hist = add_smas(hist, (20, 50, 200))
        title = f"{stonk} · {range_key}"
        fig = plot_candles_with_volume(
            hist, title,
            show_sma20=show_sma20, show_sma50=show_sma50, show_sma200=show_sma200
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 2) RIESGO DE INVERSIÓN
# =====================================================
elif menu == "Riesgo de Inversión":
    st.markdown("<h1>Riesgo de Inversión</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Beta, alpha y varianza vs. un índice de referencia.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        stonk = st.text_input("Símbolo", "NVDA").upper()
    with c2:
        market = st.text_input("Índice de referencia (p. ej. ^GSPC, ^IXIC)", "^GSPC").upper()
    with c3:
        range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("1 año"))

    period, interval = range_to_yf_params(range_key)
    df_stock = get_history(stonk, period, interval)
    df_market = get_history(market, period, interval)

    if df_stock.empty or df_market.empty:
        st.warning("No se pudieron obtener los datos.")
    else:
        s_ret, _, _ = return_metrics(df_stock)
        m_ret, _, _ = return_metrics(df_market)

        beta, alpha = compute_beta_alpha(s_ret["Return"], m_ret["Return"])
        var_market = m_ret["Return"].var()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Beta", f"{beta:.3f}")
        with c2:
            st.metric("Varianza mercado", f"{var_market:.6f}")
        with c3:
            st.metric("Alpha", f"{alpha:.4f}")

        st.subheader("Relación de rendimientos (acción vs mercado)")
        st.scatter_chart(pd.DataFrame({"Stock": s_ret["Return"], "Market": m_ret["Return"]}).dropna())

# =====================================================
# 3) CAPM
# =====================================================
elif menu == "CAPM":
    st.markdown("<h1>CAPM</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Rendimiento esperado en función de β y del premio por riesgo.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        rf = st.number_input("Tasa libre de riesgo (0.04 = 4%)", value=0.04)
    with c2:
        rm = st.number_input("Rendimiento esperado del mercado (0.10 = 10%)", value=0.10)
    with c3:
        use_data_beta = st.toggle("Estimar β desde datos", value=False)

    if use_data_beta:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            stonk = st.text_input("Ticker del activo", "AAPL").upper()
        with c2:
            market = st.text_input("Benchmark (p. ej. ^GSPC)", "^GSPC").upper()
        with c3:
            range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("3 años"))
        period, interval = range_to_yf_params(range_key)
        s = get_history(stonk, period, interval)
        m = get_history(market, period, interval)
        if s.empty or m.empty:
            st.warning("No se pudo estimar β (datos insuficientes). Se usará β=1.0.")
            beta = 1.0
        else:
            s_ret, _, _ = return_metrics(s)
            m_ret, _, _ = return_metrics(m)
            beta, _ = compute_beta_alpha(s_ret["Return"], m_ret["Return"])
            if np.isnan(beta):
                beta = 1.0
    else:
        beta = st.number_input("β del activo", value=1.2)

    expected_return = rf + beta * (rm - rf)
    st.metric("Rendimiento esperado (CAPM)", f"{expected_return*100:.2f}%")

    betas = np.linspace(0, 2, 50)
    returns = rf + betas * (rm - rf)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=betas, y=returns, mode="lines", name="Línea SML"))
    fig.add_trace(go.Scatter(x=[beta], y=[expected_return],
                             mode="markers+text", text=["Tu activo"], textposition="bottom center",
                             marker=dict(size=10, color="red"), name="Activo"))
    fig.update_layout(title="Security Market Line (SML)", xaxis_title="Beta", yaxis_title="Rendimiento esperado",
                      template="simple_white", height=520, margin=dict(l=40, r=20, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 4) MARKOWITZ
# =====================================================
elif menu == "Optimización de Portafolio (Markowitz)":
    st.markdown("<h1>Optimización de Portafolio (Markowitz)</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Frontera eficiente y portafolio de máximo Sharpe.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        tickers_input = st.text_input("Tickers (separados por comas)", "AAPL,MSFT,NVDA")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    with c2:
        range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("3 años"))

    period, interval = range_to_yf_params(range_key)
    prices = load_prices(tickers, period=period, interval=interval)

    if prices.empty or prices.shape[1] < 2:
        st.warning("Necesito al menos 2 tickers con datos para construir la frontera eficiente.")
    else:
        rets = prices.pct_change().dropna()
        mean_returns = rets.mean() * 252
        cov_matrix = rets.cov() * 252

        c1, c2 = st.columns(2)
        with c1:
            num_portfolios = st.slider("Número de portafolios simulados", 1000, 10000, 3000, step=500)
        with c2:
            rf = st.number_input("Tasa libre de riesgo (0.04 = 4%)", value=0.04, step=0.01)

        results = np.zeros((3, num_portfolios))  # [vol, ret, sharpe]
        weights_record = []
        rng = np.random.default_rng(42)

        for i in range(num_portfolios):
            w = rng.random(len(tickers))
            w /= w.sum()
            port_ret = np.dot(w, mean_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
            results[0, i] = port_vol
            results[1, i] = port_ret
            results[2, i] = sharpe
            weights_record.append(w)

        max_sharpe_idx = np.nanargmax(results[2])
        ms_vol, ms_ret, ms_sharpe = results[:, max_sharpe_idx]
        ms_weights = weights_record[max_sharpe_idx]

        st.markdown(f"**Mejor Sharpe**: {ms_sharpe:.2f} · **Rendimiento**: {ms_ret:.2%} · **Riesgo**: {ms_vol:.2%}")
        st.dataframe(
            pd.Series(ms_weights, index=tickers, name="Pesos óptimos (Máx. Sharpe)").to_frame().T,
            use_container_width=True
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results[0, :], y=results[1, :], mode="markers",
            marker=dict(color=results[2, :], colorscale="Viridis", showscale=True, colorbar_title="Sharpe"),
            name="Portafolios simulados",
            hovertemplate="Riesgo: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[ms_vol], y=[ms_ret],
            mode="markers+text", text=["Máx. Sharpe"], textposition="top center",
            marker=dict(color="red", size=10), name="Máx. Sharpe"
        ))
        fig.update_layout(
            title=f"Frontera Eficiente · {range_key}",
            xaxis_title="Riesgo (desv. estándar anualizada)",
            yaxis_title="Rendimiento esperado anualizado",
            template="simple_white", height=620, margin=dict(l=40, r=20, t=48, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 5) MONTE CARLO
# =====================================================
elif menu == "Simulación Monte Carlo":
    st.markdown("<h1>Simulación Monte Carlo</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Trayectorias de precio a 1 año basadas en μ y σ estimados del periodo.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        stonk = st.text_input("Símbolo", "AAPL").upper()
    with c2:
        range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("1 año"))
    with c3:
        simulations = st.slider("N.º de simulaciones", 50, 2000, 300, step=50)

    period, interval = range_to_yf_params(range_key)
    df = get_history(stonk, period, interval)

    if df.empty:
        st.warning("No hay datos para simular.")
    else:
        df_ret, ann_ret, ann_vol = return_metrics(df)
        S0 = df_ret["Close"].iloc[-1]
        T = 1
        N = 252

        np.random.seed(42)
        dt = T / N
        price_paths = np.zeros((N, simulations))
        price_paths[0] = S0

        for t in range(1, N):
            rand = np.random.standard_normal(simulations)
            price_paths[t] = price_paths[t-1] * np.exp(
                (ann_ret - 0.5 * ann_vol**2) * dt + ann_vol * np.sqrt(dt) * rand
            )

        fig = go.Figure()
        for i in range(simulations):
            fig.add_trace(go.Scatter(y=price_paths[:, i], mode="lines",
                                     line=dict(width=0.7), showlegend=False))
        fig.update_layout(
            title=f"Simulación Monte Carlo de {stonk} · μ={ann_ret:.2%}, σ={ann_vol:.2%} · {range_key}",
            xaxis_title="Días", yaxis_title="Precio simulado",
            template="simple_white", height=680, margin=dict(l=40, r=20, t=48, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="small">Proyecto académico – Ingeniería Financiera</div>', unsafe_allow_html=True)
