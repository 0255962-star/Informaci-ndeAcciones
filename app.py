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

st.sidebar.title("📘 Menú de Finanzas")
menu = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "📊 Consulta de Acciones",
        "📈 Tasas de Retorno",
        "📉 Riesgo de Inversión",
        "📘 CAPM",
        "📈 Optimización de Portafolio (Markowitz)",
        "🎲 Simulación Monte Carlo",
    ]
)

# =====================================================
# SELECTOR DE PERIODO (para TODAS las gráficas)
# =====================================================
RANGE_OPTIONS = ["1 semana", "1 mes", "6 meses", "1 año", "YTD", "3 años", "5 años"]

def range_to_yf_params(range_key: str):
    """
    Devuelve (period, interval) compatibles con yfinance.
    Usamos interval mayor para periodos largos para aligerar y hacer más legible.
    """
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
# FUNCIONES AUXILIARES
# =====================================================
@st.cache_data(ttl=3600)
def get_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Descarga histórico OHLCV (auto_adjust=True).
    Devuelve DataFrame indexado por fecha con columnas: Open, High, Low, Close, Volume.
    """
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
    """
    Calcula retornos simple y log, rendimiento y volatilidad anualizados.
    """
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
    """
    Devuelve precios de CIERRE ajustado por ticker (columnas=tickers).
    Soporta 1+ tickers y distintas formas de salida de yfinance.
    """
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
        # Multi-ticker
        if "Close" in raw.columns.levels[0]:
            prices = raw["Close"].copy()
        elif "Adj Close" in raw.columns.levels[0]:
            prices = raw["Adj Close"].copy()
        else:
            # Primer nivel disponible (defensivo)
            first = raw.columns.levels[0][0]
            prices = raw[first].copy()
    else:
        # Un ticker
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
# 1️⃣ CONSULTA DE ACCIONES
# =====================================================
if menu == "📊 Consulta de Acciones":
    st.title("📊 Consulta de Acciones")
    st.write("Visualiza información general, descripción y gráficos interactivos de la empresa.")

    st.write("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        stonk = st.text_input("Símbolo de la acción (ej. MSFT, AAPL, NVDA, TSLA)", "MSFT").strip().upper()
    with c2:
        range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("6 meses"))

    # Info de la empresa
    ticker = yf.Ticker(stonk)
    info = ticker.info if hasattr(ticker, "info") else {}

    st.subheader("🏢 Nombre de la empresa")
    st.write(info.get("longName", "No disponible"))

    st.subheader("📝 Descripción del negocio (inglés)")
    st.write(info.get("longBusinessSummary", "No disponible."))

    # Gráfica
    st.write("---")
    st.subheader("📈 Gráfica de Velas con Volumen")
    period, interval = range_to_yf_params(range_key)
    hist = get_history(stonk, period, interval)

    if hist.empty:
        st.warning("No se pudo obtener información histórica.")
    else:
        hist_sma = add_smas(hist, (20, 50, 200))

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(
            x=hist_sma.index, open=hist_sma["Open"], high=hist_sma["High"],
            low=hist_sma["Low"], close=hist_sma["Close"], name="OHLC",
            increasing_line_color="rgb(16,130,59)", increasing_fillcolor="rgba(16,130,59,0.9)",
            decreasing_line_color="rgb(200,30,30)", decreasing_fillcolor="rgba(200,30,30,0.9)",
            line=dict(width=1.25), whiskerwidth=0.3
        ), row=1, col=1)

        for col, color in zip(["SMA20", "SMA50", "SMA200"], ["#c218f0", "#ff9900", "#c0b000"]):
            fig.add_trace(go.Scatter(x=hist_sma.index, y=hist_sma[col], mode="lines",
                                     line=dict(color=color, width=1.5), name=col), row=1, col=1)

        colors = ["rgba(22,163,74,0.75)" if c >= o else "rgba(220,38,38,0.75)"
                  for o, c in zip(hist_sma["Open"], hist_sma["Close"])]
        fig.add_trace(go.Bar(x=hist_sma.index, y=hist_sma["Volume"],
                             marker_color=colors, name="Volumen"), row=2, col=1)

        fig.update_layout(height=750, title=f"{stonk} · {range_key}",
                          xaxis_rangeslider_visible=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 2️⃣ TASAS DE RETORNO
# =====================================================
elif menu == "📈 Tasas de Retorno":
    st.title("📈 Cálculo de Tasas de Retorno")
    st.write("Calcula **rendimientos simples, logarítmicos y anualizados** de una acción.")

    c1, c2 = st.columns([2, 1])
    with c1:
        stonk = st.text_input("Símbolo de la acción", "AAPL").upper()
    with c2:
        range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("1 año"))

    period, interval = range_to_yf_params(range_key)
    df = get_history(stonk, period, interval)

    if df.empty:
        st.warning("No se pudo obtener información.")
    else:
        df_ret, ann_ret, ann_vol = return_metrics(df)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("📈 Rendimiento anualizado", f"{ann_ret*100:.2f}%")
        with c2:
            st.metric("📉 Volatilidad anualizada", f"{ann_vol*100:.2f}%")

        st.subheader("📊 Retornos diarios")
        st.line_chart(df_ret["Return"])

        st.subheader("📊 Retornos acumulados")
        cumulative = (1 + df_ret["Return"]).cumprod() - 1
        st.area_chart(cumulative)

# =====================================================
# 3️⃣ RIESGO DE INVERSIÓN
# =====================================================
elif menu == "📉 Riesgo de Inversión":
    st.title("📉 Análisis de Riesgo y Volatilidad")
    st.write("Calcula **desviación estándar, varianza, beta y alpha** frente a un índice.")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        stonk = st.text_input("Símbolo de la acción", "NVDA").upper()
    with c2:
        market = st.text_input("Índice de referencia (ej. ^GSPC, ^IXIC)", "^GSPC").upper()
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
            st.metric("📊 Beta", f"{beta:.3f}")
        with c2:
            st.metric("⚙️ Varianza mercado", f"{var_market:.6f}")
        with c3:
            st.metric("💡 Alpha", f"{alpha:.4f}")

        st.subheader("Relación entre rendimientos (acción vs mercado)")
        st.scatter_chart(pd.DataFrame({"Stock": s_ret["Return"], "Market": m_ret["Return"]}).dropna())

# =====================================================
# 4️⃣ CAPM (con opción de estimar β desde datos)
# =====================================================
elif menu == "📘 CAPM":
    st.title("📘 Modelo CAPM - Capital Asset Pricing Model")
    st.write("Calcula el rendimiento esperado de un activo en función de su **β** y del **premio por riesgo**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        rf = st.number_input("Tasa libre de riesgo (0.04 = 4%)", value=0.04)
    with c2:
        rm = st.number_input("Rend. esperado del mercado (0.10 = 10%)", value=0.10)
    with c3:
        use_data_beta = st.toggle("Estimar β desde datos", value=False)

    if use_data_beta:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            stonk = st.text_input("Ticker del activo", "AAPL").upper()
        with c2:
            market = st.text_input("Benchmark (ej. ^GSPC)", "^GSPC").upper()
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
    st.metric("📈 Rendimiento esperado (CAPM)", f"{expected_return*100:.2f}%")

    # Línea SML
    betas = np.linspace(0, 2, 50)
    returns = rf + betas * (rm - rf)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=betas, y=returns, mode="lines", name="Línea SML"))
    fig.add_trace(go.Scatter(x=[beta], y=[expected_return],
                             mode="markers+text", text=["Tu activo"], textposition="bottom center",
                             marker=dict(size=10, color="red")))
    fig.update_layout(title="Security Market Line (SML)", xaxis_title="Beta", yaxis_title="Rendimiento esperado",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 5️⃣ MARKOWITZ (usa el mismo selector de periodo)
# =====================================================
elif menu == "📈 Optimización de Portafolio (Markowitz)":
    st.title("📈 Optimización de Portafolio - Modelo de Markowitz")
    st.write("Calcula la **frontera eficiente** con varios activos usando rendimientos y covarianzas anualizadas.")

    c1, c2 = st.columns([3, 1])
    with c1:
        tickers_input = st.text_input("Introduce tickers separados por comas", "AAPL,MSFT,NVDA")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    with c2:
        range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("3 años"))

    period, interval = range_to_yf_params(range_key)
    prices = load_prices(tickers, period=period, interval=interval)

    if prices.empty or prices.shape[1] < 2:
        st.warning("Necesito al menos **2 tickers** con datos para construir la frontera eficiente.")
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
            pd.Series(ms_weights, index=tickers, name="Peso óptimo (Máx. Sharpe)").to_frame().T,
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
            marker=dict(color="red", size=10)
        ))
        fig.update_layout(
            title=f"Frontera Eficiente (Markowitz) · {range_key}",
            xaxis_title="Riesgo (Desv. estándar anualizada)",
            yaxis_title="Rendimiento esperado anualizado",
            template="plotly_white", height=600
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 6️⃣ MONTE CARLO (usa el mismo selector para estimar μ y σ)
# =====================================================
elif menu == "🎲 Simulación Monte Carlo":
    st.title("🎲 Simulación Monte Carlo")
    st.write("Simula trayectorias de precios a 1 año con base en retorno y volatilidad anualizados del periodo elegido.")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        stonk = st.text_input("Símbolo de la acción a simular", "AAPL").upper()
    with c2:
        range_key = st.selectbox("Periodo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("1 año"))
    with c3:
        simulations = st.slider("N° de simulaciones", 50, 2000, 300, step=50)

    period, interval = range_to_yf_params(range_key)
    df = get_history(stonk, period, interval)

    if df.empty:
        st.warning("No hay datos para simular.")
    else:
        df_ret, ann_ret, ann_vol = return_metrics(df)
        S0 = df_ret["Close"].iloc[-1]
        T = 1  # 1 año
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
            fig.add_trace(go.Scatter(y=price_paths[:, i], mode="lines", line=dict(width=0.7), showlegend=False))
        fig.update_layout(title=f"Simulación Monte Carlo de {stonk} · μ={ann_ret:.2%}, σ={ann_vol:.2%} · {range_key}",
                          xaxis_title="Días", yaxis_title="Precio simulado",
                          template="plotly_white", height=700)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por **Alejandro Rodrigo Gascón de Alba** – Ingeniería Financiera UP 💼")
