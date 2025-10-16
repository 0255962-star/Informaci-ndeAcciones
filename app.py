import os
import json
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

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
# FUNCIONES AUXILIARES
# =====================================================
@st.cache_data(ttl=3600)
def get_history(symbol, period="1y"):
    """
    Histórico de un solo ticker con auto_adjust=True.
    Devuelve OHLCV con índice de fecha.
    """
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    return df.dropna()

def get_return_metrics(df):
    """
    Calcula retornos simple y log, retorno y volatilidad anualizados.
    Devuelve (df_con_retornos, annual_return, volatility).
    """
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    avg_daily = df["Return"].mean()
    annual_return = (1 + avg_daily) ** 252 - 1
    volatility = df["Return"].std() * np.sqrt(252)
    return df.dropna(), annual_return, volatility

@st.cache_data(ttl=1800)
def load_prices(tickers, period="1y"):
    """
    Devuelve un DataFrame de precios de CIERRE (ajustado) por ticker (columnas=tickers).
    Soporta 1+ tickers y distintas formas de salida de yfinance.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t for t in tickers if t]

    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers, period=period, auto_adjust=True,
        progress=False, threads=False
    )
    if raw.empty:
        return pd.DataFrame()

    # Múltiples tickers → columnas MultiIndex (campo -> ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.levels[0]:
            prices = raw["Close"].copy()
        elif "Adj Close" in raw.columns.levels[0]:
            prices = raw["Adj Close"].copy()
        else:
            # Último recurso: primer nivel disponible
            first_level = raw.columns.levels[0][0]
            prices = raw[first_level].copy()
    else:
        # Un solo ticker → columnas planas
        if "Close" in raw.columns:
            prices = raw[["Close"]].copy()
        elif "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy().rename(columns={"Adj Close": "Close"})
        else:
            return pd.DataFrame()
        prices.columns = [tickers[0]]

    return prices.dropna(how="all")

# =====================================================
# 1️⃣ CONSULTA DE ACCIONES
# =====================================================
if menu == "📊 Consulta de Acciones":
    st.title("📊 Consulta de Acciones")
    st.write("Visualiza información general, descripción y gráficos interactivos de la empresa.")

    st.write("---")
    st.subheader("🔍 Buscar acción")
    stonk = st.text_input("Símbolo de la acción (ej. MSFT, AAPL, NVDA, TSLA)", "MSFT").strip().upper()

    ticker = yf.Ticker(stonk)
    info = ticker.info if hasattr(ticker, "info") else {}

    st.subheader("🏢 Nombre de la empresa")
    st.write(info.get("longName", "No disponible"))

    st.subheader("📝 Descripción del negocio (inglés)")
    st.write(info.get("longBusinessSummary", "No disponible."))

    st.write("---")
    st.subheader("📈 Gráfica de Velas con Volumen")
    hist = get_history(stonk, "6mo")

    if hist.empty:
        st.warning("No se pudo obtener información histórica.")
    else:
        # SMAs
        hist = hist.copy()
        hist["SMA20"] = hist["Close"].rolling(window=20).mean()
        hist["SMA50"] = hist["Close"].rolling(window=50).mean()
        hist["SMA200"] = hist["Close"].rolling(window=200).mean()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"],
            name="OHLC",
            increasing_line_color="rgb(16,130,59)",
            increasing_fillcolor="rgba(16,130,59,0.9)",
            decreasing_line_color="rgb(200,30,30)",
            decreasing_fillcolor="rgba(200,30,30,0.9)",
            line=dict(width=1.25), whiskerwidth=0.3
        ), row=1, col=1)

        for col, color in zip(["SMA20", "SMA50", "SMA200"], ["#ff00ff", "#ffa500", "#ffcc00"]):
            fig.add_trace(go.Scatter(x=hist.index, y=hist[col], mode="lines",
                                     line=dict(color=color, width=1.5), name=col), row=1, col=1)

        colors = ["rgba(22,163,74,0.75)" if c >= o else "rgba(220,38,38,0.75)"
                  for o, c in zip(hist["Open"], hist["Close"])]
        fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"],
                             marker_color=colors, name="Volumen"), row=2, col=1)

        fig.update_layout(height=750, title=f"Histórico de {stonk}",
                          xaxis_rangeslider_visible=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 2️⃣ TASAS DE RETORNO
# =====================================================
elif menu == "📈 Tasas de Retorno":
    st.title("📈 Cálculo de Tasas de Retorno")
    st.write("Calcula **rendimientos simples, logarítmicos y anualizados** de una acción.")

    stonk = st.text_input("Símbolo de la acción", "AAPL").upper()
    df = get_history(stonk, "1y")

    if df.empty:
        st.warning("No se pudo obtener información.")
    else:
        df, annual_return, volatility = get_return_metrics(df)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("📈 Rendimiento anualizado", f"{annual_return*100:.2f}%")
        with c2:
            st.metric("📉 Volatilidad anualizada", f"{volatility*100:.2f}%")

        st.subheader("📊 Retornos diarios")
        st.line_chart(df["Return"])

        st.subheader("📊 Retornos acumulados")
        cumulative = (1 + df["Return"]).cumprod() - 1
        st.area_chart(cumulative)

# =====================================================
# 3️⃣ RIESGO DE INVERSIÓN
# =====================================================
elif menu == "📉 Riesgo de Inversión":
    st.title("📉 Análisis de Riesgo y Volatilidad")
    st.write("Calcula **desviación estándar, varianza, beta y alpha** frente a un índice.")

    col1, col2 = st.columns(2)
    with col1:
        stonk = st.text_input("Símbolo de la acción", "NVDA").upper()
    with col2:
        market = st.text_input("Índice de referencia (ej. ^GSPC, ^IXIC)", "^GSPC").upper()

    df_stock = get_history(stonk, "1y")
    df_market = get_history(market, "1y")

    if df_stock.empty or df_market.empty:
        st.warning("No se pudieron obtener los datos.")
    else:
        df_stock, _, _ = get_return_metrics(df_stock)
        df_market, _, _ = get_return_metrics(df_market)

        merged = pd.merge(df_stock["Return"], df_market["Return"],
                          left_index=True, right_index=True, suffixes=("_stock", "_market"))
        cov = np.cov(merged["Return_stock"], merged["Return_market"])[0][1]
        var_market = merged["Return_market"].var()
        beta = cov / var_market if var_market != 0 else np.nan
        alpha = merged["Return_stock"].mean() - beta * merged["Return_market"].mean()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("📊 Beta", f"{beta:.3f}")
        with c2:
            st.metric("⚙️ Varianza mercado", f"{var_market:.6f}")
        with c3:
            st.metric("💡 Alpha", f"{alpha:.4f}")

        st.subheader("Relación entre rendimientos")
        st.scatter_chart(merged)

# =====================================================
# 4️⃣ CAPM
# =====================================================
elif menu == "📘 CAPM":
    st.title("📘 Modelo CAPM - Capital Asset Pricing Model")
    st.write("Calcula el rendimiento esperado de un activo en función de su beta y del premio por riesgo.")

    rf = st.number_input("Tasa libre de riesgo (ej. 0.04 = 4%)", value=0.04)
    rm = st.number_input("Rendimiento esperado del mercado (ej. 0.10 = 10%)", value=0.10)
    beta = st.number_input("Beta del activo", value=1.2)

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
    fig.update_layout(title="Security Market Line (SML)", xaxis_title="Beta", yaxis_title="Rendimiento esperado")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 5️⃣ MARKOWITZ (con fix load_prices)
# =====================================================
elif menu == "📈 Optimización de Portafolio (Markowitz)":
    st.title("📈 Optimización de Portafolio - Modelo de Markowitz")
    st.write("Calcula la **frontera eficiente** con varios activos usando rendimientos y covarianzas anualizadas.")

    tickers_input = st.text_input("Introduce tickers separados por comas", "AAPL,MSFT,NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    period = st.selectbox("Periodo de datos", ["1y", "3y", "5y"], index=1)

    prices = load_prices(tickers, period=period)

    if prices.empty or prices.shape[1] < 2:
        st.warning("Necesito al menos **2 tickers** con datos para construir la frontera eficiente.")
    else:
        # Rendimientos diarios y anualizados
        rets = prices.pct_change().dropna()
        mean_returns = rets.mean() * 252
        cov_matrix = rets.cov() * 252

        num_portfolios = st.slider("Número de portafolios simulados", 1000, 10000, 3000, step=500)
        rf = st.number_input("Tasa libre de riesgo (p.ej., 0.04 = 4%)", value=0.04, step=0.01)

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
            title="Frontera Eficiente (Markowitz)",
            xaxis_title="Riesgo (Desviación estándar anualizada)",
            yaxis_title="Rendimiento esperado anualizado",
            template="plotly_white", height=600
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 6️⃣ MONTE CARLO
# =====================================================
elif menu == "🎲 Simulación Monte Carlo":
    st.title("🎲 Simulación Monte Carlo")
    st.write("Simula trayectorias de precios a 1 año con base en retorno y volatilidad anualizados.")

    stonk = st.text_input("Símbolo de la acción a simular", "AAPL").upper()
    df = get_history(stonk, "1y")

    if df.empty:
        st.warning("No hay datos para simular.")
    else:
        df, annual_return, volatility = get_return_metrics(df)
        S0 = df["Close"].iloc[-1]
        T = 1  # 1 año
        N = 252
        simulations = st.slider("N° de simulaciones", 50, 2000, 300, step=50)

        np.random.seed(42)
        dt = T / N
        price_paths = np.zeros((N, simulations))
        price_paths[0] = S0

        for t in range(1, N):
            rand = np.random.standard_normal(simulations)
            price_paths[t] = price_paths[t-1] * np.exp(
                (annual_return - 0.5 * volatility**2)*dt + volatility*np.sqrt(dt)*rand
            )

        fig = go.Figure()
        for i in range(simulations):
            fig.add_trace(go.Scatter(y=price_paths[:, i], mode="lines", line=dict(width=0.7), showlegend=False))
        fig.update_layout(title=f"Simulación Monte Carlo de {stonk}",
                          xaxis_title="Días", yaxis_title="Precio simulado",
                          template="plotly_white", height=700)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por **Alejandro Rodrigo Gascón de Alba** – Ingeniería Financiera UP 💼")
