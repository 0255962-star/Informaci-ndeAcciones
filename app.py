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
st.set_page_config(page_title="Modelo Financiero Huizar", page_icon="📊", layout="wide")

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
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    return df.dropna()

def get_return_metrics(df):
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    avg_daily = df["Return"].mean()
    annual_return = (1 + avg_daily) ** 252 - 1
    volatility = df["Return"].std() * np.sqrt(252)
    return df.dropna(), annual_return, volatility

# =====================================================
# 1️⃣ CONSULTA DE ACCIONES
# =====================================================
if menu == "📊 Consulta de Acciones":
    st.title("📊 Consulta de Acciones - MODELO FINANCIERO HUIZAR")
    st.write("Visualiza información general, descripción y gráficos interactivos de la empresa.")

    st.write("---")
    st.subheader("🔍 Buscar acción")
    stonk = st.text_input("Símbolo de la acción (ej. MSFT, AAPL, NVDA, TSLA)", "MSFT").strip().upper()

    ticker = yf.Ticker(stonk)
    info = ticker.info

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
        hist["SMA20"] = hist["Close"].rolling(window=20).mean()
        hist["SMA50"] = hist["Close"].rolling(window=50).mean()
        hist["SMA200"] = hist["Close"].rolling(window=200).mean()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"],
            name="OHLC", increasing_line_color="green", decreasing_line_color="red"
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

    st.write("""
    Calcula y analiza los **rendimientos simples, logarítmicos y anualizados**
    de una acción en distintos periodos.
    """)

    stonk = st.text_input("Símbolo de la acción", "AAPL").upper()
    df = get_history(stonk, "1y")

    if df.empty:
        st.warning("No se pudo obtener información.")
    else:
        df, annual_return, volatility = get_return_metrics(df)

        st.metric("📈 Rendimiento anualizado", f"{annual_return*100:.2f}%")
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

    st.write("""
    Calcula la **desviación estándar, varianza y beta** de una acción frente al índice de mercado.
    """)

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
        beta = cov / var_market
        alpha = merged["Return_stock"].mean() - beta * merged["Return_market"].mean()

        st.metric("📊 Beta", f"{beta:.3f}")
        st.metric("⚙️ Varianza del mercado", f"{var_market:.6f}")
        st.metric("💡 Alpha", f"{alpha:.4f}")

        st.subheader("Relación entre rendimientos")
        st.scatter_chart(merged)

# =====================================================
# 4️⃣ CAPM
# =====================================================
elif menu == "📘 CAPM":
    st.title("📘 Modelo CAPM - Capital Asset Pricing Model")

    st.write("""
    Calcula el rendimiento esperado de un activo en función de su beta y del premio por riesgo del mercado.
    """)

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
# 5️⃣ MARKOWITZ
# =====================================================
elif menu == "📈 Optimización de Portafolio (Markowitz)":
    st.title("📈 Optimización de Portafolio - Modelo de Markowitz")

    st.write("""
    Calcula la **frontera eficiente** de un portafolio a partir de varios activos.
    """)

    tickers = st.text_input("Introduce tickers separados por comas", "AAPL,MSFT,NVDA").split(",")
    tickers = [t.strip().upper() for t in tickers]

    data = yf.download(tickers, period="1y")["Adj Close"].dropna()
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    num_portfolios = 3000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i] = portfolio_std
        results[1,i] = portfolio_return
        results[2,i] = results[1,i] / results[0,i]
        weights_record.append(weights)

    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_ratio = results[:, max_sharpe_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results[0,:], y=results[1,:], mode="markers",
        marker=dict(color=results[2,:], colorscale="Viridis", showscale=True),
        text=[f"Sharpe: {s:.2f}" for s in results[2,:]],
        name="Portafolios simulados"
    ))
    fig.add_trace(go.Scatter(
        x=[max_sharpe_ratio[0]], y=[max_sharpe_ratio[1]],
        mode="markers+text", text=["Máx. Sharpe"], textposition="top center",
        marker=dict(color="red", size=10)
    ))
    fig.update_layout(title="Frontera Eficiente (Markowitz)",
                      xaxis_title="Riesgo (Desv. estándar)",
                      yaxis_title="Rendimiento esperado",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 6️⃣ MONTE CARLO
# =====================================================
elif menu == "🎲 Simulación Monte Carlo":
    st.title("🎲 Simulación Monte Carlo")

    st.write("""
    Simula escenarios posibles de precios futuros con base en la volatilidad y rendimientos esperados.
    """)

    stonk = st.text_input("Símbolo de la acción a simular", "AAPL").upper()
    df = get_history(stonk, "1y")

    if df.empty:
        st.warning("No hay datos para simular.")
    else:
        df, annual_return, volatility = get_return_metrics(df)
        S0 = df["Close"].iloc[-1]
        T = 1  # 1 año
        N = 252
        simulations = 200

        np.random.seed(42)
        dt = T/N
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
