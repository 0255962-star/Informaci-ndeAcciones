import os
import json
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Modelo Financiero Huizar", page_icon="📊", layout="wide")

# ========================= GEMINI CONFIG =========================
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# ========================= FUNCIONES DE AYUDA =========================
@st.cache_data(ttl=3600)
def get_info(symbol: str):
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def get_history(symbol: str, period="6mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        df = df.reset_index()
        return df.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception:
        return pd.DataFrame()

# ========================= INTERFAZ PRINCIPAL =========================
st.sidebar.title("📘 Menú de Finanzas")
menu = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "📊 Consulta de Acciones",
        "📈 Tasas de Retorno",
        "📉 Riesgo de Inversión",
        "📊 Regresiones Financieras",
        "📈 Optimización de Portafolio (Markowitz)",
        "📘 CAPM",
        "🎲 Simulación Monte Carlo",
    ]
)

# ========================= 1. CONSULTA DE ACCIONES =========================
if menu == "📊 Consulta de Acciones":
    st.title("📊 Consulta de Acciones - MODELO FINANCIERO HUIZAR")
    st.write("Visualiza datos bursátiles, resumen de empresa y gráfica de velas.")

    st.write("---")
    st.subheader("🔍 Buscar acción")
    stonk = st.text_input("Símbolo de la acción (ej. MSFT, AAPL, NVDA)", "MSFT").strip().upper()

    info = get_info(stonk)
    st.subheader("🏢 Nombre de la empresa")
    st.write(info.get("longName", "N/A"))

    summary = info.get("longBusinessSummary", "")
    st.subheader("📝 Descripción del negocio (inglés)")
    st.write(summary if summary else "No hay descripción disponible.")

    hist = get_history(stonk, period="6mo", interval="1d")

    if not hist.empty:
        hist["SMA20"] = hist["Close"].rolling(window=20).mean()
        hist["SMA50"] = hist["Close"].rolling(window=50).mean()
        hist["SMA200"] = hist["Close"].rolling(window=200).mean()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

        fig.add_trace(
            go.Candlestick(
                x=hist["Date"], open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"],
                name="OHLC",
                increasing_line_color="green", decreasing_line_color="red"
            ),
            row=1, col=1
        )

        for col, color in zip(["SMA20", "SMA50", "SMA200"], ["#ff00ff", "#ffa500", "#ffcc00"]):
            fig.add_trace(
                go.Scatter(x=hist["Date"], y=hist[col], mode="lines", line=dict(color=color, width=1.5), name=col),
                row=1, col=1
            )

        colors = ["rgba(22,163,74,0.75)" if c >= o else "rgba(220,38,38,0.75)"
                  for o, c in zip(hist["Open"], hist["Close"])]
        fig.add_trace(go.Bar(x=hist["Date"], y=hist["Volume"], marker_color=colors, name="Volumen"), row=2, col=1)

        fig.update_layout(height=750, title=f"Histórico de {stonk}", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se pudo obtener información del historial.")

# ========================= 2. TASAS DE RETORNO =========================
elif menu == "📈 Tasas de Retorno":
    st.title("📈 Cálculo de Tasas de Retorno")
    st.write("""
    Aquí calcularemos **rendimientos simples, logarítmicos y anuales**.
    En la siguiente versión te integraré un ejemplo completo con gráficas y fórmulas.
    """)

# ========================= 3. RIESGO DE INVERSIÓN =========================
elif menu == "📉 Riesgo de Inversión":
    st.title("📉 Riesgo y Volatilidad")
    st.write("""
    Este módulo medirá la **desviación estándar, varianza, y beta**.
    Ideal para comparar riesgo entre acciones.
    """)

# ========================= 4. REGRESIONES FINANCIERAS =========================
elif menu == "📊 Regresiones Financieras":
    st.title("📊 Regresiones Financieras")
    st.write("""
    Aquí aplicaremos regresiones lineales y multivariadas para análisis financiero.
    """)

# ========================= 5. OPTIMIZACIÓN DE PORTAFOLIO =========================
elif menu == "📈 Optimización de Portafolio (Markowitz)":
    st.title("📈 Optimización de Portafolio - Modelo de Markowitz")
    st.write("""
    Este módulo mostrará la **frontera eficiente** y el cálculo de portafolios óptimos.
    """)

# ========================= 6. CAPM =========================
elif menu == "📘 CAPM":
    st.title("📘 Capital Asset Pricing Model (CAPM)")
    st.write("""
    Aquí implementaremos el **modelo CAPM** para calcular rendimientos esperados 
    en función del riesgo sistemático (beta).
    """)

# ========================= 7. SIMULACIÓN MONTE CARLO =========================
elif menu == "🎲 Simulación Monte Carlo":
    st.title("🎲 Simulación Monte Carlo")
    st.write("""
    Aquí realizaremos simulaciones aleatorias para predecir el valor futuro de activos financieros.
    """)

# ========================= FOOTER =========================
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por **Alejandro Rodrigo Gascón de Alba** – Ingeniería Financiera UP 💼")

