import os
import json
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Consulta de Acciones - GASCON", page_icon="📊")

# =====================================================
# 🔐 GEMINI CONFIG
# =====================================================
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not API_KEY:
    st.warning("⚠️ Falta GOOGLE_API_KEY en Secrets (Gemini no disponible).")

BASES = [
    "https://generativelanguage.googleapis.com/v1",
    "https://generativelanguage.googleapis.com/v1beta",
]
PREFERRED = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.0-pro",
    "gemini-pro",
]

def list_models(base: str):
    try:
        r = requests.get(base + "/models", params={"key": API_KEY}, timeout=15)
        if r.status_code == 200:
            return True, r.json()
        return False, r.text
    except Exception as e:
        return False, str(e)

def pick_model_and_base():
    last_err = None
    for base in BASES:
        ok, data = list_models(base)
        if not ok:
            last_err = data
            continue
        models = data.get("models", [])
        supported = {m.get("name","").split("/")[-1]
                     for m in models
                     if any(x in ("generateContent","generate_content")
                            for x in (m.get("supportedGenerationMethods") or m.get("supported_generation_methods") or []))}
        for pref in PREFERRED:
            if pref in supported:
                return base, pref
        if supported:
            return base, sorted(supported)[0]
        last_err = "No hay modelos con generateContent."
    raise RuntimeError(f"Gemini no disponible: {last_err}")

def generate_content_rest(base: str, model: str, text: str) -> str:
    url = f"{base}/models/{model}:generateContent"
    payload = {"contents":[{"parts":[{"text": text}]}]}
    headers = {"Content-Type":"application/json"}
    r = requests.post(url, params={"key": API_KEY}, headers=headers, data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    data = r.json()
    for cand in data.get("candidates", []):
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        for p in parts:
            txt = (p.get("text") or "").strip()
            if txt:
                return txt
    return ""

def translate_to_spanish(text: str) -> str:
    if not text or not API_KEY:
        return ""
    try:
        base, model = pick_model_and_base()
        prompt = (
            "Traduce al español el siguiente texto sobre una empresa. "
            "Usa español claro y natural de negocios. Conserva nombres propios y números. "
            "Solo devuelve la traducción, sin explicaciones.\n\n" + text
        )
        return generate_content_rest(base, model, prompt)
    except Exception:
        return ""

# =====================================================
# 📦 YFINANCE HELPERS
# =====================================================
@st.cache_data(ttl=3600)
def get_info(symbol: str):
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def get_history(symbol: str, period="6mo", interval="1d"):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c.capitalize() if isinstance(c, str) else c for c in df.columns]
        return df.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception:
        return pd.DataFrame()

def range_to_yf_params(range_key: str):
    mapping = {
        "1 semana": ("7d", "1d"),
        "1 mes": ("1mo", "1d"),
        "6 meses": ("6mo", "1d"),
        "YTD": ("ytd", "1d"),
        "1 año": ("1y", "1d"),
        "3 años": ("3y", "1wk"),
        "5 años": ("5y", "1wk"),
    }
    return mapping.get(range_key, ("6mo", "1d"))

# =====================================================
# 🖥️ UI
# =====================================================
st.title("📊 Consulta de Acciones - MODELO FINANCIERO HUIZAR")

stonk = st.text_input("Ingresa el símbolo de la acción (ej. MSFT, NVDA, AAPL)", "MSFT").strip().upper()

# INFO BÁSICA
info = get_info(stonk)
st.subheader("🏢 Nombre de la empresa")
st.write(info.get("longName", "N/A"))

summary = info.get("longBusinessSummary", "")
st.subheader("📝 Descripción del negocio (inglés)")
st.write(summary if summary else "No hay descripción disponible.")

if summary and st.button("Traducir al español 🇪🇸", use_container_width=True):
    with st.spinner("Traduciendo con Gemini..."):
        translated = translate_to_spanish(summary)
        if translated:
            st.subheader("🇪🇸 Descripción del negocio (traducción)")
            st.write(translated)

# =====================================================
# 📈 GRÁFICA INTERACTIVA PLOTLY
# =====================================================
st.write("---")
st.subheader("📈 Gráfica de Velas (OHLC) con Volumen")

RANGE_OPTIONS = ["1 semana", "1 mes", "6 meses", "YTD", "1 año", "3 años", "5 años"]
range_key = st.selectbox("Rango de tiempo", RANGE_OPTIONS, index=RANGE_OPTIONS.index("6 meses"))

period, interval = range_to_yf_params(range_key)
hist = get_history(stonk, period, interval)

if hist.empty:
    st.warning("No se pudo obtener el historial de precios para graficar.")
else:
    # Calcular medias móviles
    hist["SMA20"] = hist["Close"].rolling(window=20).mean()
    hist["SMA50"] = hist["Close"].rolling(window=50).mean()
    hist["SMA200"] = hist["Close"].rolling(window=200).mean()

    # Crear figura con subgráfica de volumen
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Velas
    fig.add_trace(
        go.Candlestick(
            x=hist["Date"],
            open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"],
            name="OHLC",
            increasing_line_color="rgb(16,130,59)",
            increasing_fillcolor="rgba(16,130,59,0.9)",
            decreasing_line_color="rgb(200,30,30)",
            decreasing_fillcolor="rgba(200,30,30,0.9)",
            line=dict(width=1.25),
            whiskerwidth=0.3,
        ),
        row=1, col=1
    )

    # SMA lines
    for col, color in zip(["SMA20", "SMA50", "SMA200"], ["#ff00ff", "#ffa500", "#ffcc00"]):
        fig.add_trace(
            go.Scatter(
                x=hist["Date"], y=hist[col],
                mode="lines", line=dict(color=color, width=1.5),
                name=col
            ),
            row=1, col=1
        )

    # Volumen
    colors = ["rgba(22,163,74,0.75)" if c >= o else "rgba(220,38,38,0.75)"
              for o, c in zip(hist["Open"], hist["Close"])]
    fig.add_trace(
        go.Bar(
            x=hist["Date"], y=hist["Volume"], name="Volumen",
            marker_color=colors, opacity=0.8
        ),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        title=f"{stonk} · {range_key}",
        template="plotly_white",
        height=760,
        margin=dict(l=40, r=25, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="year", stepmode="todate", label="YTD"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
        ),
        yaxis=dict(title="Precio"),
        yaxis2=dict(title="Volumen"),
        dragmode="pan",
    )

    st.plotly_chart(fig, use_container_width=True)

