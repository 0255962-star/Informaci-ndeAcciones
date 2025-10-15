import os
import json
import requests
import streamlit as st
import yfinance as yf

# 📦 Data & plotting
import pandas as pd
from datetime import date

import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Consulta de Acciones - GASCON", page_icon="📊")

# =========================
# 🔐 API KEY (Gemini)
# =========================
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not API_KEY:
    st.error('Falta GOOGLE_API_KEY en Secrets. Pega exactamente:  GOOGLE_API_KEY = "TU_CLAVE_AQUI"')
    st.stop()

# =========================
# 🔧 Gemini endpoints/modelos
# =========================
BASES = [
    "https://generativelanguage.googleapis.com/v1",       # preferido
    "https://generativelanguage.googleapis.com/v1beta",   # fallback
]

PREFERRED = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.0-pro",
    "gemini-pro",
]

def list_models(base: str):
    """Devuelve (ok, models_json_o_error) del endpoint /models."""
    url = f"{base}/models"
    try:
        r = requests.get(url, params={"key": API_KEY}, timeout=20)
        if r.status_code == 200:
            return True, r.json()
        return False, r.json() if "application/json" in r.headers.get("content-type","") else r.text
    except Exception as e:
        return False, str(e)

def pick_model_and_base():
    """
    Intenta /v1 primero: lista modelos y elige uno que soporte generateContent.
    Si falla, intenta /v1beta. Devuelve (base, model_name).
    """
    last_err = None
    for base in BASES:
        ok, data = list_models(base)
        if not ok:
            last_err = data
            continue
        models = data.get("models", [])
        supported = set()
        for m in models:
            name = m.get("name")
            methods = m.get("supportedGenerationMethods") or m.get("supported_generation_methods") or []
            if not name:
                continue
            short = name.split("/")[-1]
            if any(x in ("generateContent", "generate_content") for x in methods):
                supported.add(short)
        for pref in PREFERRED:
            if pref in supported:
                return base, pref
        if supported:
            return base, sorted(supported)[0]
        last_err = "No hay modelos con generateContent en este endpoint."
    raise RuntimeError(f"No se pudo seleccionar modelo/base. Último error: {last_err}")

# Elegimos base y modelo válidos según tu API key
try:
    BASE, MODEL = pick_model_and_base()
    st.caption(f"Endpoint seleccionado: {BASE}  ·  Modelo: {MODEL}")
except Exception as e:
    st.error(f"No se pudo obtener la lista de modelos: {e}")
    st.stop()

def generate_content_rest(base: str, model: str, text: str) -> str:
    """Llama a :generateContent en el base/model dado, devuelve texto plano."""
    url = f"{base}/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": text}]}]}
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}
    r = requests.post(url, params=params, headers=headers, data=json.dumps(payload), timeout=30)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"{r.status_code} {err}")
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
    if not text:
        return ""
    prompt = (
        "Traduce al español el siguiente texto sobre una empresa. "
        "Usa español claro y natural de negocios. Conserva nombres propios y números. "
        "Solo devuelve la traducción, sin explicaciones.\n\n" + text
    )
    return generate_content_rest(BASE, MODEL, prompt)

# =========================
# 📦 yfinance cache
# =========================
@st.cache_data(ttl=3600)
def get_info(symbol: str):
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}

def _ytd_dates():
    today = date.today()
    start = date(today.year, 1, 1)
    return pd.Timestamp(start), pd.Timestamp(today)

RANGE_OPTIONS = [
    "1 semana",
    "1 mes",
    "6 meses",
    "YTD",
    "1 año",
    "3 años",
    "5 años",
]

def range_to_query(range_key: str):
    if range_key == "1 semana":
        return {"period": "7d", "interval": "1d"}
    if range_key == "1 mes":
        return {"period": "1mo", "interval": "1d"}
    if range_key == "6 meses":
        return {"period": "6mo", "interval": "1d"}
    if range_key == "YTD":
        start, end = _ytd_dates()
        return {"start": start, "end": end, "interval": "1d"}
    if range_key == "1 año":
        return {"period": "1y", "interval": "1d"}
    if range_key == "3 años":
        return {"period": "3y", "interval": "1d"}
    if range_key == "5 años":
        return {"period": "5y", "interval": "1d"}
    return {"period": "6mo", "interval": "1d"}  # fallback seguro

@st.cache_data(ttl=3600)
def get_history(symbol: str, range_key: str) -> pd.DataFrame:
    """Descarga OHLCV y devuelve DataFrame con índice DatetimeIndex."""
    q = range_to_query(range_key)
    try:
        t = yf.Ticker(symbol)
        if "period" in q:
            df = t.history(period=q["period"], interval=q["interval"], auto_adjust=False)
            if df is None or df.empty:
                df = yf.download(symbol, period=q["period"], interval=q["interval"], auto_adjust=False, progress=False, threads=False)
        else:
            df = t.history(start=q["start"], end=q["end"], interval=q["interval"], auto_adjust=False)
            if df is None or df.empty:
                df = yf.download(symbol, start=q["start"], end=q["end"], interval=q["interval"], auto_adjust=False, progress=False, threads=False)

        if df is None or df.empty:
            return pd.DataFrame()

        # Aplana MultiIndex si aparece
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Normaliza índice
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index(pd.to_datetime(df["Date"])).drop(columns=["Date"])
            else:
                df.index = pd.to_datetime(df.index)

        # Orden y columnas clave
        df = df.sort_index()
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        out = df[keep].dropna(subset=["Open", "High", "Low", "Close"])
        return out
    except Exception:
        return pd.DataFrame()

# =========================
# 🖥️ UI
# =========================
st.title("📊 Consulta de Acciones - MODELO FINANCIERO HUIZAR")

# Entrada de símbolo
stonk = st.text_input("Ingresa el símbolo de la acción", "MSFT").strip().upper()

# Reset traducción si cambia símbolo
if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = stonk
if st.session_state.last_symbol != stonk:
    st.session_state.pop("translated_es", None)
    st.session_state.last_symbol = stonk

# Bloque de info básica
info = get_info(stonk)

st.subheader("🏢 Nombre de la empresa")
st.write(info.get("longName", "N/A"))

summary = info.get("longBusinessSummary", "")
st.subheader("📝 Descripción del negocio (original / inglés)")
st.write(summary if summary else "No hay descripción disponible.")

st.write("---")
if st.button("Traducir a español 🇪🇸", use_container_width=True):
    if not summary:
        st.warning("No hay descripción para traducir.")
    else:
        with st.spinner("Traduciendo con Gemini..."):
            try:
                st.session_state.translated_es = translate_to_spanish(summary)
            except Exception as e:
                st.error(f"Error al traducir con Gemini: {e}")

if st.session_state.get("translated_es"):
    st.subheader("🇪🇸 Descripción del negocio (traducción)")
    st.write(st.session_state.translated_es)

# =========================
# 📈 Gráfica de Velas (Plotly) + Volumen
# =========================
st.write("---")
st.subheader("📈 Gráfica de Velas (OHLC) con Volumen")

# Selector de rango temporal (actualiza automáticamente al cambiar)
default_range = "6 meses"
range_key = st.selectbox(
    "Rango de tiempo",
    RANGE_OPTIONS,
    index=RANGE_OPTIONS.index(default_range),
    help="Selecciona el periodo para la gráfica de velas. La vista se actualiza al instante."
)

hist = get_history(stonk, range_key)

if hist.empty:
    st.warning("No se pudo obtener el historial para graficar.")
else:
    # Crear figura con eje secundario para volumen
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.75, 0.25]
    )

    # Velas
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="OHLC"
        ),
        row=1, col=1
    )

    # Volumen
    if "Volume" in hist.columns:
        fig.add_trace(
            go.Bar(
                x=hist.index,
                y=hist["Volume"],
                name="Volumen",
                opacity=0.6
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Volumen", row=2, col=1)

    # Layout: título, sliders, rangeselector
    fig.update_layout(
        title=f"{stonk} · {range_key} (Velas)",
        margin=dict(l=40, r=20, t=60, b=40),
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
                    dict(step="all", label="All")
                ]
            ),
        ),
        yaxis=dict(title="Precio"),
        showlegend=False,
        height=700
    )

    # Mejoras visuales para velas
    fig.update_traces(
        selector=dict(type="candlestick"),
        increasing_line_color="#16a34a",  # verde
        decreasing_line_color="#dc2626"   # rojo
    )

    st.plotly_chart(fig, use_container_width=True)


