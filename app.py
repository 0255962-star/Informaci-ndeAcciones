import os
import json
import requests
import streamlit as st
import yfinance as yf

# 🔹 NUEVO: data/plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 🔹 Candlestick (interactivo) + indicadores
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 🔹 TA-Lib (con fallback a pandas-ta)
try:
    import talib as ta
    TA_BACKEND = "talib"
except Exception:
    ta = None
    try:
        import pandas_ta as pta
        TA_BACKEND = "pandas-ta"
    except Exception:
        pta = None
        TA_BACKEND = None

st.set_page_config(page_title="Consulta de Acciones - GASCON", page_icon="📊")

API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not API_KEY:
    st.error('Falta GOOGLE_API_KEY en Secrets. Pega exactamente:  GOOGLE_API_KEY = "TU_CLAVE_AQUI"')
    st.stop()

# Endpoints a probar (list y generate)
BASES = [
    "https://generativelanguage.googleapis.com/v1",       # preferido
    "https://generativelanguage.googleapis.com/v1beta",   # fallback
]

# Preferencias de modelos por calidad/costo; usaremos el primero que exista y soporte generateContent
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

# --------- yfinance cache ----------
@st.cache_data(ttl=3600)
def get_info(symbol: str):
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}

# 🔧 Historial robusto (history() -> fallback download, aplanar MultiIndex, normalizar 'Date')
@st.cache_data(ttl=3600)
def get_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)

        if df is None or df.empty:
            return pd.DataFrame()

        # Aplanar MultiIndex si aparece
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        df = df.reset_index()

        # Normalizar columna temporal
        if "Date" not in df.columns:
            if "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            elif "date" in df.columns:
                df = df.rename(columns={"date": "Date"})

        desired = ["Date", "Open", "High", "Low", "Close", "Volume"]
        cols = [c for c in desired if c in df.columns]
        out = df[cols].dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in df.columns])
        return out
    except Exception:
        return pd.DataFrame()

# --------- UI ----------
st.title("📊 Consulta de Acciones - MODELO FINANCIERO HUIZAR")

stonk = st.text_input("Ingresa el símbolo de la acción", "MSFT").strip().upper()

# Reset traducción si cambia símbolo
if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = stonk
if st.session_state.last_symbol != stonk:
    st.session_state.pop("translated_es", None)
    st.session_state.last_symbol = stonk

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

# ─────────────────────────────────────────────────────────────
# 📈 Gráfica seaborn (OHLC + Volumen) debajo de la traducción (lo ya agregado antes)
# ─────────────────────────────────────────────────────────────
st.write("---")
st.subheader("📈 Historial de precios (Open, High, Low, Close) y Volumen")

hist = get_history(stonk, period="6mo", interval="1d")

if hist.empty or not {"Open", "High", "Low", "Close", "Volume"}.issubset(set(hist.columns)):
    st.warning("No se pudo obtener el historial para graficar.")
    if not hist.empty:
        st.caption(f"Columnas recibidas: {list(hist.columns)}")
else:
    sns.set_theme(style="whitegrid")
    fig, (ax_price, ax_vol) = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    sns.lineplot(data=hist, x="Date", y="Open", ax=ax_price, label="Open")
    sns.lineplot(data=hist, x="Date", y="High", ax=ax_price, label="High")
    sns.lineplot(data=hist, x="Date", y="Low", ax=ax_price, label="Low")
    sns.lineplot(data=hist, x="Date", y="Close", ax=ax_price, label="Close")
    ax_price.set_title(f"{stonk} · Precios diarios (últimos 6 meses)")
    ax_price.set_xlabel("")
    ax_price.set_ylabel("Precio")
    ax_price.legend(loc="upper left")
    sns.lineplot(data=hist, x="Date", y="Volume", ax=ax_vol, label="Volume")
    ax_vol.set_title("Volumen diario")
    ax_vol.set_xlabel("Fecha")
    ax_vol.set_ylabel("Volumen")
    ax_vol.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

# ─────────────────────────────────────────────────────────────
# 🕯️ NUEVO: Candlestick con Plotly + RSI usando TA-Lib (o pandas-ta si no hay TA-Lib)
# ─────────────────────────────────────────────────────────────
st.write("---")
st.subheader("🕯️ Gráfica de velas (candlestick) + RSI")

if hist.empty or not {"Open", "High", "Low", "Close", "Volume"}.issubset(set(hist.columns)):
    st.info("No hay datos suficientes para la gráfica de velas.")
else:
    d = hist.dropna().copy()

    # Calcular RSI (TA-Lib preferido, si no, pandas-ta; si nada, sin RSI)
    rsi = None
    try:
        if TA_BACKEND == "talib":
            rsi = ta.RSI(d["Close"].astype(float), timeperiod=14)
        elif TA_BACKEND == "pandas-ta":
            rsi = pta.rsi(d["Close"].astype(float), length=14)
    except Exception:
        rsi = None

    # Crear subplots: 3 paneles (velas, volumen, RSI si disponible)
    rows = 3 if rsi is not None else 2
    row_heights = [0.6, 0.25, 0.15] if rsi is not None else [0.75, 0.25]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights
    )

    # Panel 1: Candlestick
    fig.add_trace(
        go.Candlestick(
            x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="OHLC"
        ),
        row=1, col=1
    )

    # Panel 2: Volumen (barras)
    fig.add_trace(
        go.Bar(x=d["Date"], y=d["Volume"], name="Volumen"),
        row=2, col=1
    )

    # Panel 3: RSI (si existe)
    if rsi is not None:
        fig.add_trace(
            go.Scatter(x=d["Date"], y=rsi, mode="lines", name="RSI(14)"),
            row=3, col=1
        )
        # Líneas guía RSI
        fig.add_hline(y=70, line_dash="dash", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_width=1, row=3, col=1)

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Aviso del backend usado para RSI
    if TA_BACKEND is None:
        st.caption("RSI no mostrado: instala TA-Lib o pandas-ta para calcular indicadores.")
    else:
        st.caption(f"Indicadores calculados con: {TA_BACKEND}")

