import os
import json
import requests
import streamlit as st
import yfinance as yf

# Data & plotting
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
    url = f"{base}/models"
    try:
        r = requests.get(url, params={"key": API_KEY}, timeout=20)
        if r.status_code == 200:
            return True, r.json()
        return False, r.json() if "application/json" in r.headers.get("content-type","") else r.text
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

try:
    BASE, MODEL = pick_model_and_base()
    st.caption(f"Endpoint seleccionado: {BASE}  ·  Modelo: {MODEL}")
except Exception as e:
    st.error(f"No se pudo obtener la lista de modelos: {e}")
    st.stop()

def generate_content_rest(base: str, model: str, text: str) -> str:
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
    return {"period": "6mo", "interval": "1d"}  # fallback

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

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index(pd.to_datetime(df["Date"])).drop(columns=["Date"])
            else:
                df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        out = df[keep].dropna(subset=["Open", "High", "Low", "Close"])
        return out
    except Exception:
        return pd.DataFrame()

# ---------- Utilidades de legibilidad ----------
def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Re-muestrea a OHLCV con regla pandas (e.g., 'W', 'M') para despejar velas."""
    if df.empty:
        return df
    res = pd.DataFrame()
    res["Open"] = df["Open"].resample(rule).first()
    res["High"] = df["High"].resample(rule).max()
    res["Low"] = df["Low"].resample(rule).min()
    res["Close"] = df["Close"].resample(rule).last()
    if "Volume" in df.columns:
        res["Volume"] = df["Volume"].resample(rule).sum()
    res = res.dropna(subset=["Open", "High", "Low", "Close"])
    return res

def density_aware_downsample(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Elige automáticamente W/M si hay demasiados puntos; devuelve (df_res, etiqueta)."""
    n = len(df)
    if n > 800:
        return resample_ohlc(df, "M"), " (mensual)"
    if n > 300:
        return resample_ohlc(df, "W"), " (semanal)"
    return df, ""

def add_moving_averages(df: pd.DataFrame, windows=(20, 50)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"SMA{w}"] = out["Close"].rolling(w).mean()
    return out

def volume_colors(df: pd.DataFrame):
    """Colores para volumen según dirección de cierre vs apertura."""
    up_color = "rgba(22,163,74,0.7)"   # verde
    down_color = "rgba(220,38,38,0.7)" # rojo
    return [up_color if c >= o else down_color for o, c in zip(df["Open"], df["Close"])]

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
# 📈 Gráfica de Velas (Plotly) mejorada
# =========================
st.write("---")
st.subheader("📈 Velas OHLC con Volumen (alta legibilidad)")

# Controles
col1, col2, col3 = st.columns([1,1,1])
with col1:
    default_range = "6 meses"
    range_key = st.selectbox(
        "Rango de tiempo",
        RANGE_OPTIONS,
        index=RANGE_OPTIONS.index(default_range),
        help="La vista se actualiza al instante."
    )
with col2:
    auto_downsample = st.toggle("Agrupar automáticamente (W/M) si hay exceso de velas)", value=True)
with col3:
    show_sma = st.toggle("Mostrar SMA 20/50", value=True)

hist = get_history(stonk, range_key)

if hist.empty:
    st.warning("No se pudo obtener el historial para graficar.")
else:
    # Downsample automático
    suffix = ""
    df_plot = hist.copy()
    if auto_downsample:
        df_plot, suffix = density_aware_downsample(df_plot)

    # SMAs
    if show_sma:
        df_plot = add_moving_averages(df_plot, windows=(20, 50))

    # Fig con dos filas: velas y volumen
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.74, 0.26]
    )

    # Candlestick con alto contraste (relleno y borde)
    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot["Open"],
            high=df_plot["High"],
            low=df_plot["Low"],
            close=df_plot["Close"],
            name="OHLC",
            increasing_line_color="rgb(16,130,59)",
            increasing_fillcolor="rgba(16,130,59,0.9)",
            decreasing_line_color="rgb(200,30,30)",
            decreasing_fillcolor="rgba(200,30,30,0.9)",
            line=dict(width=1.25),
            whiskerwidth=0.3,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>" +
                "Open: %{open:.2f}<br>" +
                "High: %{high:.2f}<br>" +
                "Low: %{low:.2f}<br>" +
                "Close: %{close:.2f}<extra></extra>"
            )
        ),
        row=1, col=1
    )

    # Medias móviles (si se activan)
    if show_sma:
        if "SMA20" in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index, y=df_plot["SMA20"],
                    mode="lines", name="SMA 20",
                    line=dict(width=1.5, dash="solid"),
                    hovertemplate="SMA20: %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        if "SMA50" in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index, y=df_plot["SMA50"],
                    mode="lines", name="SMA 50",
                    line=dict(width=1.5, dash="dot"),
                    hovertemplate="SMA50: %{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )

    # Volumen coloreado por dirección
    if "Volume" in df_plot.columns:
        vol_colors = volume_colors(df_plot)
        fig.add_trace(
            go.Bar(
                x=df_plot.index, y=df_plot["Volume"],
                name="Volumen", marker_color=vol_colors,
                opacity=0.8,
                hovertemplate="Volumen: %{y:,}<extra></extra>"
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Volumen", row=2, col=1)

    # Layout y ejes
    fig.update_layout(
        title=f"{stonk} · {range_key}{suffix}",
        template="plotly_white",
        height=760,
        margin=dict(l=40, r=25, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(0,0,0,0.08)",
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
        yaxis=dict(
            title="Precio", showgrid=True, gridcolor="rgba(0,0,0,0.08)",
            zeroline=False
        ),
        dragmode="pan",
    )

    # Pequeñas mejoras de visibilidad
    fig.update_xaxes(showline=True, linewidth=1, linecolor="rgba(0,0,0,0.25)")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="rgba(0,0,0,0.25)")

    st.plotly_chart(fig, use_container_width=True)



